#pragma once

#include "../ggml-backend-memory-record.h"

#include <map>
#include <memory>
#include <mutex>

// SDK-free metadata and control flow used by both CUDA and HIP allocation paths.
enum class ggml_cuda_memory_api { cuda, hip };

struct ggml_cuda_memory_record {
    bool known = false;
    ggml_cuda_memory_api api = ggml_cuda_memory_api::cuda;
    ggml_backend_memory_allocation_kind kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_UNKNOWN;
    uint64_t size = 0;
    ggml_backend_dev_t device = nullptr;
    bool integrated_known = false;
    bool integrated = false;
};

inline ggml_cuda_memory_record ggml_cuda_memory_candidate(
        ggml_cuda_memory_api api, ggml_backend_memory_allocation_kind kind, uint64_t size) {
    ggml_cuda_memory_record result;
    result.api = api;
    result.kind = kind;
    result.size = size;
    return result;
}

template<typename Device, typename Managed, typename Success, typename Advise, typename Fallback>
auto ggml_cuda_memory_allocate(
        ggml_cuda_memory_record & selected, ggml_cuda_memory_api api, uint64_t size, bool unified,
        Device allocate_device, Managed allocate_managed, Success success, Advise advise, Fallback fallback)
        -> decltype(allocate_device()) {
    auto device = [&] {
        return ggml_backend_memory_record_attempt(selected,
            ggml_cuda_memory_candidate(api, GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE, size),
            allocate_device, success);
    };
    if (!unified) {
        return device();
    }
    auto status = ggml_backend_memory_record_attempt(selected,
        ggml_cuda_memory_candidate(api, GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED, size),
        allocate_managed, success);
    if (success(status)) {
        advise();
    } else if (fallback(status)) {
        return device();
    }
    return status;
}

inline const char * ggml_cuda_memory_native_api(const ggml_cuda_memory_record & record) {
    switch (record.kind) {
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE:
            return record.api == ggml_cuda_memory_api::hip ? "hipMalloc" : "cudaMalloc";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED:
            return record.api == ggml_cuda_memory_api::hip ? "hipMallocManaged" : "cudaMallocManaged";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST:
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED:
            return record.api == ggml_cuda_memory_api::hip ? "hipHostMalloc" : "cudaMallocHost";
        default:
            return nullptr;
    }
}

inline ggml_backend_memory_allocation_status ggml_cuda_memory_visit(
        const ggml_cuda_memory_record & record, const void * domain, const void * identity,
        uint64_t buffer_size, ggml_backend_memory_allocation_callback callback, void * user_data) noexcept {
    if (!record.known || !ggml_cuda_memory_native_api(record)) {
        return GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    ggml_backend_memory_allocation a = {};
    a.identity_domain = domain;
    a.identity = identity;
    a.device = record.device;
    a.kind = record.kind;
    a.origin = GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED;
    a.locality = record.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE && record.device &&
        record.integrated_known && !record.integrated ? GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL :
        GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN;
    a.native_api = ggml_cuda_memory_native_api(record);
    a.size_known = true;
    a.size_bytes = record.size; // logical native API span, not allocator/driver reserved capacity
    const ggml_backend_memory_allocation_property property = {
        record.api == ggml_cuda_memory_api::hip ? "hip.device_integrated" : "cuda.device_integrated",
        uint64_t(record.integrated),
    };
    if (record.device && record.integrated_known) {
        a.properties = &property;
        a.property_count = 1;
    }
    return ggml_backend_memory_emit_range(a, buffer_size, 0, 0, buffer_size, callback, user_data);
}

using ggml_cuda_memory_host_records = ggml_backend_memory_record_registry<ggml_cuda_memory_record>;
