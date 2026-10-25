#pragma once

#include "../ggml-backend-memory-record.h"

#include <type_traits>

struct ggml_sycl_memory_record {
    bool known = false;
    ggml_backend_memory_allocation_kind kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_UNKNOWN;
    const char * api = nullptr;
    uint64_t size = 0;
    ggml_backend_dev_t device = nullptr;
    std::shared_ptr<const void> context;
    bool ordinal_known = false;
    uint64_t ordinal = 0;
    bool device_type_known = false;
    bool discrete = false;
};

inline ggml_sycl_memory_record ggml_sycl_memory_candidate(
        ggml_backend_memory_allocation_kind kind, const char * api, uint64_t size) {
    ggml_sycl_memory_record result;
    result.kind = kind;
    result.api = api;
    result.size = size;
    return result;
}

template<typename Queue>
void ggml_sycl_memory_capture_context(ggml_sycl_memory_record * record, const Queue & queue) noexcept {
    if (!record || !record->known) { return; }
    try {
        using Context = typename std::decay<decltype(queue.get_context())>::type;
        record->context = std::make_shared<Context>(queue.get_context());
    } catch (...) {
        // Native allocation succeeded. Missing diagnostic context means unavailable coverage.
        record->context.reset();
    }
}

inline const void * ggml_sycl_memory_system_domain() {
    static const int domain = 0;
    return &domain;
}

inline ggml_backend_memory_allocation_status ggml_sycl_memory_visit(
        const ggml_sycl_memory_record & record, const void * identity, uint64_t buffer_size,
        ggml_backend_memory_allocation_callback callback, void * user_data) noexcept {
    if (!record.known || !record.api || !identity) {
        return GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    const bool system = record.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM;
    if (!system && !record.context) {
        return GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    ggml_backend_memory_allocation a = {};
    a.identity_domain = system ? ggml_sycl_memory_system_domain() : record.context.get();
    a.identity = identity;
    a.device = record.device;
    a.kind = record.kind;
    a.origin = GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED;
    a.locality = record.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE && record.device &&
        record.device_type_known && record.discrete ? GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL :
        GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN;
    a.native_api = record.api;
    a.size_known = true;
    a.size_bytes = record.size;
    ggml_backend_memory_allocation_property properties[2];
    if (record.ordinal_known) {
        properties[a.property_count++] = {"level_zero.memory_ordinal", record.ordinal};
    }
    if (record.device && record.device_type_known) {
        properties[a.property_count++] = {"level_zero.device_integrated", uint64_t(!record.discrete)};
    }
    a.properties = properties;
    return ggml_backend_memory_emit_range(a, buffer_size, 0, 0, buffer_size, callback, user_data);
}

using ggml_sycl_memory_host_records = ggml_backend_memory_record_registry<ggml_sycl_memory_record>;
