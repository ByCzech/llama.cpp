#include "../ggml/src/ggml-backend-memory-record.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Native API results and counts are injected into the same helper used by the
// production Vulkan allocator. No SDK/driver or accelerator memory is involved.
struct metadata {
    bool known = false;
    ggml_backend_memory_allocation_kind kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_UNKNOWN;
    const char * api = nullptr;
    uint64_t size = 0;
};

static void test_attempts() {
    metadata selected;
    size_t native_calls = 0;
    auto attempt = [&](const metadata & candidate, int status) {
        return ggml_backend_memory_record_attempt(selected, candidate,
            [&] { ++native_calls; return status; }, [](int result) { return result == 0; });
    };
    const metadata managed = {false, GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED, "hipMallocManaged", 4096};
    const metadata device = {false, GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE, "hipMalloc", 4096};
    GGML_ASSERT(attempt(managed, 0) == 0 && selected.known);
    GGML_ASSERT(selected.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED && !managed.known);
    GGML_ASSERT(attempt(managed, 7) == 7 && !selected.known && selected.api == nullptr && selected.size == 0);
    GGML_ASSERT(attempt(device, 0) == 0 && selected.known);
    GGML_ASSERT(selected.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE && std::strcmp(selected.api, "hipMalloc") == 0);
    GGML_ASSERT(attempt(managed, 7) == 7 && !selected.known);
    GGML_ASSERT(attempt(device, 2) == 2 && !selected.known && selected.api == nullptr);
    GGML_ASSERT(native_calls == 5);

    // Reuse of the metadata slot cannot retain a failed attempt's provenance.
    selected = managed;
    selected.known = true;
    try {
        ggml_backend_memory_record_attempt(selected, device,
            [&]() -> int { ++native_calls; throw std::runtime_error("allocation failed"); },
            [](int status) { return status == 0; });
        GGML_ABORT("allocation exception was lost");
    } catch (const std::runtime_error &) {
        GGML_ASSERT(!selected.known && selected.api == nullptr && selected.size == 0);
    }
    GGML_ASSERT(native_calls == 6);

    // Vulkan-like nullable and move-only native handles use a different success predicate.
    auto handle = ggml_backend_memory_record_attempt(selected, device,
        [] { return std::unique_ptr<int>(new int(12)); }, [](const auto & result) { return bool(result); });
    GGML_ASSERT(handle && *handle == 12 && selected.known);
    handle = ggml_backend_memory_record_attempt(selected, managed,
        [] { return std::unique_ptr<int>(); }, [](const auto & result) { return bool(result); });
    GGML_ASSERT(!handle && !selected.known);

    // Known zero is not an absent allocation, and an aliased candidate is safe.
    metadata zero = {false, GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM, "aligned_alloc", 0};
    GGML_ASSERT(attempt(zero, 0) == 0 && selected.known && selected.size == 0);
    GGML_ASSERT(attempt(selected, 0) == 0 && selected.known && std::strcmp(selected.api, "aligned_alloc") == 0);
}

struct observation {
    const void * domain;
    const void * identity;
    ggml_backend_dev_t device;
    ggml_backend_memory_allocation_kind kind;
    ggml_backend_memory_allocation_locality locality;
    uint64_t allocation_size;
    ggml_backend_memory_binding binding;
    std::string api;
};

static void collect(const ggml_backend_memory_allocation * a, const ggml_backend_memory_binding * b, void * opaque) {
    auto & seen = *static_cast<std::vector<observation> *>(opaque);
    seen.push_back({a->identity_domain, a->identity, a->device, a->kind, a->locality,
                   a->size_bytes, *b, a->native_api ? a->native_api : ""});
}

static void test_ranges_and_identity() {
    int first_domain = 0, second_domain = 0, allocation_token = 0;
    ggml_backend_device first_device = {}, second_device = {};
    ggml_backend_memory_allocation a = {};
    a.identity_domain = &first_domain;
    a.identity = &allocation_token;
    a.device = &first_device;
    a.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE;
    a.origin = GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED;
    a.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL;
    a.native_api = "cudaMalloc";
    a.size_known = true;
    a.size_bytes = 4096;
    std::vector<observation> seen;
    auto emit = [&](uint64_t capacity, uint64_t buffer_offset, uint64_t allocation_offset, uint64_t size) {
        return ggml_backend_memory_emit_range(a, capacity, buffer_offset, allocation_offset, size, collect, &seen);
    };
    GGML_ASSERT(emit(512, 0, 128, 512) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(emit(1024, 32, 3072, 992) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(seen.size() == 2 && seen[0].identity == seen[1].identity && seen[0].domain == seen[1].domain);
    GGML_ASSERT(seen[0].allocation_size == 4096 && seen[1].binding.buffer_offset == 32);
    GGML_ASSERT(seen[0].binding.known_fields == 7 && seen[0].binding.allocation_offset == 128);
    GGML_ASSERT(seen[1].binding.allocation_offset == 3072 && seen[1].binding.size_bytes == 992);
    GGML_ASSERT(seen[0].api == "cudaMalloc");

    // The same address token on another domain/device must remain distinct.
    a.identity_domain = &second_domain;
    a.device = &second_device;
    GGML_ASSERT(emit(256, 0, 0, 256) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(seen[2].identity == seen[0].identity && seen[2].domain != seen[0].domain);
    GGML_ASSERT(seen[2].device != seen[0].device);

    const auto old_size = seen.size();
    GGML_ASSERT(emit(10, 11, 0, 0) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(emit(10, 9, 0, 2) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(emit(512, 0, 4097, 0) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(emit(512, 0, 4090, 7) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    const auto max = std::numeric_limits<uint64_t>::max();
    GGML_ASSERT(emit(max, max - 1, 0, 2) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    a.size_bytes = max;
    GGML_ASSERT(emit(2, 0, max - 1, 2) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(seen.size() == old_size); // Reject before the first callback.
    GGML_ASSERT(emit(1, 0, max - 1, 1) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(emit(max, max, max, 0) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);

    a.size_bytes = 0;
    GGML_ASSERT(emit(0, 0, 0, 0) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(seen.back().allocation_size == 0 && seen.back().binding.size_bytes == 0);
    GGML_ASSERT(emit(1, 0, 0, 1) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);

    // Managed/shared or portable host memory need not have a proved device.
    a.size_bytes = 4096;
    a.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN;
    a.device = nullptr;
    a.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED;
    a.native_api = "hipMallocManaged";
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(!seen.back().device && seen.back().locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
    GGML_ASSERT(seen.back().kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED);
    a.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_SHARED;
    a.native_api = "sycl::malloc_shared";
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(seen.back().kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_SHARED);
    a.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL;
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    a.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN;
    a.size_known = false;
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    a.size_known = true;
    a.identity = nullptr;
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    a.identity = &allocation_token;
    a.identity_domain = nullptr;
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    a.identity_domain = &first_domain;
    a.property_count = 1;
    GGML_ASSERT(emit(100, 0, 80, 100) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    a.property_count = 0;
    GGML_ASSERT(ggml_backend_memory_emit_range(a, 100, 0, 80, 100, nullptr, &seen) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);

    size_t calls = 0;
    auto throwing = [](const ggml_backend_memory_allocation *, const ggml_backend_memory_binding *, void * opaque) {
        ++*static_cast<size_t *>(opaque);
        throw std::runtime_error("collector failed");
    };
    GGML_ASSERT(ggml_backend_memory_emit_range(a, 100, 0, 80, 100, throwing, &calls) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(calls == 1);
}

int main() {
    test_attempts();
    test_ranges_and_identity();
    std::puts("Allocation attempts, fallback provenance, bounded views and identity domains passed");
}
