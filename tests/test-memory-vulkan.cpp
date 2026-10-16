#include "../ggml/src/ggml-vulkan/ggml-vulkan-memory.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

// Exercise the metadata wrapper used around real allocateMemory calls, without a Vulkan SDK/device.
struct allocation_failed {};
struct observation {
    ggml_backend_memory_allocation allocation;
    ggml_backend_memory_binding binding;
    std::string api;
    std::map<std::string, uint64_t> properties;
};
using observations = std::vector<observation>;

static void collect(const ggml_backend_memory_allocation * allocation, const ggml_backend_memory_binding * binding, void * data) {
    observation result = {*allocation, *binding, allocation->native_api, {}};
    for (size_t i = 0; i < allocation->property_count; ++i) {
        GGML_ASSERT(result.properties.emplace(allocation->properties[i].name, allocation->properties[i].value).second);
    }
    result.allocation.native_api = nullptr;
    result.allocation.properties = nullptr;
    static_cast<observations *>(data)->push_back(std::move(result));
}

static void test_retry() {
    // First choice is device-local. The successful second choice is host-visible/coherent.
    const ggml_vk_memory_allocation candidates[] = {{1, 0, 1, 1, 8192}, {4, 2, 6, 0, 8192}};
    ggml_vk_memory_allocation selected;
    size_t attempts = 0;
    for (const auto & candidate : candidates) {
        try {
            const auto memory = ggml_vk_allocate_memory_record(selected, candidate, [&] {
                if (++attempts == 1) { throw allocation_failed{}; }
                return 17;
            });
            GGML_ASSERT(memory == 17);
            break;
        } catch (const allocation_failed &) {
            GGML_ASSERT(!selected.known);
        }
    }
    GGML_ASSERT(attempts == 2 && selected.known);
    ggml_backend_device device = {};
    int domain = 0, identity = 0;
    observations result;
    GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, &device, false, 0, 4096, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(result.size() == 1);
    const auto & actual = result[0];
    GGML_ASSERT(actual.properties.at("vulkan.memory_type_index") == 4);
    GGML_ASSERT(actual.properties.at("vulkan.heap_index") == 2);
    GGML_ASSERT(actual.properties.at("vulkan.memory_property_flags") == 6);
    GGML_ASSERT(actual.properties.at("vulkan.heap_flags") == 0);
    GGML_ASSERT(actual.api == "vkAllocateMemory");
    GGML_ASSERT(actual.allocation.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE);
    GGML_ASSERT(actual.allocation.origin == GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED);
    GGML_ASSERT(actual.allocation.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_NON_DEVICE_LOCAL);
    GGML_ASSERT(actual.allocation.size_known && actual.allocation.size_bytes == 8192);
    GGML_ASSERT(actual.binding.size_bytes == 4096 && actual.binding.allocation_offset == 0);
    GGML_ASSERT(actual.binding.known_fields == 7);
}

static void test_import_and_failure() {
    ggml_vk_memory_allocation selected;
    const ggml_vk_memory_allocation candidate = {3, 1, 14, 0, 4096, true};
    GGML_ASSERT(ggml_vk_allocate_memory_record(selected, candidate, [] { return 9; }) == 9);
    ggml_backend_device device = {};
    int domain = 0, identity = 0;
    observations result;
    GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, &device, false, 0, 4096, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(result[0].allocation.origin == GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_IMPORTED);
    GGML_ASSERT(result[0].allocation.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST);
    GGML_ASSERT(result[0].allocation.size_bytes == 4096);
    // Failed imports must not publish either proposed or stale successful metadata.
    try {
        ggml_vk_allocate_memory_record(selected, candidate, []() -> int { throw allocation_failed{}; });
        GGML_ABORT("allocation should fail");
    } catch (const allocation_failed &) {
        GGML_ASSERT(!selected.known);
    }
    result.clear();
    GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, &device, false, 0, 4096, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(result.empty());
    GGML_ASSERT(ggml_vk_allocate_memory_record(selected, candidate, [] { return 0; }) == 0);
    GGML_ASSERT(!selected.known);
    // The native pinned allocator's CPU fallback uses this type, not failed Vulkan metadata.
    ggml_backend_buffer_ptr cpu(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 128));
    GGML_ASSERT(cpu);
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(cpu.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(result.empty() && ggml_backend_buffer_get_size(cpu.get()) == 128);
}

static void test_pinned_alias_and_empty() {
    ggml_vk_memory_allocation selected;
    const ggml_vk_memory_allocation candidate = {0, 0, 7, 1, 8192}; // DEVICE_LOCAL and HOST_VISIBLE are independent
    GGML_ASSERT(ggml_vk_allocate_memory_record(selected, candidate, [] { return 1; }) == 1);
    ggml_backend_device device = {};
    int domain = 0, identity = 0;
    observations result;
    for (const uint64_t offset : {uint64_t(32), uint64_t(512)}) {
        GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, &device, true, offset, 128, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    }
    GGML_ASSERT(result.size() == 2);
    for (const auto & actual : result) {
        GGML_ASSERT(actual.allocation.identity_domain == &domain && actual.allocation.identity == &identity);
        GGML_ASSERT(actual.allocation.device == &device && actual.allocation.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED);
        GGML_ASSERT(actual.allocation.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL);
        GGML_ASSERT(actual.properties.at("vulkan.memory_property_flags") == 7);
        GGML_ASSERT(actual.allocation.size_bytes == 8192 && actual.binding.size_bytes == 128);
    }
    GGML_ASSERT(result[0].binding.allocation_offset == 32 && result[1].binding.allocation_offset == 512);
    result.clear();
    GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, &device, true, 8193, 1, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, &device, true, 8192, UINT64_MAX, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(ggml_vk_memory_visit(selected, &domain, &identity, nullptr, true, 0, 128, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(result.empty());
    selected = {};
    GGML_ASSERT(ggml_vk_memory_visit(selected, nullptr, nullptr, nullptr, false, 0, 0, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(result.empty());
}

int main() {
    ggml_backend_load_all();
    test_retry();
    test_import_and_failure();
    test_pinned_alias_and_empty();
    std::puts("Vulkan memory metadata PASSED");
}
