#pragma once

#include "../ggml-backend-memory-record.h"

#include <utility>

// SDK-free metadata logic, shared by native allocation attempts and CPU regression tests.
struct ggml_vk_memory_allocation {
    uint32_t memory_type_index = 0;
    uint32_t heap_index = 0;
    uint32_t property_flags = 0;
    uint32_t heap_flags = 0;
    uint64_t size_bytes = 0;
    bool imported = false;
    bool known = false;
};

// Vulkan fixes this bit value. The native translation unit checks it against the SDK.
constexpr uint32_t ggml_vk_memory_device_local_bit = 0x1;

template<typename Allocate>
auto ggml_vk_allocate_memory_record(
        ggml_vk_memory_allocation & selected, const ggml_vk_memory_allocation & candidate, Allocate allocate) -> decltype(allocate()) {
    return ggml_backend_memory_record_attempt(selected, candidate, std::move(allocate),
        [](const auto & memory) { return bool(memory); });
}

// Caller holds the host-allocation lock. Copy existing views without creating imports.
template<typename Entries, typename Views>
bool ggml_vk_memory_host_views(const Entries & entries, const void * ptr, size_t size, Views & views, size_t & offset) {
    views.clear();
    offset = 0;
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (const auto & entry : entries) {
        const auto base = reinterpret_cast<uintptr_t>(entry.base);
        if (address >= base && address - base < entry.size) {
            const size_t candidate_offset = address - base;
            if (size > entry.size - candidate_offset) {
                return false;
            }
            for (const auto & view : entry.views) {
                if (view) {
                    views.push_back(view);
                }
            }
            offset = candidate_offset;
            return !views.empty();
        }
    }
    return false;
}

inline ggml_backend_memory_allocation_status ggml_vk_memory_visit(
        const ggml_vk_memory_allocation & selected, const void * domain, const void * identity,
        ggml_backend_dev_t device, bool pinned, uint64_t offset, uint64_t size,
        ggml_backend_memory_allocation_callback callback, void * user_data) {
    if (!selected.known) {
        return size == 0 ? GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE : GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    if (!device) {
        return GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    const ggml_backend_memory_allocation_property properties[] = {
        {"vulkan.memory_type_index", selected.memory_type_index},
        {"vulkan.heap_index", selected.heap_index},
        {"vulkan.memory_property_flags", selected.property_flags},
        {"vulkan.heap_flags", selected.heap_flags},
    };
    ggml_backend_memory_allocation allocation = {};
    allocation.identity_domain = domain;
    allocation.identity = identity;
    allocation.device = device;
    allocation.kind = selected.imported ? GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST :
        pinned ? GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED : GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE;
    allocation.origin = selected.imported ? GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_IMPORTED : GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED;
    allocation.locality = selected.property_flags & ggml_vk_memory_device_local_bit ?
        GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL : GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_NON_DEVICE_LOCAL;
    allocation.native_api = "vkAllocateMemory";
    allocation.size_known = true;
    allocation.size_bytes = selected.size_bytes;
    allocation.properties = properties;
    allocation.property_count = sizeof(properties) / sizeof(properties[0]);
    return ggml_backend_memory_emit_range(allocation, size, 0, offset, size, callback, user_data);
}
