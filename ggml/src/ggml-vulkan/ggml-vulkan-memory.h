#pragma once

#include "ggml-backend-memory.h"

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
    selected = {};
    auto memory = allocate();
    if (memory) {
        selected = candidate;
        selected.known = true;
    }
    return memory;
}

inline ggml_backend_memory_allocation_status ggml_vk_memory_visit(
        const ggml_vk_memory_allocation & selected, const void * domain, const void * identity,
        ggml_backend_dev_t device, bool pinned, uint64_t offset, uint64_t size,
        ggml_backend_memory_allocation_callback callback, void * user_data) {
    if (!selected.known) {
        return size == 0 ? GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE : GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    if (!domain || !identity || !device || offset > selected.size_bytes || size > selected.size_bytes - offset) {
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
    const ggml_backend_memory_binding binding = {
        GGML_BACKEND_MEMORY_BINDING_BUFFER_OFFSET | GGML_BACKEND_MEMORY_BINDING_ALLOCATION_OFFSET | GGML_BACKEND_MEMORY_BINDING_SIZE,
        0, offset, size,
    };
    callback(&allocation, &binding, user_data);
    return GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE;
}
