#pragma once

#include "ggml-backend-memory.h"

// Allocation paths, never inspection paths, invoke the native attempt. A record
// with a bool `known` member describes only a successful attempt, including fallback.
template<typename Record, typename Attempt, typename Succeeded>
auto ggml_backend_memory_record_attempt(
        Record & selected, const Record & candidate, Attempt attempt, Succeeded succeeded) -> decltype(attempt()) {
    const Record description = candidate; // candidate may alias selected
    selected = {};
    auto result = attempt();
    if (succeeded(result)) {
        selected = description;
        selected.known = true;
    }
    return result;
}

// Emit one proved range. COMPLETE validates this relationship, not aggregate
// coverage of the whole buffer. Native identities are supplied by the owner.
inline ggml_backend_memory_allocation_status ggml_backend_memory_emit_range(
        const ggml_backend_memory_allocation & allocation, uint64_t buffer_size,
        uint64_t buffer_offset, uint64_t allocation_offset, uint64_t size,
        ggml_backend_memory_allocation_callback callback, void * user_data) noexcept {
    if (!callback || !allocation.identity_domain || !allocation.identity || !allocation.size_known ||
        (allocation.property_count && !allocation.properties) ||
        (allocation.locality != GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN && !allocation.device) ||
        buffer_offset > buffer_size || size > buffer_size - buffer_offset ||
        allocation_offset > allocation.size_bytes || size > allocation.size_bytes - allocation_offset) {
        return GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    const ggml_backend_memory_binding binding = {
        GGML_BACKEND_MEMORY_BINDING_BUFFER_OFFSET | GGML_BACKEND_MEMORY_BINDING_ALLOCATION_OFFSET |
            GGML_BACKEND_MEMORY_BINDING_SIZE,
        buffer_offset, allocation_offset, size,
    };
    try {
        callback(&allocation, &binding, user_data);
    } catch (...) {
        // As for any UNAVAILABLE result, the caller must discard all callbacks.
        return GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    }
    return GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE;
}
