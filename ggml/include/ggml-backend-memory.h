#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// TLlama read-only extension. Call only while the owner cannot free or reallocate buffers.
// Callbacks borrow leaf buffers and must not mutate their owners or retain these pointers.
// No backend allocation, synchronization, or tensor access is performed.
typedef void (*ggml_backend_buffer_visit_callback)(ggml_backend_buffer_t buffer, void * user_data);

enum ggml_backend_memory_buffer_flag {
    GGML_BACKEND_MEMORY_BUFFER_OPAQUE = 1,
    GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED = 2,
};

// Optional registry procedure "tllama_buffer_memory_flags". It must only read native metadata.
typedef uint32_t (*ggml_backend_memory_buffer_flags_t)(ggml_backend_buffer_t buffer);

GGML_API uint32_t ggml_backend_memory_buffer_flags(ggml_backend_buffer_t buffer);
GGML_API ggml_backend_dev_t ggml_backend_memory_buffer_device(ggml_backend_buffer_t buffer);

// Optional allocation inspection has its own coverage, independent of buffer sizes and placement.
enum ggml_backend_memory_allocation_status {
    GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED,
    GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE,
    GGML_BACKEND_MEMORY_ALLOCATION_PARTIAL,
    GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE,
};

enum ggml_backend_memory_allocation_kind {
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_UNKNOWN,
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE,
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST,
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED,
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED,
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_SHARED,
    GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM,
};

enum ggml_backend_memory_allocation_origin {
    GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_UNKNOWN,
    GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED,
    GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_IMPORTED,
    GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_REGISTERED,
};

enum ggml_backend_memory_allocation_locality {
    GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN,
    GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL,
    GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_NON_DEVICE_LOCAL,
};

// Optional unsigned native attributes, with names scoped to the native API (see docs).
// Presence proves availability: a zero value is valid, and absence does not mean zero.
struct ggml_backend_memory_allocation_property {
    const char * name;
    uint64_t value;
};

struct ggml_backend_memory_allocation {
    // Both identities must be non-NULL and stable while the native allocation is alive.
    // Deduplicate the pair, never the provider, device name, heap index or identity alone.
    const void * identity_domain;
    const void * identity;
    ggml_backend_dev_t device; // proved allocation association, or NULL; not the buffer provider or execution device
    enum ggml_backend_memory_allocation_kind kind;
    enum ggml_backend_memory_allocation_origin origin;
    enum ggml_backend_memory_allocation_locality locality; // relative to device; not measured physical residency
    const char * native_api; // successful allocation/import/registration API, or NULL
    bool size_known;
    uint64_t size_bytes; // native allocation span, valid only when size_known
    const struct ggml_backend_memory_allocation_property * properties;
    size_t property_count;
};

enum ggml_backend_memory_binding_field {
    GGML_BACKEND_MEMORY_BINDING_BUFFER_OFFSET     = 1,
    GGML_BACKEND_MEMORY_BINDING_ALLOCATION_OFFSET = 2,
    GGML_BACKEND_MEMORY_BINDING_SIZE              = 4,
};

// One proved relationship from the inspected buffer to an allocation. Emit each relationship once.
// Offsets use their respective buffer/allocation address spaces; only known_fields are valid.
struct ggml_backend_memory_binding {
    uint32_t known_fields;
    uint64_t buffer_offset;
    uint64_t allocation_offset;
    uint64_t size_bytes;
};

// Records and property arrays are borrowed only for this callback. Copy data needed after return.
// Identities and devices may be used only while owners remain alive; never expose them on the wire.
typedef void (*ggml_backend_memory_allocation_callback)(
        const struct ggml_backend_memory_allocation * allocation,
        const struct ggml_backend_memory_binding * binding, void * user_data);

// Optional registry procedure "tllama_buffer_memory_allocations_v1"; no required backend vtable change.
// Inspect existing metadata only: no SDK queries, backend allocation, synchronization, device/context changes or logging.
// COMPLETE may emit no records for a known empty buffer; PARTIAL may emit a proved subset.
// UNSUPPORTED/UNAVAILABLE results must be discarded even if callbacks ran. Never throw across this interface.
// The v1 procedure name fixes this ABI; incompatible records require a new versioned procedure.
typedef enum ggml_backend_memory_allocation_status (*ggml_backend_memory_buffer_allocations_t)(
        ggml_backend_buffer_t buffer, ggml_backend_memory_allocation_callback callback, void * user_data);

// Inspect a leaf or backend-specific composite. Generic multi-buffers return UNSUPPORTED: visit leaves first.
// NULL buffer/callback returns UNAVAILABLE. No fallback from buffer name, size, flags or host visibility.
GGML_API enum ggml_backend_memory_allocation_status ggml_backend_memory_buffer_allocations(
        ggml_backend_buffer_t buffer, ggml_backend_memory_allocation_callback callback, void * user_data);

// Visit a leaf, or the leaves of a ggml multi-buffer. A NULL buffer has no leaves.
// Backend-specific composite buffers remain opaque; this does not enumerate their allocations.
GGML_API void ggml_backend_buffer_visit(
        ggml_backend_buffer_t buffer, ggml_backend_buffer_visit_callback callback, void * user_data);

// Visit allocated chunks only. Shared allocator slots are visited once per call.
// Callers must deduplicate identities when combining visits from different owners.
GGML_API void ggml_gallocr_visit_buffers(
        ggml_gallocr_t galloc, ggml_backend_buffer_visit_callback callback, void * user_data);
GGML_API void ggml_backend_sched_visit_buffers(
        ggml_backend_sched_t sched, ggml_backend_buffer_visit_callback callback, void * user_data);

#ifdef __cplusplus
}
#endif
