#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// TLlama read-only extension. Call only while the owner cannot free or reallocate buffers.
// Callbacks borrow leaf buffers and must not mutate their owners or retain these pointers.
// No backend allocation, synchronization, or tensor access is performed.
typedef void (*ggml_backend_buffer_visit_callback)(ggml_backend_buffer_t buffer, void * user_data);

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
