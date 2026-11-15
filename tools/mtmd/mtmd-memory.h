#pragma once

#include "mtmd.h"
#include "../../src/llama-memory-view.h"

struct mtmd_subcontext;

enum mtmd_memory_role {
    MTMD_MEMORY_ROLE_VISION,
    MTMD_MEMORY_ROLE_AUDIO,
    MTMD_MEMORY_ROLE_AUDIO_GENERATION,
};

using mtmd_memory_projector_visitor = std::function<void(const mtmd_subcontext *, mtmd_memory_role)>;

// Private TLlama API. Borrow existing projector subcontexts only; never initialize a probe context.
// Use on the inference owner thread, not from the HTTP handler.
MTMD_API bool mtmd_visit_projectors(const mtmd_context * ctx, const mtmd_memory_projector_visitor & visitor);
MTMD_API bool mtmd_projector_visit_buffers(const mtmd_subcontext * ctx, const llama_memory_buffer_visitor & visitor);
