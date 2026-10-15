#pragma once

#include "llama.h"
#include "ggml-backend-memory.h"

#include <functional>

// Private TLlama inspection API. Visit only on the owner thread at a safe point.
// Pointers are borrowed for the call; the receiver must copy metadata, not retain buffers.
enum llama_memory_component {
    LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS,
    LLAMA_MEMORY_COMPONENT_KV_CACHE,
    LLAMA_MEMORY_COMPONENT_RECURRENT_STATE,
    LLAMA_MEMORY_COMPONENT_STATE_OTHER,
    LLAMA_MEMORY_COMPONENT_COMPUTE,
    LLAMA_MEMORY_COMPONENT_OUTPUT,
    LLAMA_MEMORY_COMPONENT_MOE_CACHE,
    LLAMA_MEMORY_COMPONENT_CONTROL_VECTOR,
    LLAMA_MEMORY_COMPONENT_ADAPTER_WEIGHTS,
    LLAMA_MEMORY_COMPONENT_PROJECTOR_WEIGHTS,
    LLAMA_MEMORY_COMPONENT_PROJECTOR_COMPUTE,
};

enum llama_memory_backing {
    LLAMA_MEMORY_BACKING_UNKNOWN,
    LLAMA_MEMORY_BACKING_ALLOCATED,
    LLAMA_MEMORY_BACKING_MMAP,
};

struct llama_memory_buffer_view {
    ggml_backend_buffer_t buffer;
    llama_memory_component component;
    llama_memory_backing backing;
};

using llama_memory_buffer_visitor = std::function<void(const llama_memory_buffer_view &)>;

inline void llama_memory_visit_buffer(
        ggml_backend_buffer_t buffer, llama_memory_component component,
        const llama_memory_buffer_visitor & visitor,
        llama_memory_backing backing = LLAMA_MEMORY_BACKING_ALLOCATED) {
    struct visit_data {
        const llama_memory_buffer_visitor & visitor;
        llama_memory_component component;
        llama_memory_backing backing;
    } data = {visitor, component, backing};
    ggml_backend_buffer_visit(buffer, [](ggml_backend_buffer_t leaf, void * user_data) {
        const auto & data = *(const visit_data *) user_data;
        data.visitor({leaf, data.component, data.backing});
    }, &data);
}

// Model weights only; adapters are separate owners and are not included here.
// False means no real allocation (for example no_alloc) or an unsupported source.
LLAMA_API bool llama_model_visit_buffers(const llama_model * model, const llama_memory_buffer_visitor & visitor);

// Context state, compute, output, control vectors, MoE cache and saved device state.
// Does not revisit model weights. No state is a valid case (for example embeddings).
// False can accompany partial results if the state implementation is unsupported.
LLAMA_API bool llama_context_visit_buffers(const llama_context * ctx, const llama_memory_buffer_visitor & visitor);

// Loaded adapters are resident owners even when they are not currently selected by a context.
using llama_memory_lora_visitor = std::function<void(const llama_adapter_lora *)>;
LLAMA_API bool llama_model_visit_loras(const llama_model * model, const llama_memory_lora_visitor & visitor);
LLAMA_API bool llama_adapter_lora_visit_buffers(const llama_adapter_lora * adapter, const llama_memory_buffer_visitor & visitor);
