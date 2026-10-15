#pragma once

#include "llama-memory-view.h"

#include <string>
#include <utility>
#include <vector>

struct LLM_TN_IMPL;

enum llama_memory_model_part {
    LLAMA_MEMORY_MODEL_PART_INPUT,
    LLAMA_MEMORY_MODEL_PART_OUTPUT,
    LLAMA_MEMORY_MODEL_PART_REPEATING,
    LLAMA_MEMORY_MODEL_PART_MTP,
    LLAMA_MEMORY_MODEL_PART_OTHER,
};

struct llama_memory_tensor_role {
    const ggml_tensor * tensor;
    llama_memory_model_part part;
    int32_t layer;
};

struct llama_memory_placement_view {
    llama_memory_model_part part;
    int32_t layer; // native block index, or -1 for non-layer weights
    ggml_backend_buffer_t buffer; // borrowed; NULL if no buffer is known
    bool known; // both the role and the actual buffer are known
};

using llama_memory_placement_visitor = std::function<void(const llama_memory_placement_view &)>;

inline ggml_backend_buffer_t llama_memory_tensor_buffer(const ggml_tensor * tensor) {
    for (; tensor; tensor = tensor->view_src) {
        if (tensor->buffer) {
            return tensor->buffer;
        }
    }
    return nullptr;
}

void llama_memory_record_tensor_role(
        std::vector<llama_memory_tensor_role> & roles, const ggml_tensor * tensor,
        const LLM_TN_IMPL & name, bool duplicated, int32_t n_layer, int32_t n_layer_all);

bool llama_memory_visit_tensor_roles(
        const std::vector<llama_memory_tensor_role> & roles,
        const std::vector<std::pair<std::string, ggml_tensor *>> & tensors,
        int32_t n_layer, int32_t n_layer_all, const llama_memory_placement_visitor & visitor);

// Logical storage placement of loaded weights, not physical residency or execution profiling.
// False can accompany partial results. Shared tensors can occur under more than one role.
LLAMA_API bool llama_model_visit_placement(const llama_model * model, const llama_memory_placement_visitor & visitor);
