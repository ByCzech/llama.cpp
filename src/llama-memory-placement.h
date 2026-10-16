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

enum llama_memory_assignment_source {
    LLAMA_MEMORY_ASSIGNMENT_UNKNOWN,
    LLAMA_MEMORY_ASSIGNMENT_SELECTION,
    LLAMA_MEMORY_ASSIGNMENT_OVERRIDE,
    LLAMA_MEMORY_ASSIGNMENT_MMAP,
    LLAMA_MEMORY_ASSIGNMENT_LAZY,
};

struct llama_memory_tensor_assignment {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_dev_t selection_device = nullptr; // supports_op candidate, not necessarily the assigned device
    ggml_backend_buffer_type_t buffer_type = nullptr; // final loader choice, checked against the actual allocation on visit
    llama_memory_assignment_source source = LLAMA_MEMORY_ASSIGNMENT_UNKNOWN;
};

struct llama_memory_tensor_role {
    const ggml_tensor * tensor;
    llama_memory_model_part part;
    int32_t layer;
    llama_memory_tensor_assignment assignment = {};
};

struct llama_memory_placement_view {
    llama_memory_model_part part;
    int32_t layer; // native block index, or -1 for non-layer weights
    ggml_backend_buffer_t buffer; // borrowed; NULL if no buffer is known
    bool known; // both the role and the actual buffer are known
    llama_memory_tensor_assignment assignment = {}; // device is NULL when the final assignment cannot be proved
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
        const LLM_TN_IMPL & name, bool duplicated, int32_t n_layer, int32_t n_layer_all,
        const llama_memory_tensor_assignment & assignment = {});

bool llama_memory_visit_tensor_roles(
        const std::vector<llama_memory_tensor_role> & roles,
        const std::vector<std::pair<std::string, ggml_tensor *>> & tensors,
        int32_t n_layer, int32_t n_layer_all, const llama_memory_placement_visitor & visitor);

// Logical storage placement of loaded weights, not physical residency or execution profiling.
// False can accompany partial results. Shared tensors can occur under more than one role.
LLAMA_API bool llama_model_visit_placement(const llama_model * model, const llama_memory_placement_visitor & visitor);
