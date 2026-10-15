#include "llama-memory-placement.h"
#include "llama-arch.h"

#include <set>

void llama_memory_record_tensor_role(
        std::vector<llama_memory_tensor_role> & roles, const ggml_tensor * tensor,
        const LLM_TN_IMPL & name, bool duplicated, int32_t n_layer, int32_t n_layer_all) {
    if (!tensor) {
        return;
    }
    auto tensor_kind = name.tensor;
    if (tensor_kind == LLM_TENSOR_TOKEN_EMBD && duplicated) {
        tensor_kind = LLM_TENSOR_OUTPUT;
    }
    const auto layer = llm_tensor_info_for(tensor_kind).layer;
    auto part = LLAMA_MEMORY_MODEL_PART_OTHER;
    int32_t index = -1;
    if (layer == LLM_TENSOR_LAYER_INPUT) {
        part = LLAMA_MEMORY_MODEL_PART_INPUT;
    } else if (layer == LLM_TENSOR_LAYER_OUTPUT) {
        part = LLAMA_MEMORY_MODEL_PART_OUTPUT;
    } else if (name.bid >= 0 && name.bid < n_layer_all) {
        part = name.bid < n_layer ? LLAMA_MEMORY_MODEL_PART_REPEATING : LLAMA_MEMORY_MODEL_PART_MTP;
        index = name.bid;
    }
    roles.push_back({tensor, part, index});
}

bool llama_memory_visit_tensor_roles(
        const std::vector<llama_memory_tensor_role> & roles,
        const std::vector<std::pair<std::string, ggml_tensor *>> & tensors,
        int32_t n_layer, int32_t n_layer_all, const llama_memory_placement_visitor & visitor) {
    bool complete = true;
    std::set<const ggml_tensor *> classified;
    std::set<int32_t> layers;
    for (const auto & role : roles) {
        auto * buffer = llama_memory_tensor_buffer(role.tensor);
        visitor({role.part, role.layer, buffer, buffer != nullptr});
        complete = buffer != nullptr && complete;
        classified.insert(role.tensor);
        if (role.layer >= 0) {
            layers.insert(role.layer);
        }
    }
    for (const auto & tensor : tensors) {
        if (classified.count(tensor.second) == 0) {
            visitor({LLAMA_MEMORY_MODEL_PART_OTHER, -1, llama_memory_tensor_buffer(tensor.second), false});
            complete = false;
        }
    }
    // Reused blocks or new loading paths may bypass tensor creation. Keep these layers unknown.
    for (int32_t layer = 0; layer < n_layer_all; ++layer) {
        if (layers.count(layer) == 0) {
            const auto part = layer < n_layer ? LLAMA_MEMORY_MODEL_PART_REPEATING : LLAMA_MEMORY_MODEL_PART_MTP;
            visitor({part, layer, nullptr, false});
            complete = false;
        }
    }
    return complete;
}
