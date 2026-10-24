#pragma once

#include "llama-memory-placement.h"

struct llama_memory_weight_range {
    size_t offset;
    size_t size;
};

struct llama_memory_weight_storage_view {
    llama_memory_placement_view placement;
    std::vector<llama_memory_weight_range> ranges; // sorted, disjoint buffer-relative byte ranges
    const char * missing_reason = nullptr; // NULL means exact ranges, including a known empty tensor
};

// Private owner-thread metadata inspection. No tensor access, allocation inspection or SDK calls.
// Buffer identity bounds deduplication; equal offsets or addresses in different buffers are not aliases.
// A non-NULL missing_reason invalidates all ranges, not just a suffix. Placement has independent coverage.
LLAMA_API llama_memory_weight_storage_view llama_memory_resolve_weight_storage(const llama_memory_placement_view & placement);

using llama_memory_weight_storage_visitor = std::function<void(const llama_memory_weight_storage_view &)>;

// Model weights only; false can accompany usable results. Does not sum roles or filter input weights.
// Tensor and buffer pointers are borrowed for this call; copy metadata and never serialize addresses.
LLAMA_API bool llama_model_visit_weight_storage(const llama_model * model, const llama_memory_weight_storage_visitor & visitor);
