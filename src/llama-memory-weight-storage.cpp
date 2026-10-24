#include "llama-memory-weight-storage.h"

#include <algorithm>
#include <array>
#include <limits>

namespace {

constexpr size_t max_ranges = 4096;
constexpr size_t max_view_depth = 64;

bool add(size_t a, size_t b, size_t & result) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

bool multiply(size_t a, size_t b, size_t & result) {
    if (b && a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

struct dimension {
    size_t count;
    size_t stride;
};

const char * layout(const ggml_tensor & tensor, std::array<dimension, GGML_MAX_DIMS> & dims, size_t & extent) {
    if (tensor.type < 0 || tensor.type >= GGML_TYPE_COUNT) {
        return "invalid_tensor_layout";
    }
    const auto block = ggml_blck_size(tensor.type);
    const auto bytes = ggml_type_size(tensor.type);
    if (block <= 0 || bytes == 0) {
        return "unsupported_tensor_type";
    }
    bool empty = false;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (tensor.ne[i] < 0 || uint64_t(tensor.ne[i]) > std::numeric_limits<size_t>::max()) {
            return "invalid_tensor_layout";
        }
        empty |= tensor.ne[i] == 0;
        dims[i] = {size_t(tensor.ne[i]), tensor.nb[i]};
    }
    if (tensor.ne[0] % block != 0) {
        return "invalid_quantized_layout";
    }
    dims[0].count /= block;
    extent = empty ? 0 : bytes;
    if (!empty) {
        for (const auto & dim : dims) {
            size_t tail;
            if (!multiply(dim.count - 1, dim.stride, tail) || !add(extent, tail, extent)) {
                return "tensor_range_overflow";
            }
        }
    }
    return nullptr;
}

void merge_ranges(std::vector<llama_memory_weight_range> & ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const llama_memory_weight_range & a, const llama_memory_weight_range & b) {
        return a.offset < b.offset;
    });
    size_t used = 0;
    for (const auto range : ranges) {
        if (used && range.offset <= ranges[used - 1].offset + ranges[used - 1].size) {
            auto & last = ranges[used - 1];
            last.size = std::max(last.offset + last.size, range.offset + range.size) - last.offset;
        } else {
            ranges[used++] = range;
        }
    }
    ranges.resize(used);
}

const char * resolve(const llama_memory_placement_view & view, std::vector<llama_memory_weight_range> & ranges) {
    if (!view.tensor) {
        return "tensor_unavailable";
    }
    if (!view.buffer) {
        return "tensor_buffer_unavailable";
    }
    bool found_buffer = false;
    size_t depth = 0;
    for (auto * tensor = view.tensor; tensor; tensor = tensor->view_src) {
        if (++depth > max_view_depth || (tensor->buffer && tensor->buffer != view.buffer)) {
            return "invalid_tensor_view";
        }
        found_buffer |= tensor->buffer == view.buffer;
    }
    if (!found_buffer) {
        return "tensor_buffer_unavailable";
    }
    if (ggml_backend_memory_buffer_flags(view.buffer) & GGML_BACKEND_MEMORY_BUFFER_OPAQUE) {
        return "opaque_tensor_storage";
    }

    std::array<dimension, GGML_MAX_DIMS> dims;
    size_t extent;
    if (const auto * reason = layout(*view.tensor, dims, extent)) {
        return reason;
    }
    if (extent == 0) {
        return nullptr;
    }

    const ggml_tensor * current = view.tensor;
    for (size_t depth = 0; current; ++depth) {
        if (depth == max_view_depth) {
            return "invalid_tensor_view";
        }
        if (!current->data || (current->buffer && current->buffer != view.buffer)) {
            return "invalid_tensor_view";
        }
        const auto * parent = current->view_src;
        if (parent) {
            std::array<dimension, GGML_MAX_DIMS> ignored;
            size_t parent_extent, current_extent, end;
            if (layout(*parent, ignored, parent_extent) || layout(*current, ignored, current_extent) ||
                    !add(current->view_offs, current_extent, end) || end > parent_extent) {
                return "invalid_tensor_view";
            }
            const auto address = reinterpret_cast<uintptr_t>(parent->data);
            if (!parent->data || current->view_offs > std::numeric_limits<uintptr_t>::max() - address ||
                    reinterpret_cast<uintptr_t>(current->data) != address + current->view_offs) {
                return "invalid_tensor_view";
            }
        }
        current = parent;
    }

    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(view.buffer));
    const auto address = reinterpret_cast<uintptr_t>(view.tensor->data);
    if (!base || address < base || address - base > std::numeric_limits<size_t>::max()) {
        return "tensor_offset_unavailable";
    }
    const size_t offset = address - base;
    const size_t capacity = ggml_backend_buffer_get_size(view.buffer);
    if (offset > capacity || extent > capacity - offset || extent > std::numeric_limits<uintptr_t>::max() - address) {
        return "tensor_range_out_of_bounds";
    }

    const auto * root = view.tensor;
    while (root->view_src) { root = root->view_src; }
    if (root != view.tensor) {
        std::array<dimension, GGML_MAX_DIMS> ignored;
        size_t root_extent;
        const auto root_address = reinterpret_cast<uintptr_t>(root->data);
        if (layout(*root, ignored, root_extent) || root_address < base || root_address - base > capacity ||
                root_extent > capacity - (root_address - base) ||
                root_extent > std::numeric_limits<uintptr_t>::max() - root_address) {
            return "tensor_range_out_of_bounds";
        }
    }
    auto * provider = ggml_backend_memory_buffer_device(view.buffer);
    if (root->extra && provider && ggml_backend_dev_type(provider) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        // CPU repackers may change storage layout or expand it beyond the logical tensor payload.
        // A full dense tensor with an unchanged storage size is safe; a view needs a repacker-specific map.
        if (root != view.tensor || !ggml_is_contiguous(root) ||
                ggml_backend_buffer_get_alloc_size(view.buffer, root) != extent) {
            return "transformed_tensor_storage";
        }
    }

    std::sort(dims.begin(), dims.end(), [](const dimension & a, const dimension & b) { return a.stride < b.stride; });
    ranges.push_back({0, ggml_type_size(view.tensor->type)});
    for (const auto & dim : dims) {
        if (dim.count <= 1 || dim.stride == 0) {
            continue;
        }
        // Sorting strides makes dense and transposed tensors a single range without element scans.
        if (ranges.size() == 1 && dim.stride <= ranges[0].size) {
            ranges[0].size += (dim.count - 1) * dim.stride; // checked by layout()
            continue;
        }
        if (dim.count > max_ranges / ranges.size()) {
            return "tensor_range_limit";
        }
        const auto previous = ranges;
        for (size_t i = 1; i < dim.count; ++i) {
            for (const auto range : previous) {
                ranges.push_back({range.offset + i * dim.stride, range.size});
            }
        }
        merge_ranges(ranges);
    }
    for (auto & range : ranges) {
        range.offset += offset; // the whole extent was checked against the buffer
    }
    return nullptr;
}

} // namespace

llama_memory_weight_storage_view llama_memory_resolve_weight_storage(const llama_memory_placement_view & placement) {
    llama_memory_weight_storage_view result = {placement, {}, nullptr};
    result.missing_reason = resolve(placement, result.ranges);
    if (result.missing_reason) {
        result.ranges.clear();
    }
    return result;
}

bool llama_model_visit_weight_storage(const llama_model * model, const llama_memory_weight_storage_visitor & visitor) {
    bool complete = true;
    const bool placement_complete = llama_model_visit_placement(model, [&](const llama_memory_placement_view & placement) {
        const auto storage = llama_memory_resolve_weight_storage(placement);
        complete = !storage.missing_reason && placement.known && placement.assignment.device && complete;
        visitor(storage);
    });
    return placement_complete && complete;
}
