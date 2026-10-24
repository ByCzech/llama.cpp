#include "../src/llama-memory-weight-storage.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>

// Synthetic buffer-relative addresses. Tests never allocate or access the declared tensor storage.
struct fixture {
    ggml_backend_reg reg = {};
    ggml_backend_device device = {};
    ggml_backend_buffer_type type = {};
    ggml_backend_buffer_t buffer;
    uintptr_t base = 0x1000;
    bool opaque = false;
    size_t base_reads = 0;

    explicit fixture(size_t size = 4096) {
        reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
            if (std::strcmp(name, "tllama_buffer_memory_flags") != 0) {
                return nullptr;
            }
            static ggml_backend_memory_buffer_flags_t inspect = [](ggml_backend_buffer_t buffer) -> uint32_t {
                return static_cast<fixture *>(buffer->context)->opaque ? GGML_BACKEND_MEMORY_BUFFER_OPAQUE : 0;
            };
            return reinterpret_cast<void *>(inspect);
        };
        device.reg = &reg;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
        type.device = &device;
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
        ggml_backend_buffer_i iface = {};
        iface.get_base = [](ggml_backend_buffer_t buffer) -> void * {
            auto & self = *static_cast<fixture *>(buffer->context);
            ++self.base_reads;
            return reinterpret_cast<void *>(self.base);
        };
        buffer = ggml_backend_buffer_init(&type, iface, this, size);
    }

    ~fixture() { ggml_backend_buffer_free(buffer); }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;

    ggml_tensor tensor(size_t offset = 0, int64_t count = 8, ggml_type kind = GGML_TYPE_F32) const {
        ggml_tensor result = {};
        result.type = kind;
        result.buffer = buffer;
        result.data = reinterpret_cast<void *>(base + offset);
        result.ne[0] = count;
        result.nb[0] = ggml_type_size(kind);
        result.nb[1] = result.nb[0] * (count / ggml_blck_size(kind));
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            result.ne[i] = 1;
            if (i > 1) { result.nb[i] = result.nb[i - 1]; }
        }
        return result;
    }

    llama_memory_placement_view placement(const ggml_tensor & tensor) {
        llama_memory_tensor_assignment assignment = {&device, &device, &type, LLAMA_MEMORY_ASSIGNMENT_SELECTION};
        return {LLAMA_MEMORY_MODEL_PART_REPEATING, 7, buffer, true, assignment, &tensor};
    }

    llama_memory_weight_storage_view inspect(const ggml_tensor & tensor) {
        return llama_memory_resolve_weight_storage(placement(tensor));
    }
};

static void exact(fixture & source, const ggml_tensor & tensor, std::initializer_list<llama_memory_weight_range> expected) {
    const auto result = source.inspect(tensor);
    GGML_ASSERT(!result.missing_reason && result.ranges.size() == expected.size());
    size_t i = 0;
    for (const auto range : expected) {
        GGML_ASSERT(result.ranges[i].offset == range.offset && result.ranges[i].size == range.size);
        ++i;
    }
    GGML_ASSERT(result.placement.tensor == &tensor && result.placement.buffer == source.buffer);
    GGML_ASSERT(result.placement.part == LLAMA_MEMORY_MODEL_PART_REPEATING && result.placement.layer == 7);
    GGML_ASSERT(result.placement.assignment.source == LLAMA_MEMORY_ASSIGNMENT_SELECTION);
}

static void unknown(fixture & source, const ggml_tensor & tensor, const char * reason) {
    const auto result = source.inspect(tensor);
    GGML_ASSERT(result.missing_reason && std::strcmp(result.missing_reason, reason) == 0);
    GGML_ASSERT(result.ranges.empty());
}

static void test_layouts() {
    fixture source;
    auto tensor = source.tensor(12);
    exact(source, tensor, {{12, 32}});
    tensor.ne[1] = 3;
    tensor.nb[1] = 64;
    exact(source, tensor, {{12, 32}, {76, 32}, {140, 32}});
    tensor.ne[0] = 3;
    tensor.ne[1] = 8;
    tensor.nb[0] = 64;
    tensor.nb[1] = 4;
    exact(source, tensor, {{12, 32}, {76, 32}, {140, 32}}); // transposed, same storage
    tensor.nb[0] = 32;
    exact(source, tensor, {{12, 96}});
    tensor.nb[0] = 16;
    exact(source, tensor, {{12, 64}}); // overlapping rows count their union
    tensor.nb[0] = 0;
    exact(source, tensor, {{12, 32}});
    auto quantized = source.tensor(32, 64, GGML_TYPE_Q4_0);
    const auto block_bytes = ggml_type_size(quantized.type);
    exact(source, quantized, {{32, 2 * block_bytes}});
    quantized.ne[1] = 2;
    quantized.nb[1] = 3 * block_bytes;
    exact(source, quantized, {{32, 2 * block_bytes}, {32 + 3 * block_bytes, 2 * block_bytes}});
    quantized.nb[0] = 2 * block_bytes;
    quantized.ne[1] = 1;
    exact(source, quantized, {{32, block_bytes}, {32 + 2 * block_bytes, block_bytes}});
    quantized.ne[0] = 33;
    unknown(source, quantized, "invalid_quantized_layout");
    tensor = source.tensor();
    tensor.ne[1] = 0;
    tensor.data = nullptr;
    exact(source, tensor, {});
    tensor.ne[2] = -1;
    unknown(source, tensor, "invalid_tensor_layout");
    tensor = source.tensor();
    tensor.type = GGML_TYPE_COUNT;
    unknown(source, tensor, "invalid_tensor_layout");
}

static void test_views_and_coverage() {
    fixture source;
    auto backing = source.tensor(64, 128);
    auto view = source.tensor(80, 16);
    view.buffer = nullptr;
    view.view_src = &backing;
    view.view_offs = 16;
    exact(source, view, {{80, 64}});
    auto nested = source.tensor(84, 4);
    nested.buffer = nullptr;
    nested.view_src = &view;
    nested.view_offs = 4;
    exact(source, nested, {{84, 16}});
    backing.ne[0] = 4096;
    unknown(source, nested, "tensor_range_out_of_bounds");
    backing.ne[0] = 128;
    nested.data = reinterpret_cast<void *>(source.base + 85);
    unknown(source, nested, "invalid_tensor_view");
    nested.data = reinterpret_cast<void *>(source.base + 84);
    nested.ne[0] = 100;
    unknown(source, nested, "invalid_tensor_view");
    nested.ne[0] = 4;
    fixture other;
    view.buffer = other.buffer;
    unknown(source, nested, "invalid_tensor_view");
    view.buffer = nullptr;
    view.view_src = &view;
    unknown(source, nested, "invalid_tensor_view");

    auto tensor = source.tensor();
    auto transformed = source.tensor();
    transformed.extra = &source;
    exact(source, transformed, {{0, 32}});
    source.type.iface.get_alloc_size = [](ggml_backend_buffer_type_t, const ggml_tensor *) { return size_t(128); };
    unknown(source, transformed, "transformed_tensor_storage");
    source.type.iface.get_alloc_size = nullptr;
    auto packed_view = source.tensor(4, 2);
    packed_view.view_src = &transformed;
    packed_view.view_offs = 4;
    unknown(source, packed_view, "transformed_tensor_storage");
    auto placement = source.placement(tensor);
    placement.assignment = {};
    placement.known = false;
    const auto still_known = llama_memory_resolve_weight_storage(placement);
    GGML_ASSERT(!still_known.missing_reason && still_known.ranges.size() == 1);
    GGML_ASSERT(!still_known.placement.assignment.device && !still_known.placement.known);
    placement.tensor = nullptr;
    GGML_ASSERT(std::strcmp(llama_memory_resolve_weight_storage(placement).missing_reason, "tensor_unavailable") == 0);
    placement.tensor = &tensor;
    placement.buffer = nullptr;
    GGML_ASSERT(std::strcmp(llama_memory_resolve_weight_storage(placement).missing_reason, "tensor_buffer_unavailable") == 0);
    tensor.buffer = nullptr;
    unknown(source, tensor, "tensor_buffer_unavailable");
    tensor.buffer = source.buffer;
    tensor.data = nullptr;
    unknown(source, tensor, "invalid_tensor_view");
    source.opaque = true;
    source.base_reads = 0;
    unknown(source, tensor, "opaque_tensor_storage");
    GGML_ASSERT(source.base_reads == 0); // no inspection of opaque backend addresses
    bool visited = false;
    GGML_ASSERT(!llama_model_visit_weight_storage(nullptr, [&](const llama_memory_weight_storage_view &) { visited = true; }));
    GGML_ASSERT(!visited);
}

static void test_limits() {
    fixture source;
    auto tensor = source.tensor(4092, 1);
    exact(source, tensor, {{4092, 4}});
    tensor.ne[0] = 2;
    unknown(source, tensor, "tensor_range_out_of_bounds");
    tensor = source.tensor();
    tensor.data = reinterpret_cast<void *>(source.base - 1);
    unknown(source, tensor, "tensor_offset_unavailable");
    tensor = source.tensor();
    tensor.nb[0] = std::numeric_limits<size_t>::max();
    unknown(source, tensor, "tensor_range_overflow");
    source.buffer->size = 1024 * 1024;
    tensor = source.tensor(0, 4097);
    tensor.nb[0] = 8;
    unknown(source, tensor, "tensor_range_limit");
    source.base = std::numeric_limits<uintptr_t>::max() - 15;
    tensor = source.tensor(0, 8);
    unknown(source, tensor, "tensor_range_out_of_bounds");
}

static void test_sparse_mapping_and_identity() {
    constexpr size_t mib = 1024 * 1024;
    fixture source(1024 * mib), copy(1024 * mib);
    const auto first = source.tensor(0, mib / 4);
    const auto last = source.tensor(1023 * mib, mib / 4);
    exact(source, first, {{0, mib}});
    exact(source, last, {{1023 * mib, mib}});
    GGML_ASSERT(ggml_backend_buffer_get_size(source.buffer) == 1024 * mib);
    GGML_ASSERT(source.inspect(first).ranges[0].size + source.inspect(last).ranges[0].size == 2 * mib);
    const auto duplicate = copy.tensor(0, mib / 4);
    const auto a = source.inspect(first), b = copy.inspect(duplicate);
    GGML_ASSERT(first.data == duplicate.data); // device/sentinel addresses may coincide
    GGML_ASSERT(a.placement.buffer != b.placement.buffer); // do not deduplicate them
}

static void test_random_layouts() {
    fixture source(16384);
    std::mt19937 random(42);
    for (size_t trial = 0; trial < 1000; ++trial) {
        auto tensor = source.tensor(16);
        std::set<size_t> expected, actual;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            tensor.ne[dim] = 1 + random() % 4;
            tensor.nb[dim] = random() % 40;
        }
        for (int64_t a = 0; a < tensor.ne[0]; ++a)
        for (int64_t b = 0; b < tensor.ne[1]; ++b)
        for (int64_t c = 0; c < tensor.ne[2]; ++c)
        for (int64_t d = 0; d < tensor.ne[3]; ++d) {
            const auto offset = 16 + a * tensor.nb[0] + b * tensor.nb[1] + c * tensor.nb[2] + d * tensor.nb[3];
            for (size_t i = 0; i < sizeof(float); ++i) { expected.insert(offset + i); }
        }
        const auto storage = source.inspect(tensor);
        GGML_ASSERT(!storage.missing_reason);
        size_t end = 0;
        for (const auto range : storage.ranges) {
            GGML_ASSERT(range.offset > end && range.size > 0);
            end = range.offset + range.size;
            for (size_t i = range.offset; i < end; ++i) { actual.insert(i); }
        }
        GGML_ASSERT(actual == expected);
    }
}

int main() {
    test_layouts();
    test_views_and_coverage();
    test_limits();
    test_sparse_mapping_and_identity();
    test_random_layouts();
    std::puts("weight storage ranges: PASSED");
}
