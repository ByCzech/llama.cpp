#include "../tools/server/server-memory.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstring>
#include <cstdio>

// Device metadata fixtures do not emulate allocation or inference. Sizes are read
// through the same native buffer API as real CPU allocations below.
struct fixture {
    ggml_backend_reg reg = {};
    ggml_backend_device dev = {};
    ggml_backend_buffer_type type = {};
    ggml_backend_buffer buf = {};
    enum ggml_backend_dev_type kind;
    uint32_t flags;
    bool host;
    bool legacy_split = false;

    fixture(enum ggml_backend_dev_type kind, bool host = false, uint32_t flags = 0) : kind(kind), flags(flags), host(host) {
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.context = this;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "Fixture"; };
        reg.iface.get_proc_address = [](ggml_backend_reg_t r, const char * name) -> void * {
            auto * self = static_cast<fixture *>(r->context);
            if (self->legacy_split) {
                return std::strcmp(name, "ggml_backend_split_buffer_type") == 0 ? r->context : nullptr;
            }
            if (std::strcmp(name, "tllama_buffer_memory_flags") != 0) { return nullptr; }
            static ggml_backend_memory_buffer_flags_t probe = [](ggml_backend_buffer_t b) {
                return static_cast<fixture *>(b->context)->flags;
            };
            return reinterpret_cast<void *>(probe);
        };
        dev.reg = &reg;
        dev.context = this;
        dev.iface.get_name = [](ggml_backend_dev_t) { return "same-name"; };
        dev.iface.get_type = [](ggml_backend_dev_t d) { return static_cast<fixture *>(d->context)->kind; };
        type.device = &dev;
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "same-buffer-name"; };
        type.iface.is_host = [](ggml_backend_buffer_type_t t) { return static_cast<fixture *>(t->context)->host; };
        buf.buft = &type;
        buf.context = this;
        buf.size = 4096;
    }
};

static llama_memory_buffer_view view(ggml_backend_buffer_t buffer) {
    return {buffer, LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS, LLAMA_MEMORY_BACKING_ALLOCATED};
}

static uint64_t total(const json & result) {
    uint64_t sum = 0;
    for (const auto & buffer : result["buffers"]) {
        if (!buffer["size_bytes"].is_null()) { sum += buffer["size_bytes"].get<uint64_t>(); }
    }
    return sum;
}

static void test_deduplication() {
    auto * first = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096);
    auto * second = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 8192);
    GGML_ASSERT(first && second);
    server_memory_collector collector;
    const auto main = collector.owner(&first, "model", "main");
    GGML_ASSERT(collector.owner(&first, "model", "mtp") == main);
    const auto draft = collector.owner(&second, "model", "draft");
    ggml_backend_buffer_t children[] = {first, second};
    auto * multi = ggml_backend_multi_buffer_alloc_buffer(children, 2);
    llama_memory_visit_buffer(multi, LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS,
        [&](const llama_memory_buffer_view & v) { collector.buffer(main, v); });
    collector.buffer(main, view(first));
    collector.buffer(draft, view(first));
    collector.placement(main, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, first, true});
    collector.placement(main, {LLAMA_MEMORY_MODEL_PART_REPEATING, 1, second, true});
    const auto result = collector.finish();
    GGML_ASSERT(result["status"] == "known");
    GGML_ASSERT(result["buffers"].size() == 2);
    GGML_ASSERT(total(result) == ggml_backend_buffer_get_size(first) + ggml_backend_buffer_get_size(second));
    GGML_ASSERT(result["buffers"][0]["owners"].size() == 2);
    GGML_ASSERT(result["owners"][main]["roles"].size() == 2);
    GGML_ASSERT(result["placement"]["status"] == "known");
    GGML_ASSERT(result["placement"]["layer_groups"].size() == 1);
    GGML_ASSERT(result["placement"]["layer_groups"][0]["location"] == "cpu");
    GGML_ASSERT(result["placement"]["layer_groups"][0]["layer_indices"].size() == 2);
    GGML_ASSERT(result["buffers"][0]["memory_kind"] == "host");
    GGML_ASSERT(result["devices"][0]["type"] == "cpu");
    // The container owns both children. Free only after copying all metadata.
    ggml_backend_buffer_free(multi);
    GGML_ASSERT(total(result) == 12288);
}

static void test_device_classification() {
    fixture cpu(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    fixture gpu(GGML_BACKEND_DEVICE_TYPE_GPU);
    fixture igpu(GGML_BACKEND_DEVICE_TYPE_IGPU);
    fixture pinned(GGML_BACKEND_DEVICE_TYPE_GPU, true, GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED);
    server_memory_collector collector;
    const auto owner = collector.owner(&collector, "model", "main");
    for (auto * source : {&cpu, &gpu, &igpu, &pinned}) { collector.buffer(owner, view(&source->buf)); }
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &cpu.buf, true});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &gpu.buf, true});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 1, &igpu.buf, true});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_INPUT, -1, &pinned.buf, true});
    const auto result = collector.finish();
    GGML_ASSERT(total(result) == 4 * 4096);
    // Distinct native devices must not collapse just because their names match.
    GGML_ASSERT(result["devices"].size() == 4);
    GGML_ASSERT(result["buffers"][1]["memory_kind"] == "device");
    GGML_ASSERT(result["buffers"][2]["memory_kind"] == "device");
    GGML_ASSERT(result["buffers"][3]["memory_kind"] == "host_pinned");
    GGML_ASSERT(result["status"] == "known");
    GGML_ASSERT(result["placement"]["status"] == "partial");
    bool mixed = false, integrated = false, unknown = false;
    for (const auto & group : result["placement"]["layer_groups"]) {
        mixed |= group["location"] == "mixed";
        integrated |= group["location"] == "accelerator" && group["device_ids"][0] == "d3";
        unknown |= group["location"] == "unknown";
    }
    GGML_ASSERT(mixed && integrated && unknown);
}

static void test_partial_and_unknown() {
    fixture known(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    fixture opaque(GGML_BACKEND_DEVICE_TYPE_META, false, GGML_BACKEND_MEMORY_BUFFER_OPAQUE);
    server_memory_collector collector;
    const auto owner = collector.owner(&collector, "context", "main");
    collector.buffer(owner, view(&known.buf));
    collector.buffer(owner, view(&opaque.buf));
    collector.buffer(owner, view(nullptr));
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &opaque.buf, true});
    auto result = collector.finish();
    GGML_ASSERT(result["status"] == "partial");
    GGML_ASSERT(total(result) == 4096);
    GGML_ASSERT(result["buffers"][1]["size_bytes"].is_null());
    GGML_ASSERT(result["buffers"][1]["memory_kind"] == "unknown");
    GGML_ASSERT(result["missing_components"].size() == 2);
    GGML_ASSERT(result["placement"]["status"] == "unavailable");
    // Disagreeing observations of one allocation must not silently claim either backing.
    auto mapped = view(&known.buf);
    mapped.backing = LLAMA_MEMORY_BACKING_MMAP;
    collector.buffer(owner, mapped);
    result = collector.finish();
    GGML_ASSERT(result["buffers"][0]["backing"] == "unknown");
    GGML_ASSERT(result["missing_components"].size() == 3);
    fixture old_split(GGML_BACKEND_DEVICE_TYPE_GPU);
    old_split.legacy_split = true;
    GGML_ASSERT(ggml_backend_memory_buffer_flags(&old_split.buf) & GGML_BACKEND_MEMORY_BUFFER_OPAQUE);

    server_memory_native native;
    result = native.capture({});
    GGML_ASSERT(result["status"] == "unavailable");
    GGML_ASSERT(result["buffers"].empty());
    GGML_ASSERT(result["missing_components"][0]["owner_id"].is_null());
}

int main() {
    ggml_backend_load_all();
    test_deduplication();
    test_device_classification();
    test_partial_and_unknown();
    std::puts("memory report PASSED");
}
