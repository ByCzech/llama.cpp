#include "../tools/server/server-memory.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

// Synthetic addresses and buffer spans, with no tensor data access or GPU execution.
struct fixture {
    ggml_backend_reg reg = {};
    ggml_backend_device device = {};
    ggml_backend_buffer_type type = {};
    ggml_backend_buffer buffer = {};
    uintptr_t base = 0x1000;
    size_t base_reads = 0;
    enum ggml_backend_dev_type kind;

    explicit fixture(size_t capacity = 4096, enum ggml_backend_dev_type kind = GGML_BACKEND_DEVICE_TYPE_CPU) : kind(kind) {
        reg.iface.get_name = [](ggml_backend_reg_t) { return "WeightFixture"; };
        device.reg = &reg;
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "same-device-name"; };
        device.iface.get_type = [](ggml_backend_dev_t dev) { return static_cast<fixture *>(dev->context)->kind; };
        type.device = &device;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "same-buffer-name"; };
        type.iface.is_host = [](ggml_backend_buffer_type_t type) { return ggml_backend_dev_type(type->device) == GGML_BACKEND_DEVICE_TYPE_CPU; };
        buffer.buft = &type;
        buffer.context = this;
        buffer.size = capacity;
        buffer.iface.get_base = [](ggml_backend_buffer_t buffer) -> void * {
            auto & self = *static_cast<fixture *>(buffer->context);
            ++self.base_reads;
            return reinterpret_cast<void *>(self.base);
        };
    }

    void inventory(server_memory_collector & collector, size_t owner) {
        collector.buffer(owner, {&buffer, LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS, LLAMA_MEMORY_BACKING_MMAP});
    }

    void tensor(server_memory_collector & collector, size_t owner, size_t offset, size_t size,
                llama_memory_model_part part = LLAMA_MEMORY_MODEL_PART_REPEATING, int32_t layer = 0,
                ggml_backend_dev_t assigned = nullptr, bool unknown = false) {
        GGML_ASSERT(size % sizeof(float) == 0);
        ggml_tensor tensor = {};
        tensor.type = GGML_TYPE_F32;
        tensor.buffer = &buffer;
        tensor.data = reinterpret_cast<void *>(base + offset);
        tensor.ne[0] = size / sizeof(float);
        tensor.nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) { tensor.ne[i] = 1; tensor.nb[i] = size; }
        ggml_set_name(&tensor, "same-weight-name");
        const llama_memory_tensor_assignment assignment = {
            unknown ? nullptr : assigned ? assigned : &device, &device, &type, LLAMA_MEMORY_ASSIGNMENT_SELECTION};
        const llama_memory_placement_view placement = {part, layer, &buffer, true, assignment, &tensor};
        collector.placement(owner, placement);
        collector.weight_storage(owner, llama_memory_resolve_weight_storage(placement));
    }
};

static uint64_t total(const json & weights, bool exclude_input_only = false) {
    uint64_t bytes = 0;
    for (const auto & interval : weights["intervals"]) {
        bool selected = !exclude_input_only;
        for (const auto & relation : interval["relationships"]) { selected |= relation["component"] != "input_weights"; }
        if (selected) { bytes += interval["size_bytes"].get<uint64_t>(); }
    }
    return bytes;
}

static void test_sparse_mapping() {
    constexpr size_t mib = 1024 * 1024;
    fixture source(1024 * mib);
    server_memory_collector collector;
    const auto owner = collector.owner(&source, "model", "main");
    source.inventory(collector, owner);
    source.tensor(collector, owner, 0, mib);
    source.tensor(collector, owner, 1023 * mib, mib);
    source.tensor(collector, owner, 0, mib); // repeated metadata is not another storage allocation
    const auto report = collector.finish();
    GGML_ASSERT(report["buffers"][0]["size_bytes"] == 1024 * mib);
    GGML_ASSERT(report["weight_storage"]["status"] == "known");
    GGML_ASSERT(report["weight_storage"]["intervals"].size() == 2);
    GGML_ASSERT(total(report["weight_storage"]) == 2 * mib);
    GGML_ASSERT(report["allocation_locality"]["status"] == "unavailable");
    const auto reads = source.base_reads;
    GGML_ASSERT(collector.finish() == report && source.base_reads == reads);
}

static void test_shared_roles_and_owners() {
    fixture source;
    server_memory_collector collector;
    const auto main = collector.owner(&source, "model", "main");
    GGML_ASSERT(collector.owner(&source, "model", "mtp") == main);
    source.inventory(collector, main);
    source.tensor(collector, main, 0, 8, LLAMA_MEMORY_MODEL_PART_INPUT, -1);
    source.tensor(collector, main, 4, 8, LLAMA_MEMORY_MODEL_PART_OUTPUT, -1);
    const auto first = collector.finish()["weight_storage"];
    GGML_ASSERT(first["intervals"].size() == 3 && total(first) == 12 && total(first, true) == 8);
    GGML_ASSERT(first["intervals"][0]["size_bytes"] == 4 && first["intervals"][0]["relationships"].size() == 1);
    GGML_ASSERT(first["intervals"][1]["size_bytes"] == 4 && first["intervals"][1]["relationships"].size() == 2);
    GGML_ASSERT(first["intervals"][2]["size_bytes"] == 4 && first["intervals"][2]["relationships"].size() == 1);
    int independent_model = 0;
    const auto draft = collector.owner(&independent_model, "model", "draft");
    source.inventory(collector, draft);
    source.tensor(collector, draft, 4, 8, LLAMA_MEMORY_MODEL_PART_MTP, 40);
    source.tensor(collector, main, 0, 8, LLAMA_MEMORY_MODEL_PART_INPUT, -1);
    const auto report = collector.finish();
    const auto weights = report["weight_storage"];
    GGML_ASSERT(weights["status"] == "known" && total(weights) == 12 && total(weights, true) == 8);
    GGML_ASSERT(weights["intervals"][1]["relationships"].size() == 3);
    GGML_ASSERT(weights["intervals"][2]["relationships"].size() == 2);
    GGML_ASSERT(report["owners"][main]["roles"].size() == 2 && report["buffers"].size() == 1);
}

static void test_independent_copies() {
    fixture first, second;
    server_memory_collector collector;
    const auto main = collector.owner(&first, "model", "main");
    const auto draft = collector.owner(&second, "model", "draft");
    first.inventory(collector, main);
    second.inventory(collector, draft);
    first.tensor(collector, main, 0, 8);
    second.tensor(collector, draft, 0, 8);
    const auto weights = collector.finish()["weight_storage"];
    GGML_ASSERT(weights["intervals"].size() == 2 && total(weights) == 16);
    GGML_ASSERT(weights["intervals"][0]["buffer_id"] != weights["intervals"][1]["buffer_id"]);
}

static void test_assignment_conflict() {
    fixture source, gpu(4096, GGML_BACKEND_DEVICE_TYPE_GPU);
    server_memory_collector collector;
    const auto owner = collector.owner(&source, "model", "main");
    source.inventory(collector, owner);
    source.tensor(collector, owner, 0, 8, LLAMA_MEMORY_MODEL_PART_INPUT, -1);
    source.tensor(collector, owner, 4, 8, LLAMA_MEMORY_MODEL_PART_OUTPUT, -1, &gpu.device);
    const auto report = collector.finish();
    const auto weights = report["weight_storage"];
    GGML_ASSERT(report["status"] == "known" && report["placement"]["status"] == "known");
    GGML_ASSERT(weights["status"] == "partial" && weights["missing_components"].size() == 2);
    GGML_ASSERT(total(weights) == 12 && weights["intervals"].size() == 3);
    const auto conflict = weights["intervals"][1]["assignment"];
    GGML_ASSERT(conflict["status"] == "unavailable" && conflict["device_ids"].empty());
    GGML_ASSERT(conflict["reasons"] == json::array({"assignment_conflict"}));
    GGML_ASSERT(weights["intervals"][0]["assignment"]["status"] == "known");
    GGML_ASSERT(weights["intervals"][2]["assignment"]["status"] == "known");
}

static void test_partial_and_unavailable() {
    fixture source;
    server_memory_collector collector;
    const auto owner = collector.owner(&source, "model", "main");
    source.inventory(collector, owner);
    GGML_ASSERT(collector.finish()["weight_storage"]["status"] == "unavailable"); // unvisited model is not empty
    source.tensor(collector, owner, 0, 8, LLAMA_MEMORY_MODEL_PART_REPEATING, 0, nullptr, true);
    auto report = collector.finish();
    GGML_ASSERT(report["status"] == "known" && report["weight_storage"]["status"] == "partial");
    GGML_ASSERT(total(report["weight_storage"]) == 8);
    GGML_ASSERT(report["weight_storage"]["intervals"][0]["assignment"]["device_ids"].empty());
    collector.weight_missing(owner, "mtp_layers", 40, "opaque_tensor_storage");
    collector.weight_missing(owner, "mtp_layers", 40, "opaque_tensor_storage");
    report = collector.finish();
    GGML_ASSERT(report["weight_storage"]["missing_components"].size() == 2);
    GGML_ASSERT(total(report["weight_storage"]) == 8);
    fixture known;
    server_memory_collector only_missing;
    const auto other = only_missing.owner(&known, "model", "main");
    known.inventory(only_missing, other);
    only_missing.weight_missing(other, "model_weights", -1, "tensor_storage_unavailable");
    GGML_ASSERT(only_missing.finish()["weight_storage"]["status"] == "unavailable");
    known.tensor(only_missing, other, 0, 0);
    GGML_ASSERT(only_missing.finish()["weight_storage"]["status"] == "partial");
    GGML_ASSERT(only_missing.finish()["weight_storage"]["intervals"].empty());
}

static void test_invalid_ranges() {
    fixture source;
    server_memory_collector collector;
    const auto owner = collector.owner(&source, "model", "main");
    source.inventory(collector, owner);
    const llama_memory_tensor_assignment assignment = {&source.device, &source.device, &source.type, LLAMA_MEMORY_ASSIGNMENT_SELECTION};
    llama_memory_weight_storage_view view = {{LLAMA_MEMORY_MODEL_PART_OTHER, -1, &source.buffer, true, assignment}, {}, nullptr};
    view.ranges = {{4090, 16}};
    collector.weight_storage(owner, view);
    const auto rejected = collector.finish()["weight_storage"];
    GGML_ASSERT(rejected["status"] == "unavailable" && rejected["intervals"].empty());
    GGML_ASSERT(rejected["missing_components"][0]["reason"] == "invalid_storage_range");
    view.ranges = {{std::numeric_limits<size_t>::max() - 1, 16}};
    collector.weight_storage(owner, view);
    GGML_ASSERT(collector.finish()["weight_storage"] == rejected);
    view.ranges = {{0, 8}};
    view.placement.known = false;
    collector.weight_storage(owner, view);
    GGML_ASSERT(collector.finish()["weight_storage"]["intervals"].empty());
    server_memory_collector missing_inventory;
    const auto missing_owner = missing_inventory.owner(&source, "model", "main");
    view.placement.known = true;
    missing_inventory.weight_storage(missing_owner, view);
    GGML_ASSERT(missing_inventory.finish()["weight_storage"]["missing_components"][0]["reason"] == "buffer_not_in_inventory");
}

static void test_copy_and_independent_failure() {
    fixture source;
    server_memory_collector collector;
    const auto owner = collector.owner(&source, "model", "main");
    source.inventory(collector, owner);
    source.tensor(collector, owner, 0, 8);
    const auto before = collector.finish();
    auto copy = collector;
    source.tensor(copy, owner, 8, 8);
    GGML_ASSERT(total(copy.finish()["weight_storage"]) == 16);
    GGML_ASSERT(collector.finish() == before);
    collector.weight_missing(owner, "repeating_layers", 1, "collection_failed");
    const auto after = collector.finish();
    GGML_ASSERT(after["buffers"] == before["buffers"] && after["placement"] == before["placement"]);
    GGML_ASSERT(after["allocation_locality"] == before["allocation_locality"] && after["status"] == before["status"]);
    GGML_ASSERT(after["weight_storage"]["status"] == "partial" && total(after["weight_storage"]) == 8);
    GGML_ASSERT(copy.finish()["weight_storage"]["status"] == "known");
}

static void test_random_relationships() {
    std::mt19937 random(74);
    for (size_t trial = 0; trial < 100; ++trial) {
        fixture source;
        server_memory_collector collector;
        const auto owner = collector.owner(&source, "model", "main");
        source.inventory(collector, owner);
        std::map<size_t, std::set<int>> expected, actual;
        for (size_t i = 0; i < 25; ++i) {
            const auto offset = 4 * (random() % 24), size = 4 * (1 + random() % 8);
            const int layer = random() % 3;
            source.tensor(collector, owner, offset, size, LLAMA_MEMORY_MODEL_PART_REPEATING, layer);
            for (size_t byte = offset; byte < offset + size; ++byte) { expected[byte].insert(layer); }
        }
        const auto weights = collector.finish()["weight_storage"];
        GGML_ASSERT(weights["status"] == "known");
        uint64_t previous_end = 0;
        for (const auto & interval : weights["intervals"]) {
            const auto offset = interval["offset_bytes"].get<uint64_t>(), size = interval["size_bytes"].get<uint64_t>();
            GGML_ASSERT(offset >= previous_end && size > 0);
            previous_end = offset + size;
            for (const auto & relation : interval["relationships"]) {
                GGML_ASSERT(relation["component"] == "repeating_layers");
                for (size_t byte = offset; byte < offset + size; ++byte) { actual[byte].insert(relation["layer_index"].get<int>()); }
            }
        }
        GGML_ASSERT(actual == expected && total(weights) == expected.size());
    }
}

int main() {
    test_sparse_mapping();
    test_shared_roles_and_owners();
    test_independent_copies();
    test_assignment_conflict();
    test_partial_and_unavailable();
    test_invalid_ranges();
    test_copy_and_independent_failure();
    test_random_relationships();
    std::puts("weight storage report: PASSED");
}
