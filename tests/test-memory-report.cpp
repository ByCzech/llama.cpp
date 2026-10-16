#include "../tools/server/server-memory.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstring>
#include <cstdio>
#include <stdexcept>

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
    bool allocation_metadata = false, throw_allocation = false, fail_name_once = false;
    size_t allocation_probes = 0;
    int domain = 0, token = 0;
    ggml_backend_memory_allocation_status allocation_status = GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE;
    struct allocation_record {
        ggml_backend_memory_allocation allocation;
        ggml_backend_memory_binding binding;
        std::vector<ggml_backend_memory_allocation_property> properties;
    };
    std::vector<allocation_record> records;

    fixture(enum ggml_backend_dev_type kind, bool host = false, uint32_t flags = 0) : kind(kind), flags(flags), host(host) {
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.context = this;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "Fixture"; };
        reg.iface.get_proc_address = [](ggml_backend_reg_t r, const char * name) -> void * {
            auto * self = static_cast<fixture *>(r->context);
            if (std::strcmp(name, "tllama_buffer_memory_allocations_v1") == 0 && self->allocation_metadata) {
                static ggml_backend_memory_buffer_allocations_t inspect = [](
                        ggml_backend_buffer_t buffer, ggml_backend_memory_allocation_callback callback, void * user) {
                    auto & self = *static_cast<fixture *>(buffer->context);
                    ++self.allocation_probes;
                    if (self.throw_allocation) { throw std::runtime_error("fixture metadata failure"); }
                    for (const auto & record : self.records) {
                        auto allocation = record.allocation;
                        allocation.properties = record.properties.data();
                        allocation.property_count = record.properties.size();
                        callback(&allocation, &record.binding, user);
                    }
                    return self.allocation_status;
                };
                return reinterpret_cast<void *>(inspect);
            }
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
        dev.iface.get_name = [](ggml_backend_dev_t dev) {
            auto & self = *static_cast<fixture *>(dev->context);
            if (self.fail_name_once) { self.fail_name_once = false; throw std::runtime_error("fixture device failure"); }
            return "same-name";
        };
        dev.iface.get_type = [](ggml_backend_dev_t d) { return static_cast<fixture *>(d->context)->kind; };
        type.device = &dev;
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "same-buffer-name"; };
        type.iface.is_host = [](ggml_backend_buffer_type_t t) { return static_cast<fixture *>(t->context)->host; };
        buf.buft = &type;
        buf.context = this;
        buf.size = 4096;
        ggml_backend_memory_allocation allocation = {};
        allocation.identity_domain = &domain;
        allocation.identity = &token;
        allocation.device = &dev;
        allocation.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE;
        allocation.origin = GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED;
        allocation.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL;
        allocation.native_api = "vkAllocateMemory";
        allocation.size_known = true;
        allocation.size_bytes = 8192;
        records.push_back({allocation, {7, 0, 0, 4096}, {{"vulkan.memory_type_index", 0},
            {"vulkan.heap_index", 0}, {"vulkan.memory_property_flags", 3}, {"vulkan.heap_flags", 1}}});
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

static void test_assignment_evidence() {
    fixture cpu(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    fixture pinned(GGML_BACKEND_DEVICE_TYPE_IGPU, true, GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED);
    fixture opaque(GGML_BACKEND_DEVICE_TYPE_META, false, GGML_BACKEND_MEMORY_BUFFER_OPAQUE);
    server_memory_collector collector;
    const auto owner = collector.owner(&collector, "model", "main");
    collector.buffer(owner, view(&pinned.buf));
    collector.buffer(owner, view(&cpu.buf));
    const llama_memory_tensor_assignment selected = {&cpu.dev, &pinned.dev, &pinned.type, LLAMA_MEMORY_ASSIGNMENT_SELECTION};
    auto overridden = selected;
    overridden.source = LLAMA_MEMORY_ASSIGNMENT_OVERRIDE;
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &pinned.buf, true, selected});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 1, &pinned.buf, true, overridden});
    auto gpu = selected;
    gpu.device = &pinned.dev;
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 2, &pinned.buf, true, gpu});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 3, &pinned.buf, true, selected});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 3, &pinned.buf, true, gpu});
    auto unknown = overridden;
    unknown.device = nullptr;
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 4, &pinned.buf, true, unknown});
    // Invalidated loader evidence must not fall back to this buffer's CPU provider.
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 5, &cpu.buf, true, unknown});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 6, &cpu.buf, true});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 7, &opaque.buf, true, selected});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 8, &pinned.buf, false, selected});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 9, &pinned.buf, true, selected});
    collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 9, &pinned.buf, true, unknown});
    const auto result = collector.finish();
    GGML_ASSERT(result["status"] == "known" && total(result) == 8192);
    GGML_ASSERT(result["buffers"][0]["memory_kind"] == "host_pinned");
    GGML_ASSERT(result["buffers"][0]["device_ids"] == json::array({"d1"}));
    GGML_ASSERT(result["placement"]["status"] == "partial");
    const auto & groups = result["placement"]["layer_groups"];
    GGML_ASSERT(groups.size() == 10); // source, assignment and coverage differences prevent merging
    for (const auto & group : groups) {
        const auto index = group["layer_indices"][0].get<int>();
        const auto & evidence = group["assignment"];
        if (index == 0 || index == 1) {
            GGML_ASSERT(group["location"] == "cpu");
            GGML_ASSERT(group["device_ids"] == json::array({"d1"}));
            GGML_ASSERT(evidence["device_ids"] == json::array({"d2"}));
            GGML_ASSERT(evidence["sources"] == json::array({index == 0 ? "loader_selection" : "loader_override"}));
        } else if (index == 2) {
            GGML_ASSERT(group["location"] == "accelerator");
            GGML_ASSERT(evidence["device_ids"] == json::array({"d1"}));
        } else if (index == 3) {
            GGML_ASSERT(group["location"] == "mixed");
            GGML_ASSERT(evidence["device_ids"].size() == 2);
        } else if (index == 6) {
            GGML_ASSERT(group["location"] == "cpu");
            GGML_ASSERT(evidence["sources"] == json::array({"native_buffer"}));
        } else {
            GGML_ASSERT(group["location"] == "unknown");
            GGML_ASSERT(evidence["status"] == (index == 9 ? "partial" : "unavailable"));
            GGML_ASSERT(evidence["device_ids"].size() == (index == 9 ? 1u : 0u));
            GGML_ASSERT(evidence["reasons"] == json::array({index == 7 ? "opaque_distribution" :
                index == 8 ? "tensor_unclassified" : "assignment_unknown"}));
        }
    }
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

static void test_locality_aliases() {
    json saved;
    {
        fixture first(GGML_BACKEND_DEVICE_TYPE_GPU), second(GGML_BACKEND_DEVICE_TYPE_GPU);
        first.allocation_metadata = second.allocation_metadata = true;
        first.records[0].properties.push_back({"unsafe.handle", uint64_t(reinterpret_cast<uintptr_t>(&first.token))});
        second.records = first.records; // two providers/buffers, one proved allocation and association
        server_memory_collector collector;
        const auto main = collector.owner(&first, "model", "main");
        const auto draft = collector.owner(&second, "model", "draft");
        collector.buffer(main, view(&first.buf));
        collector.buffer(draft, view(&second.buf));
        collector.buffer(draft, view(&first.buf));
        collector.placement(main, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &first.buf, true});
        collector.placement(draft, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &second.buf, true});
        saved = collector.finish();
        const auto & locality = saved["allocation_locality"];
        GGML_ASSERT(saved["status"] == "known" && saved["placement"]["status"] == "known");
        GGML_ASSERT(total(saved) == 8192 && saved["buffers"].size() == 2);
        GGML_ASSERT(locality["status"] == "known" && locality["allocations"].size() == 1 && locality["bindings"].size() == 2);
        GGML_ASSERT(locality["allocations"][0]["size_bytes"] == 8192);
        GGML_ASSERT(locality["allocations"][0]["device_ids"] == json::array({"d1"}));
        GGML_ASSERT(saved["buffers"][1]["device_ids"] == json::array({"d2"}));
        GGML_ASSERT(locality["allocations"][0]["native_properties"]["vulkan.heap_index"] == 0);
        GGML_ASSERT(locality["bindings"][0]["allocation_id"] == locality["bindings"][1]["allocation_id"]);
        GGML_ASSERT(locality["bindings"][0]["size_bytes"] == 4096);
        GGML_ASSERT(first.allocation_probes == 1 && second.allocation_probes == 1);
        GGML_ASSERT(saved.dump().find("unsafe.handle") == std::string::npos);
        GGML_ASSERT(saved.dump().find(std::to_string(reinterpret_cast<uintptr_t>(&first.token))) == std::string::npos);
    }
    GGML_ASSERT(saved["allocation_locality"]["physical_aliasing"] == "unknown");
    GGML_ASSERT(saved["allocation_locality"]["allocations"][0]["native_api"] == "vkAllocateMemory");
}

static void test_locality_coverage() {
    fixture first(GGML_BACKEND_DEVICE_TYPE_GPU), second(GGML_BACKEND_DEVICE_TYPE_GPU);
    first.allocation_metadata = second.allocation_metadata = true;
    second.records[0].allocation.identity = first.records[0].allocation.identity; // different domains, equal token/heap/name
    second.allocation_status = GGML_BACKEND_MEMORY_ALLOCATION_PARTIAL;
    server_memory_collector collector;
    const auto owner = collector.owner(&collector, "model", "main");
    collector.buffer(owner, view(&first.buf));
    collector.buffer(owner, view(&second.buf));
    const auto report = collector.finish();
    GGML_ASSERT(report["status"] == "known" && total(report) == 8192);
    GGML_ASSERT(report["allocation_locality"]["status"] == "partial");
    GGML_ASSERT(report["allocation_locality"]["allocations"].size() == 2);
    GGML_ASSERT(report["allocation_locality"]["missing_components"][0]["reason"] == "incomplete_allocation_metadata");

    fixture opaque(GGML_BACKEND_DEVICE_TYPE_GPU, false, GGML_BACKEND_MEMORY_BUFFER_OPAQUE);
    opaque.allocation_metadata = true;
    server_memory_collector independent;
    const auto other = independent.owner(&opaque, "model", "main");
    independent.buffer(other, view(&opaque.buf));
    const auto separate = independent.finish();
    GGML_ASSERT(separate["status"] == "unavailable" && separate["buffers"][0]["size_bytes"].is_null());
    GGML_ASSERT(separate["allocation_locality"]["status"] == "known");

    fixture unknown(GGML_BACKEND_DEVICE_TYPE_GPU);
    unknown.allocation_metadata = true;
    auto & allocation = unknown.records[0].allocation;
    allocation.device = nullptr;
    allocation.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED;
    allocation.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN;
    allocation.native_api = "cudaMallocManaged";
    allocation.size_known = false;
    unknown.records[0].properties.clear();
    unknown.records[0].binding = {0, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    server_memory_collector managed;
    managed.buffer(managed.owner(&unknown, "model", "main"), view(&unknown.buf));
    const auto result = managed.finish()["allocation_locality"];
    GGML_ASSERT(result["status"] == "known");
    GGML_ASSERT(result["allocations"][0]["kind"] == "managed" && result["allocations"][0]["device_ids"].empty());
    GGML_ASSERT(result["allocations"][0]["size_bytes"].is_null() && result["allocations"][0]["native_properties"].empty());
    GGML_ASSERT(result["bindings"][0]["buffer_offset_bytes"].is_null() && result["bindings"][0]["size_bytes"].is_null());
}

static void test_locality_failures() {
    for (int scenario = 0; scenario < 6; ++scenario) {
        fixture source(GGML_BACKEND_DEVICE_TYPE_GPU);
        source.allocation_metadata = scenario != 0;
        source.throw_allocation = scenario == 1;
        if (scenario == 2) { source.allocation_status = GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE; }
        if (scenario == 3) { source.records[0].allocation.identity = nullptr; }
        if (scenario == 4) { source.records[0].properties.push_back(source.records[0].properties.front()); }
        if (scenario == 5) { source.allocation_status = GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED; }
        server_memory_collector collector;
        const auto owner = collector.owner(&source, "model", "main");
        collector.buffer(owner, view(&source.buf));
        collector.placement(owner, {LLAMA_MEMORY_MODEL_PART_REPEATING, 0, &source.buf, true});
        const auto result = collector.finish();
        GGML_ASSERT(result["status"] == "known" && result["placement"]["status"] == "known" && total(result) == 4096);
        GGML_ASSERT(result["allocation_locality"]["status"] == "unavailable");
        GGML_ASSERT(result["allocation_locality"]["allocations"].empty() && result["allocation_locality"]["bindings"].empty());
    }
    // Failed allocation-only device metadata must not poison the common provider catalog.
    fixture associated(GGML_BACKEND_DEVICE_TYPE_GPU), provider(GGML_BACKEND_DEVICE_TYPE_GPU);
    associated.allocation_metadata = provider.allocation_metadata = true;
    associated.records[0].allocation.device = &provider.dev;
    provider.fail_name_once = true;
    server_memory_collector catalog;
    const auto catalog_owner = catalog.owner(&catalog, "model", "main");
    catalog.buffer(catalog_owner, view(&associated.buf));
    catalog.buffer(catalog_owner, view(&provider.buf));
    const auto stable = catalog.finish();
    GGML_ASSERT(stable["status"] == "known" && total(stable) == 8192);
    GGML_ASSERT(stable["devices"].size() == 2 && stable["devices"][1]["id"] == "d2");
    GGML_ASSERT(stable["buffers"][1]["device_ids"] == json::array({"d2"}));
    GGML_ASSERT(stable["allocation_locality"]["status"] == "partial");

    fixture valid(GGML_BACKEND_DEVICE_TYPE_GPU);
    valid.allocation_metadata = true;
    auto invalid = valid.records[0];
    invalid.binding.allocation_offset = UINT64_MAX;
    invalid.binding.size_bytes = 2;
    valid.records.push_back(invalid);
    server_memory_collector partial;
    partial.buffer(partial.owner(&valid, "model", "main"), view(&valid.buf));
    const auto subset = partial.finish()["allocation_locality"];
    GGML_ASSERT(subset["status"] == "partial" && subset["bindings"].size() == 1);
    GGML_ASSERT(subset["missing_components"][0]["reason"] == "invalid_metadata");

    fixture conflict(GGML_BACKEND_DEVICE_TYPE_GPU);
    conflict.allocation_metadata = true;
    server_memory_collector collector;
    const auto owner = collector.owner(&valid, "model", "main");
    collector.buffer(owner, view(&valid.buf));
    conflict.records = {valid.records[0]};
    conflict.records[0].allocation.size_bytes++;
    collector.buffer(owner, view(&conflict.buf));
    const auto inconsistent = collector.finish()["allocation_locality"];
    GGML_ASSERT(inconsistent["status"] == "unavailable" && inconsistent["allocations"].empty() && inconsistent["bindings"].empty());
    size_t conflicts = 0;
    for (const auto & missing : inconsistent["missing_components"]) { conflicts += missing["reason"] == "conflicting_allocation_metadata"; }
    GGML_ASSERT(conflicts == 2);

    server_memory_collector empty;
    GGML_ASSERT(empty.finish()["allocation_locality"]["status"] == "known");
    empty.missing(empty.owner(&empty, "model", "main"), "inventory", "transition_in_progress");
    GGML_ASSERT(empty.finish()["allocation_locality"]["status"] == "unavailable");
}

int main() {
    ggml_backend_load_all();
    test_deduplication();
    test_device_classification();
    test_assignment_evidence();
    test_partial_and_unknown();
    test_locality_aliases();
    test_locality_coverage();
    test_locality_failures();
    std::puts("memory report PASSED");
}
