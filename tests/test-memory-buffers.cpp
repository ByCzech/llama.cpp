#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-view.h"
#include "../src/llama-memory-placement.h"
#include "../src/llama-adapter.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "gguf.h"
#ifdef TLLAMA_TEST_MEMORY_REPORT
#include "../tools/server/server-memory.h"
#endif

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using component = llama_memory_component;
struct buffer_snapshot {
    component component_id;
    ggml_backend_buffer_type_t type;
    size_t size;
};
using inventory = std::map<ggml_backend_buffer_t, buffer_snapshot>;

static void record(inventory & result, const llama_memory_buffer_view & view) {
    GGML_ASSERT(view.buffer);
    GGML_ASSERT(result.emplace(view.buffer, buffer_snapshot {
        view.component, ggml_backend_buffer_get_type(view.buffer), ggml_backend_buffer_get_size(view.buffer)
    }).second);
}

static size_t bytes(const inventory & buffers, component wanted) {
    size_t total = 0;
    for (const auto & entry : buffers) {
        if (entry.second.component_id == wanted) {
            total += entry.second.size;
        }
    }
    return total;
}

static inventory inspect(const llama_context * ctx) {
    inventory result;
    GGML_ASSERT(llama_context_visit_buffers(ctx, [&](const llama_memory_buffer_view & view) { record(result, view); }));
    return result;
}

#ifdef TLLAMA_TEST_MEMORY_REPORT
static void check_report(server_memory_native & collector, const server_memory_sources & sources, bool mapped = false) {
    std::map<ggml_backend_buffer_t, size_t> native;
    const auto collect = [&](const llama_memory_buffer_view & view) {
        native[view.buffer] = ggml_backend_buffer_get_size(view.buffer);
    };
    GGML_ASSERT(llama_model_visit_buffers(llama_get_model(sources.main), collect));
    GGML_ASSERT(llama_context_visit_buffers(sources.main, collect));
    if (sources.draft) {
        GGML_ASSERT(llama_model_visit_buffers(llama_get_model(sources.draft), collect));
        GGML_ASSERT(llama_context_visit_buffers(sources.draft, collect));
    }
    uint64_t expected = 0;
    for (const auto & entry : native) { expected += entry.second; }
    const auto report = collector.capture(sources);
    GGML_ASSERT(report["status"] == "known");
    GGML_ASSERT(report["placement"]["status"] == "known");
    GGML_ASSERT(report["buffers"].size() == native.size());
    uint64_t actual = 0;
    size_t mapped_buffers = 0, model_owners = 0;
    for (const auto & buffer : report["buffers"]) {
        GGML_ASSERT(buffer["memory_kind"] == "host");
        actual += buffer["size_bytes"].get<uint64_t>();
        mapped_buffers += buffer["backing"] == "mmap" ? 1 : 0;
    }
    for (const auto & owner : report["owners"]) {
        if (owner["kind"] == "model") {
            ++model_owners;
            GGML_ASSERT(owner["roles"].size() == (sources.draft ? 2 : 1));
        }
    }
    GGML_ASSERT(model_owners == 1); // both test contexts share their actual model
    GGML_ASSERT(actual == expected);
    GGML_ASSERT((mapped_buffers > 0) == mapped);
}
#endif

static gguf_context_ptr metadata(const std::string & arch) {
    gguf_context_ptr data(gguf_init_empty());
    gguf_set_val_str(data.get(), "general.architecture", arch.c_str());
    const auto key = [&](const char * name) { return arch + "." + name; };
    for (const auto & entry : std::map<std::string, uint32_t> {
            {"context_length", 256}, {"embedding_length", 64}, {"block_count", 2},
            {"feed_forward_length", 128}, {"attention.head_count", 4},
            {"attention.head_count_kv", 2}, {"vocab_size", 32}}) {
        gguf_set_val_u32(data.get(), key(entry.first.c_str()).c_str(), entry.second);
    }
    gguf_set_val_f32(data.get(), key("attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_f32(data.get(), key("attention.layer_norm_epsilon").c_str(), 1e-5f);
    gguf_set_val_str(data.get(), "tokenizer.ggml.model", "test");
    gguf_set_val_u32(data.get(), "tokenizer.ggml.token_type_count", 2);
    std::vector<std::string> tokens;
    std::vector<const char *> token_ptrs;
    for (int i = 0; i < 32; ++i) {
        tokens.push_back("tok_" + std::to_string(i));
    }
    for (const auto & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(data.get(), "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    if (arch == "qwen35") {
        for (const auto & entry : std::map<std::string, uint32_t> {
                {"full_attention_interval", 2}, {"ssm.conv_kernel", 4},
                {"ssm.inner_size", 64}, {"ssm.state_size", 16},
                {"ssm.time_step_rank", 4}, {"ssm.group_count", 2}}) {
            gguf_set_val_u32(data.get(), key(entry.first.c_str()).c_str(), entry.second);
        }
        const uint32_t sections[] = {4, 4, 4, 4};
        gguf_set_arr_data(data.get(), key("rope.dimension_sections").c_str(), GGUF_TYPE_UINT32, sections, 4);
    }
    return data;
}

static void initialize_tensor(ggml_tensor * tensor, void * user_data) {
    auto & buffers = *(std::set<ggml_backend_buffer_t> *) user_data;
    buffers.insert(tensor->buffer);
    // User models also create optional RoPE factors; zero factors cause division by zero.
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> values(ggml_nelements(tensor), 0.01f);
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}

static llama_model_params model_params() {
    auto params = llama_model_default_params();
    static ggml_backend_dev_t devices[] = {nullptr};
    params.devices = devices;
    params.n_gpu_layers = 0;
    params.use_extra_bufts = false;
    return params;
}

// A loader-only provider: CPU-backed storage, no GPU execution or hardware claims.
// It exercises the real candidate lists, supports_op rejection and allocation fallback.
struct assignment_provider {
    ggml_backend_reg reg = {};
    ggml_backend_device device = {};
    ggml_backend_buffer_type host = {}, gpu = {};
    bool reject_gpu = false;
    bool host_fallback = false;
    size_t rejected = 0;

    static bool uses_type(const ggml_tensor * tensor, ggml_backend_buffer_type_t type) {
        if (tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == type) { return true; }
        for (auto * src : tensor->src) {
            if (src && uses_type(src, type)) { return true; }
        }
        return false;
    }

    assignment_provider() {
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "TestAssignment"; };
        reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
            if (std::strcmp(name, "tllama_buffer_memory_flags") != 0) { return nullptr; }
            static ggml_backend_memory_buffer_flags_t flags = [](ggml_backend_buffer_t buffer) -> uint32_t {
                auto * type = ggml_backend_buffer_get_type(buffer);
                auto & provider = *static_cast<assignment_provider *>(type->context);
                return type == &provider.host ? GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED : 0;
            };
            return reinterpret_cast<void *>(flags);
        };
        device.reg = &reg;
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "TestAssignmentGPU"; };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_memory = [](ggml_backend_dev_t, size_t * free, size_t * total) {
            *free = *total = 1024 * 1024 * 1024;
        };
        device.iface.get_props = [](ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
            *props = {};
            props->name = ggml_backend_dev_name(dev);
            props->description = props->name;
            props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
            props->caps.host_buffer = true;
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<assignment_provider *>(dev->context)->gpu;
        };
        device.iface.get_host_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<assignment_provider *>(dev->context)->host;
        };
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * op) {
            auto & provider = *static_cast<assignment_provider *>(dev->context);
            const bool rejected = provider.reject_gpu && uses_type(op, &provider.gpu);
            provider.rejected += rejected;
            return !rejected;
        };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
            return buft->device == dev;
        };
        for (auto * buft : {&host, &gpu}) {
            buft->device = &device;
            buft->context = this;
            buft->iface.get_name = [](ggml_backend_buffer_type_t type) {
                auto & provider = *static_cast<assignment_provider *>(type->context);
                return type == &provider.host ? "TestAssignmentHost" : "TestAssignmentDevice";
            };
            buft->iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(64); };
            buft->iface.is_host = [](ggml_backend_buffer_type_t type) {
                return type == &static_cast<assignment_provider *>(type->context)->host;
            };
            buft->iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
                auto & provider = *static_cast<assignment_provider *>(type->context);
                auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
                if (buffer && !(type == &provider.host && provider.host_fallback)) {
                    buffer->buft = type;
                }
                return buffer;
            };
        }
    }
};

static void test_assignments(const char * path, bool tied = false) {
#ifndef TLLAMA_TEST_MEMORY_REPORT
    (void) tied;
#endif
    assignment_provider provider;
    ggml_backend_dev_t devices[] = {&provider.device, nullptr};
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu);
    enum scenario { full, partial, rejected, fallback, mapped, host_override, cpu_override, gpu_override };
    for (auto test : {full, partial, rejected, fallback, mapped, host_override, cpu_override, gpu_override}) {
        auto params = model_params();
        params.devices = devices;
        params.n_gpu_layers = test == partial ? 2 : 3;
        params.load_mode = test == mapped ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        provider.reject_gpu = test == rejected;
        provider.host_fallback = test == fallback;
        provider.rejected = 0;
        llama_model_tensor_buft_override overrides[] = {
            {"token_embd\\.weight", test == cpu_override ? ggml_backend_cpu_buffer_type() :
                                  test == gpu_override ? &provider.gpu : &provider.host},
            {nullptr, nullptr},
        };
        if (test == host_override || test == cpu_override || test == gpu_override) {
            params.tensor_buft_overrides = overrides;
        }
        llama_model_ptr model(llama_model_load_from_file(path, params));
        GGML_ASSERT(model);
        size_t inputs = 0, outputs = 0, cpu_layers = 0, gpu_layers = 0;
        GGML_ASSERT(llama_model_visit_placement(model.get(), [&](const llama_memory_placement_view & view) {
            GGML_ASSERT(view.known && view.buffer);
            bool on_cpu = view.part == LLAMA_MEMORY_MODEL_PART_INPUT || test == rejected ||
                (test == partial && view.part == LLAMA_MEMORY_MODEL_PART_REPEATING && view.layer == 0);
            if (test == gpu_override && view.part == LLAMA_MEMORY_MODEL_PART_INPUT) { on_cpu = false; }
            const auto & assignment = view.assignment;
            if (test == cpu_override && assignment.source == LLAMA_MEMORY_ASSIGNMENT_OVERRIDE) { on_cpu = true; }
            auto * expected = on_cpu ? cpu : &provider.device;
            if (test == host_override && assignment.source == LLAMA_MEMORY_ASSIGNMENT_OVERRIDE) {
                GGML_ASSERT(!assignment.device); // raw GPU-provided host storage does not prove assignment
                GGML_ASSERT(assignment.source == LLAMA_MEMORY_ASSIGNMENT_OVERRIDE);
            } else {
                if (assignment.device != expected) {
                    std::fprintf(stderr, "assignment scenario %d part %d layer %d: expected %s, got %s (%s)\n",
                        int(test), int(view.part), view.layer, ggml_backend_dev_name(expected),
                        assignment.device ? ggml_backend_dev_name(assignment.device) : "unknown",
                        ggml_backend_buffer_name(view.buffer));
                }
                GGML_ASSERT(assignment.device == expected);
            }
            if (on_cpu && test != mapped && test != fallback) {
                GGML_ASSERT(ggml_backend_buffer_get_type(view.buffer) == &provider.host);
                GGML_ASSERT(ggml_backend_buft_get_device(&provider.host) == &provider.device);
            }
            if (view.part == LLAMA_MEMORY_MODEL_PART_INPUT) {
                ++inputs;
                if (test == mapped) {
                    GGML_ASSERT(assignment.source == LLAMA_MEMORY_ASSIGNMENT_MMAP);
                    GGML_ASSERT(ggml_backend_buffer_get_type(view.buffer) != &provider.host);
                } else if (test == host_override || test == cpu_override || test == gpu_override) {
                    GGML_ASSERT(assignment.source == LLAMA_MEMORY_ASSIGNMENT_OVERRIDE);
                } else if (test != host_override) {
                    GGML_ASSERT(assignment.selection_device == &provider.device);
                    GGML_ASSERT(assignment.source == LLAMA_MEMORY_ASSIGNMENT_SELECTION);
                }
            } else if (view.part == LLAMA_MEMORY_MODEL_PART_OUTPUT) {
                ++outputs;
            } else if (view.part == LLAMA_MEMORY_MODEL_PART_REPEATING) {
                (on_cpu ? cpu_layers : gpu_layers)++;
            }
        }));
        GGML_ASSERT(inputs && outputs && (cpu_layers || gpu_layers));
        GGML_ASSERT((provider.rejected > 0) == (test == rejected));
        if (test == partial) { GGML_ASSERT(cpu_layers && gpu_layers); }
#ifdef TLLAMA_TEST_MEMORY_REPORT
        server_memory_collector collector, legacy;
        const auto owner = collector.owner(model.get(), "model", "main");
        const auto old_owner = legacy.owner(model.get(), "model", "main");
        GGML_ASSERT(llama_model_visit_buffers(model.get(), [&](const llama_memory_buffer_view & view) {
            collector.buffer(owner, view);
            legacy.buffer(old_owner, view);
        }));
        GGML_ASSERT(llama_model_visit_placement(model.get(), [&](const llama_memory_placement_view & view) {
            collector.placement(owner, view);
            auto old = view;
            old.assignment = {};
            legacy.placement(old_owner, old);
        }));
        const auto report = collector.finish();
        const auto before = legacy.finish();
        GGML_ASSERT(report["status"] == "known");
        GGML_ASSERT(report["buffers"] == before["buffers"] && report["owners"] == before["owners"]);
        GGML_ASSERT(report["placement"]["status"] == (test == host_override ? "partial" : "known"));
        if (test == full) {
            GGML_ASSERT(before["placement"]["status"] == "partial");
            GGML_ASSERT(report["devices"].size() == 2); // provider GPU and assigned CPU
            bool pinned = false;
            for (const auto & buffer : report["buffers"]) { pinned |= buffer["memory_kind"] == "host_pinned"; }
            GGML_ASSERT(pinned);
        }
        size_t layer_count = 0, input_count = 0, output_count = 0;
        for (const auto & group : report["placement"]["layer_groups"]) {
            const auto & component = group["component"];
            const auto & evidence = group["assignment"];
            std::set<std::string> assigned_types;
            for (const auto & id : evidence["device_ids"]) {
                bool found = false;
                for (const auto & device : report["devices"]) {
                    if (device["id"] == id) {
                        assigned_types.insert(device["type"].get<std::string>());
                        found = true;
                    }
                }
                GGML_ASSERT(found);
            }
            std::string expected = "accelerator";
            if (component == "input_weights") {
                ++input_count;
                expected = test == host_override ? "unknown" : test == gpu_override ? "accelerator" : "cpu";
                if (test == full) {
                    GGML_ASSERT(group["device_ids"] != evidence["device_ids"]);
                    GGML_ASSERT(evidence["sources"] == json::array({"loader_selection"}));
                }
            } else if (component == "output_weights") {
                ++output_count;
                if (test == rejected) { expected = "cpu"; }
                if (tied && test == cpu_override) { expected = "mixed"; }
                if (tied && test == host_override) { expected = "unknown"; }
            } else {
                GGML_ASSERT(component == "repeating_layers");
                for (const auto & index : group["layer_indices"]) {
                    ++layer_count;
                    const auto location = test == rejected || (test == partial && index == 0) ? "cpu" : "accelerator";
                    GGML_ASSERT(group["location"] == location);
                }
                expected = group["location"].get<std::string>();
            }
            GGML_ASSERT(group["location"] == expected);
            if (expected == "unknown") {
                GGML_ASSERT(evidence["status"] == (component == "output_weights" ? "partial" : "unavailable"));
                GGML_ASSERT(evidence["reasons"] == json::array({"assignment_unknown"}));
            } else {
                GGML_ASSERT(evidence["status"] == "known" && evidence["reasons"].empty());
                GGML_ASSERT(assigned_types == (expected == "mixed" ? std::set<std::string>{"cpu", "gpu"} :
                    std::set<std::string>{expected == "cpu" ? "cpu" : "gpu"}));
            }
        }
        GGML_ASSERT(layer_count == 2 && input_count == 1 && output_count == 1);
#endif
    }
    // A changed allocation must not turn a formerly selected GPU into a proved CPU assignment.
    auto data = metadata("llama");
    auto params = model_params();
    params.devices = devices;
    params.n_gpu_layers = 3;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    provider.reject_gpu = provider.host_fallback = false;
    struct init_data { std::set<ggml_backend_buffer_t> buffers; ggml_tensor * gpu_tensor = nullptr; } initialized;
    llama_model_ptr model(llama_model_init_from_user(data.get(), [](ggml_tensor * tensor, void * opaque) {
        auto & init = *static_cast<init_data *>(opaque);
        initialize_tensor(tensor, &init.buffers);
        if (!ggml_backend_buffer_is_host(tensor->buffer)) { init.gpu_tensor = tensor; }
    }, &initialized, params));
    GGML_ASSERT(model && initialized.gpu_tensor);
    auto * saved = initialized.gpu_tensor->buffer;
    auto * replacement = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 64);
    GGML_ASSERT(replacement);
    initialized.gpu_tensor->buffer = replacement;
    size_t invalidated = 0;
    GGML_ASSERT(llama_model_visit_placement(model.get(), [&](const llama_memory_placement_view & view) {
        if (view.buffer == replacement) {
            GGML_ASSERT(view.known && !view.assignment.device);
            ++invalidated;
        }
    }));
    GGML_ASSERT(invalidated);
    initialized.gpu_tensor->buffer = saved;
    ggml_backend_buffer_free(replacement);
    std::puts("memory placement: loader assignments PASSED");
}

#ifdef TLLAMA_TEST_MEMORY_REPORT
static void test_partial_assignment_report() {
    assignment_provider provider;
    ggml_backend_dev_t devices[] = {&provider.device, nullptr};
    auto data = metadata("llama");
    gguf_set_val_u32(data.get(), "llama.block_count", 40);
    auto params = model_params();
    params.devices = devices;
    params.n_gpu_layers = 20; // 19 repeating layers plus output
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    std::set<ggml_backend_buffer_t> initialized;
    llama_model_ptr model(llama_model_init_from_user(data.get(), initialize_tensor, &initialized, params));
    GGML_ASSERT(model);
    server_memory_collector collector;
    const auto owner = collector.owner(model.get(), "model", "main");
    GGML_ASSERT(llama_model_visit_buffers(model.get(), [&](const llama_memory_buffer_view & view) { collector.buffer(owner, view); }));
    GGML_ASSERT(llama_model_visit_placement(model.get(), [&](const llama_memory_placement_view & view) { collector.placement(owner, view); }));
    const auto report = collector.finish();
    GGML_ASSERT(report["status"] == "known" && report["placement"]["status"] == "known");
    std::set<int> cpu_layers, gpu_layers;
    bool input = false, output = false;
    for (const auto & group : report["placement"]["layer_groups"]) {
        GGML_ASSERT(group["assignment"]["status"] == "known");
        if (group["component"] == "repeating_layers") {
            auto & layers = group["location"] == "cpu" ? cpu_layers : gpu_layers;
            for (const auto & layer : group["layer_indices"]) { GGML_ASSERT(layers.insert(layer.get<int>()).second); }
        } else if (group["component"] == "input_weights") {
            input = group["location"] == "cpu";
        } else if (group["component"] == "output_weights") {
            output = group["location"] == "accelerator";
        }
    }
    GGML_ASSERT(input && output && cpu_layers.size() == 21 && gpu_layers.size() == 19);
    for (int layer = 0; layer < 40; ++layer) { GGML_ASSERT((layer < 21 ? cpu_layers : gpu_layers).count(layer)); }
}
#endif

static llama_context_ptr context(llama_model * model, bool embeddings = false) {
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 16;
    params.n_ubatch = 16;
    params.n_seq_max = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.offload_kqv = false;
    params.op_offload = false;
    params.embeddings = embeddings;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    GGML_ASSERT(ctx);
    return ctx;
}

static void check_native_sizes(const llama_context * ctx, const inventory & weights, const inventory & state) {
    struct sizes { size_t model = 0, context = 0, compute = 0; };
    std::map<ggml_backend_buffer_type_t, sizes> actual;
    for (const auto & entry : weights) {
        actual[entry.second.type].model += entry.second.size;
    }
    for (const auto & entry : state) {
        auto & size = actual[entry.second.type];
        if (entry.second.component_id == LLAMA_MEMORY_COMPONENT_KV_CACHE || entry.second.component_id == LLAMA_MEMORY_COMPONENT_RECURRENT_STATE) {
            size.context += entry.second.size;
        } else if (entry.second.component_id == LLAMA_MEMORY_COMPONENT_COMPUTE) {
            size.compute += entry.second.size;
        }
    }
    for (const auto & entry : llama_get_memory_breakdown(ctx)) {
        GGML_ASSERT(actual[entry.first].model == entry.second.model);
        GGML_ASSERT(actual[entry.first].context == entry.second.context);
        GGML_ASSERT(actual[entry.first].compute == entry.second.compute);
        actual.erase(entry.first);
    }
    GGML_ASSERT(actual.empty());
}

static void check_placement(const llama_model * model, int32_t expected_layers, bool mtp = false) {
    std::set<int32_t> layers;
    std::set<int32_t> mtp_layers;
    bool input = false;
    bool output = false;
    GGML_ASSERT(llama_model_visit_placement(model, [&](const llama_memory_placement_view & view) {
        GGML_ASSERT(view.known && view.buffer);
        GGML_ASSERT(ggml_backend_buffer_is_host(view.buffer));
        GGML_ASSERT(view.assignment.device);
        GGML_ASSERT(ggml_backend_dev_type(view.assignment.device) == GGML_BACKEND_DEVICE_TYPE_CPU);
        if (view.part == LLAMA_MEMORY_MODEL_PART_REPEATING) {
            layers.insert(view.layer);
        } else if (view.part == LLAMA_MEMORY_MODEL_PART_MTP) {
            mtp_layers.insert(view.layer);
        } else if (view.part == LLAMA_MEMORY_MODEL_PART_INPUT) {
            input = true;
        } else if (view.part == LLAMA_MEMORY_MODEL_PART_OUTPUT) {
            output = true;
        }
    }));
    GGML_ASSERT(input && output);
    GGML_ASSERT(layers.size() == (size_t) expected_layers);
    for (int32_t layer = 0; layer < expected_layers; ++layer) {
        GGML_ASSERT(layers.count(layer) == 1);
    }
    GGML_ASSERT(mtp_layers.size() == (mtp ? 1u : 0u));
    if (mtp) {
        GGML_ASSERT(mtp_layers.count(expected_layers) == 1);
    }
}

static void test_lora(llama_model * model) {
    ggml_context_ptr tensors(ggml_init({65536, nullptr, false}));
    gguf_context_ptr data(gguf_init_empty());
    gguf_set_val_str(data.get(), "general.architecture", "llama");
    gguf_set_val_str(data.get(), "general.type", "adapter");
    gguf_set_val_str(data.get(), "adapter.type", "lora");
    gguf_set_val_f32(data.get(), "adapter.lora.alpha", 2.0f);
    auto * a = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 64, 2);
    auto * b = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 2, 64);
    ggml_set_name(a, "blk.0.attn_q.weight.lora_a");
    ggml_set_name(b, "blk.0.attn_q.weight.lora_b");
    ggml_set_f32(a, 0.01f);
    ggml_set_f32(b, 0.01f);
    gguf_add_tensor(data.get(), a);
    gguf_add_tensor(data.get(), b);
    FILE * file = std::tmpfile();
    GGML_ASSERT(file && gguf_write_to_file_ptr(data.get(), file, false));
    std::rewind(file);
    auto * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    std::fclose(file);
    GGML_ASSERT(adapter);
    size_t owners = 0;
    GGML_ASSERT(llama_model_visit_loras(model, [&](const llama_adapter_lora * found) {
        GGML_ASSERT(found == adapter);
        ++owners;
    }));
    GGML_ASSERT(owners == 1);
    inventory buffers;
    GGML_ASSERT(llama_adapter_lora_visit_buffers(adapter, [&](const llama_memory_buffer_view & view) { record(buffers, view); }));
    size_t actual = 0;
    for (const auto & buffer : adapter->bufs) {
        GGML_ASSERT(buffers.count(buffer.get()) == 1);
        actual += ggml_backend_buffer_get_size(buffer.get());
    }
    GGML_ASSERT(actual > 0 && bytes(buffers, LLAMA_MEMORY_COMPONENT_ADAPTER_WEIGHTS) == actual);
    llama_adapter_lora_free(adapter);
    GGML_ASSERT(llama_model_visit_loras(model, [&](const llama_adapter_lora *) { GGML_ABORT("freed adapter visited"); }));
}

static void test_tied_output(const char * path) {
    ggml_context * raw_tensors = nullptr;
    gguf_context_ptr source(gguf_init_from_file(path, {false, &raw_tensors}));
    ggml_context_ptr tensors(raw_tensors);
    GGML_ASSERT(source && tensors);
    gguf_context_ptr tied(gguf_init_empty());
    gguf_set_kv(tied.get(), source.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        auto * tensor = ggml_get_tensor(tensors.get(), gguf_get_tensor_name(source.get(), i));
        GGML_ASSERT(tensor);
        if (std::string(tensor->name) != "output.weight") {
            gguf_add_tensor(tied.get(), tensor);
        }
    }
    const auto tied_path = std::string(path) + ".tied";
    GGML_ASSERT(gguf_write_to_file(tied.get(), tied_path.c_str(), false));
    llama_model_ptr model(llama_model_load_from_file(tied_path.c_str(), model_params()));
    GGML_ASSERT(model);
    check_placement(model.get(), 2);
    size_t output_roles = 0;
    GGML_ASSERT(llama_model_visit_placement(model.get(), [&](const llama_memory_placement_view & view) {
        output_roles += view.part == LLAMA_MEMORY_MODEL_PART_OUTPUT ? 1 : 0;
    }));
    GGML_ASSERT(output_roles >= 2); // output norm and the aliased embedding weight
    model.reset();
    test_assignments(tied_path.c_str(), true);
    GGML_ASSERT(std::remove(tied_path.c_str()) == 0);
}

struct placement_init_data {
    std::set<ggml_backend_buffer_t> buffers;
    ggml_tensor * tensor = nullptr;
};

static void test_missing_placement() {
    auto data = metadata("llama");
    placement_init_data initialized;
    llama_model_ptr model(llama_model_init_from_user(data.get(), [](ggml_tensor * tensor, void * user_data) {
        auto & data = *(placement_init_data *) user_data;
        initialize_tensor(tensor, &data.buffers);
        data.tensor = tensor;
    }, &initialized, model_params()));
    GGML_ASSERT(model && initialized.tensor);
    auto * tensor = initialized.tensor;
    auto * buffer = tensor->buffer;
    tensor->buffer = nullptr;
    size_t unknown = 0;
    GGML_ASSERT(!llama_model_visit_placement(model.get(), [&](const llama_memory_placement_view & view) {
        if (!view.known) {
            GGML_ASSERT(view.buffer == nullptr);
            GGML_ASSERT(!view.assignment.device);
            GGML_ASSERT(view.assignment.source == LLAMA_MEMORY_ASSIGNMENT_UNKNOWN);
            ++unknown;
        }
    }));
    GGML_ASSERT(unknown > 0);
    ggml_tensor source = {};
    source.buffer = buffer;
    tensor->view_src = &source;
    check_placement(model.get(), 2);
    tensor->view_src = nullptr;
    tensor->buffer = buffer;
}

static void test_model(const std::string & arch, const char * model_path) {
    auto data = metadata(arch);
    std::set<ggml_backend_buffer_t> initialized;
    llama_model_ptr model(llama_model_init_from_user(data.get(), initialize_tensor, &initialized, model_params()));
    GGML_ASSERT(model);
    inventory weights;
    GGML_ASSERT(llama_model_visit_buffers(model.get(), [&](const llama_memory_buffer_view & view) {
        record(weights, view);
        GGML_ASSERT(view.component == LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS);
        GGML_ASSERT(view.backing == LLAMA_MEMORY_BACKING_ALLOCATED);
        GGML_ASSERT(initialized.count(view.buffer) == 1);
    }));
    GGML_ASSERT(weights.size() == initialized.size());
    GGML_ASSERT(bytes(weights, LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS) > 0);
    check_placement(model.get(), 2);
    auto ctx = context(model.get(), arch == "bert");
    const auto initial = inspect(ctx.get());
    GGML_ASSERT(bytes(initial, LLAMA_MEMORY_COMPONENT_OUTPUT) > 0);
    GGML_ASSERT(bytes(initial, LLAMA_MEMORY_COMPONENT_COMPUTE) > 0);
    GGML_ASSERT((bytes(initial, LLAMA_MEMORY_COMPONENT_KV_CACHE) > 0) == (arch != "bert"));
    GGML_ASSERT((bytes(initial, LLAMA_MEMORY_COMPONENT_RECURRENT_STATE) > 0) == (arch == "qwen35"));
    check_native_sizes(ctx.get(), weights, initial);
#ifdef TLLAMA_TEST_MEMORY_REPORT
    server_memory_native collector;
    check_report(collector, {ctx.get()});
#endif
    const auto repeated = inspect(ctx.get());
    GGML_ASSERT(repeated.size() == initial.size());
    for (const auto & entry : initial) {
        GGML_ASSERT(repeated.count(entry.first) == 1);
    }

    if (arch == "llama") {
        test_lora(model.get());
        // The same model can have multiple contexts without creating another weight allocation.
        auto ctx_other = context(model.get());
#ifdef TLLAMA_TEST_MEMORY_REPORT
        check_report(collector, {ctx.get(), ctx_other.get(), nullptr, true});
#endif
        inventory shared_weights;
        GGML_ASSERT(llama_model_visit_buffers(model.get(), [&](const llama_memory_buffer_view & view) { record(shared_weights, view); }));
        GGML_ASSERT(shared_weights.size() == initialized.size());
        for (const auto & entry : shared_weights) {
            GGML_ASSERT(initialized.count(entry.first) == 1);
        }

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        auto batch = llama_batch_init(8, 0, 1);
        batch.n_tokens = 8;
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = 1;
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
        llama_batch_free(batch);
        const auto after_decode = inspect(ctx.get());
        GGML_ASSERT(bytes(after_decode, LLAMA_MEMORY_COMPONENT_OUTPUT) > bytes(initial, LLAMA_MEMORY_COMPONENT_OUTPUT));
        check_native_sizes(ctx.get(), weights, after_decode);
        const auto flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
        const auto state_size = llama_state_seq_get_size_ext(ctx.get(), 0, flags);
        GGML_ASSERT(state_size > 0);
        std::vector<uint8_t> saved(state_size);
        GGML_ASSERT(llama_state_seq_get_data_ext(ctx.get(), saved.data(), saved.size(), 0, flags) == state_size);
        GGML_ASSERT(bytes(inspect(ctx.get()), LLAMA_MEMORY_COMPONENT_STATE_OTHER) > 0);
        std::vector<float> control(64, 0.01f);
        GGML_ASSERT(llama_set_adapter_cvec(ctx.get(), control.data(), control.size(), 64, 1, 1) == 0);
        GGML_ASSERT(bytes(inspect(ctx.get()), LLAMA_MEMORY_COMPONENT_CONTROL_VECTOR) > 0);
#ifdef TLLAMA_TEST_MEMORY_REPORT
        // Static weights stay cached; output, saved state and control-vector allocations must refresh.
        check_report(collector, {ctx.get(), ctx_other.get(), nullptr, true});
        collector.reset(); // before either owner is released
#endif

        llama_model_save_to_file(model.get(), model_path);
        for (bool mmap : {false, true}) {
            auto params = model_params();
            params.n_gpu_layers = 99; // the explicit empty device list still puts real weights on CPU
            params.load_mode = mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
            llama_model_ptr loaded(llama_model_load_from_file(model_path, params));
            GGML_ASSERT(loaded);
            size_t total = 0;
            GGML_ASSERT(llama_model_visit_buffers(loaded.get(), [&](const llama_memory_buffer_view & view) {
                GGML_ASSERT(view.backing == (mmap ? LLAMA_MEMORY_BACKING_MMAP : LLAMA_MEMORY_BACKING_ALLOCATED));
                total += ggml_backend_buffer_get_size(view.buffer);
            }));
            GGML_ASSERT(total > 0);
            check_placement(loaded.get(), 2);
#ifdef TLLAMA_TEST_MEMORY_REPORT
            auto loaded_ctx = context(loaded.get());
            server_memory_native mapped_collector;
            check_report(mapped_collector, {loaded_ctx.get()}, mmap);
            mapped_collector.reset();
#endif
        }
        test_assignments(model_path);
        test_tied_output(model_path);
        GGML_ASSERT(std::remove(model_path) == 0);
    }
    auto params = model_params();
    params.no_alloc = true;
    llama_model_ptr preflight(llama_model_init_from_user(data.get(), initialize_tensor, &initialized, params));
    GGML_ASSERT(preflight);
    bool visited = false;
    GGML_ASSERT(!llama_model_visit_buffers(preflight.get(), [&](const llama_memory_buffer_view &) { visited = true; }));
    GGML_ASSERT(!visited);
    GGML_ASSERT(!llama_model_visit_placement(preflight.get(), [&](const llama_memory_placement_view &) { visited = true; }));
    GGML_ASSERT(!visited);
    std::printf("memory buffers: %s PASSED\n", arch.c_str());
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2 || (argc == 3 && std::string(argv[1]) == "--write-model"));
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            std::fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();
    if (argc == 3) {
        auto data = metadata("llama");
        std::set<ggml_backend_buffer_t> initialized;
        llama_model_ptr model(llama_model_init_from_user(data.get(), initialize_tensor, &initialized, model_params()));
        GGML_ASSERT(model);
        llama_model_save_to_file(model.get(), argv[2]);
        return 0;
    }
    GGML_ASSERT(!llama_model_visit_buffers(nullptr, [](const llama_memory_buffer_view &) {}));
    GGML_ASSERT(!llama_context_visit_buffers(nullptr, [](const llama_memory_buffer_view &) {}));
    for (const auto & arch : {"llama", "qwen35", "bert"}) {
        test_model(arch, argv[1]);
    }
    test_missing_placement();
#ifdef TLLAMA_TEST_MEMORY_REPORT
    test_partial_assignment_report();
#endif
    {
        auto data = metadata("qwen35");
        gguf_set_val_u32(data.get(), "qwen35.block_count", 3);
        gguf_set_val_u32(data.get(), "qwen35.nextn_predict_layers", 1);
        std::set<ggml_backend_buffer_t> initialized;
        auto params = model_params();
        params.load_mtp = true;
        llama_model_ptr model(llama_model_init_from_user(data.get(), initialize_tensor, &initialized, params));
        GGML_ASSERT(model);
        check_placement(model.get(), 2, true);
        std::puts("memory placement: MTP PASSED");
    }
    llama_backend_free();
}
