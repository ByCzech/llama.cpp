#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-view.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <cstdio>
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
    auto ctx = context(model.get(), arch == "bert");
    const auto initial = inspect(ctx.get());
    GGML_ASSERT(bytes(initial, LLAMA_MEMORY_COMPONENT_OUTPUT) > 0);
    GGML_ASSERT(bytes(initial, LLAMA_MEMORY_COMPONENT_COMPUTE) > 0);
    GGML_ASSERT((bytes(initial, LLAMA_MEMORY_COMPONENT_KV_CACHE) > 0) == (arch != "bert"));
    GGML_ASSERT((bytes(initial, LLAMA_MEMORY_COMPONENT_RECURRENT_STATE) > 0) == (arch == "qwen35"));
    check_native_sizes(ctx.get(), weights, initial);
    const auto repeated = inspect(ctx.get());
    GGML_ASSERT(repeated.size() == initial.size());
    for (const auto & entry : initial) {
        GGML_ASSERT(repeated.count(entry.first) == 1);
    }

    if (arch == "llama") {
        // The same model can have multiple contexts without creating another weight allocation.
        auto ctx_other = context(model.get());
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

        llama_model_save_to_file(model.get(), model_path);
        for (bool mmap : {false, true}) {
            auto params = model_params();
            params.load_mode = mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
            llama_model_ptr loaded(llama_model_load_from_file(model_path, params));
            GGML_ASSERT(loaded);
            size_t total = 0;
            GGML_ASSERT(llama_model_visit_buffers(loaded.get(), [&](const llama_memory_buffer_view & view) {
                GGML_ASSERT(view.backing == (mmap ? LLAMA_MEMORY_BACKING_MMAP : LLAMA_MEMORY_BACKING_ALLOCATED));
                total += ggml_backend_buffer_get_size(view.buffer);
            }));
            GGML_ASSERT(total > 0);
        }
        GGML_ASSERT(std::remove(model_path) == 0);
    }
    auto params = model_params();
    params.no_alloc = true;
    llama_model_ptr preflight(llama_model_init_from_user(data.get(), initialize_tensor, &initialized, params));
    GGML_ASSERT(preflight);
    bool visited = false;
    GGML_ASSERT(!llama_model_visit_buffers(preflight.get(), [&](const llama_memory_buffer_view &) { visited = true; }));
    GGML_ASSERT(!visited);
    std::printf("memory buffers: %s PASSED\n", arch.c_str());
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            std::fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();
    GGML_ASSERT(!llama_model_visit_buffers(nullptr, [](const llama_memory_buffer_view &) {}));
    GGML_ASSERT(!llama_context_visit_buffers(nullptr, [](const llama_memory_buffer_view &) {}));
    for (const auto & arch : {"llama", "qwen35", "bert"}) {
        test_model(arch, argv[1]);
    }
    llama_backend_free();
}
