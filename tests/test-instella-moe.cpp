#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-memory-weight-storage.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

// Compare native absorbed MLA + KV cache against an independently materialized K/V oracle.
int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3);
    std::ifstream file(argv[2], std::ios::binary);
    uint32_t rows = 0, vocab = 0;
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&vocab), sizeof(vocab));
    GGML_ASSERT(rows == 7 && vocab > 0);
    std::vector<float> expected(rows * vocab);
    file.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
    GGML_ASSERT(file.good());
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
    }, nullptr);
    llama_backend_init();
    ggml_backend_dev_t devices[] = {nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    {
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        GGML_ASSERT(model && llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == (int) vocab);
        size_t visited = 0;
        GGML_ASSERT(llama_model_visit_weight_storage(model.get(), [&](const llama_memory_weight_storage_view & view) {
            GGML_ASSERT(!view.missing_reason && !view.ranges.empty());
            ++visited;
        }));
        GGML_ASSERT(visited > 30);
        float max_error[2] = {0, 0};
        for (bool flash : {false, true}) {
            for (int chunk : {1, 6}) {
                for (bool sparse : {false, true}) {
                    auto cp = llama_context_default_params();
                    cp.n_ctx = 128;
                    cp.n_batch = 16;
                    cp.n_ubatch = 16;
                    cp.n_threads = 2;
                    cp.n_threads_batch = 2;
                    cp.type_k = cp.type_v = GGML_TYPE_F32;
                    // The shared attention builder casts F32 K/V to F16 for flash attention.
                    cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
                    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
                    GGML_ASSERT(ctx);
                    llama_batch batch = llama_batch_init(16, 0, 1);
                    auto decode = [&](int begin, int count) {
                        batch.n_tokens = count;
                        for (int j = 0; j < count; ++j) {
                            batch.token[j] = begin + j + 1;
                            batch.pos[j] = begin + j;
                            batch.n_seq_id[j] = 1;
                            batch.seq_id[j][0] = 0;
                            batch.logits[j] = !sparse || count == 1 || j == 1 || j == count - 1;
                        }
                        GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
                        for (int j = 0; j < count; ++j) {
                            if (!batch.logits[j]) { continue; }
                            const float * actual = llama_get_logits_ith(ctx.get(), j);
                            GGML_ASSERT(actual);
                            for (uint32_t v = 0; v < vocab; ++v) {
                                const float ref = expected[(begin + j) * vocab + v];
                                const float error = std::abs(actual[v] - ref);
                                max_error[flash] = std::max(max_error[flash], error);
                                if (!std::isfinite(actual[v]) || error > (flash ? 1e-3f : 2e-4f) + (flash ? 1e-3f : 2e-4f) * std::abs(ref)) {
                                    std::fprintf(stderr, "flash=%d chunk=%d sparse=%d row=%d token=%u: actual=%g expected=%g error=%g\n",
                                        flash, chunk, sparse, begin + j, v, actual[v], ref, error);
                                    std::abort();
                                }
                            }
                        }
                    };
                    for (int begin = 0; begin < 6; begin += chunk) { decode(begin, chunk); }
                    decode(6, 1); // Incremental decode must read the prefill KV cache.
                    llama_batch_free(batch);
                }
            }
        }
        std::printf("Instella: 8 CPU cache/batch/FA paths passed; max absolute logit error FA-off %.9g, FA-on %.9g\n", max_error[0], max_error[1]);
    }
    llama_backend_free();
}
