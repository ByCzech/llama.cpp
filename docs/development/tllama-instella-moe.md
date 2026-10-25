# Experimental Instella-MoE support

This native fork supports the `InstellaMoEForCausalLM` family experimentally.
The initial target is AMD Instella-MoE-16B-A3B-Think. This does not imply support
for every model named Instella or qualify every quantization/backend combination.

The implementation adapts Csaba Kecskemeti's `instella-moe` branch commit
`e90a9ad690643e49eec368e12f6b4f9f7f197801`, submitted as
[llama.cpp PR 26467](https://github.com/ggml-org/llama.cpp/pull/26467), to the
pinned native fork. It is a community implementation, not an upstream merge.

## Model behavior

The graph uses existing GGML operations and the compressed MLA cache. It keeps
the direct query projection, sigmoid expert routing with correction bias,
shared experts and leading dense blocks. It adds:

- A sigmoid attention gate, computed from the normalized attention input,
  applied after value expansion and before the output projection.
- Full FarSkip connectivity. The full residual feeds the FFN; a second residual
  excluding the preceding layer's routed experts feeds attention. Attention and
  FFN use the previous block's inputs rather than sequential residual updates.
- Adjacent-pair interleaved RoPE, including the target checkpoint's YaRN setup.

The final block gathers the attention result, attention gate and full residual
consistently when only some output rows are requested. Per-layer expert FFN
sizes and expert top-k counts are read from the current metadata representation.
Tied output embeddings use the existing loader mechanism.

## Scope and limitations

The converter accepts full-range FarSkip, gated attention, a direct query
projection, bias-free attention, SiLU, sigmoid routing and contiguous MoE blocks without MTP.
Unsupported variants fail conversion explicitly. Native loading validates MLA,
head and shared-expert dimensions and rejects query LoRA and MTP metadata.
Tensor split mode is disabled for this architecture. Row/layer split modes,
LoRA, control vectors, context shifting and non-default cache quantizations
have not been qualified by the synthetic CPU test.

The patch does not add custom GPU kernels. CPU correctness does not qualify
CUDA, HIP, Vulkan, SYCL or Metal. In particular, PR discussion reported incorrect
SYCL results with a Q8_0 3-D `attn_k_b` matrix whose row length is 96. Historical
reports used `-ot attn_k_b=CPU` as a workaround. Neither a current SYCL fix nor
that workaround is qualified by this patch; preserve this as a test item.

## Reproducible offline checks

Build `llama-server` and `test-instella-moe` with tests enabled. The Python
harness requires NumPy and the repository's `gguf-py`, with no torch package or
remote model download:

```sh
python3 tests/test-instella-moe.py \
  --runner build/bin/test-instella-moe --server build/bin/llama-server
```

When Python and NumPy are present at configure time, the server build registers
`test-instella-moe-numeric` with CTest. Generated small deterministic models
retain the target's 96-wide non-RoPE key projection. The independent NumPy
reference expands K/V before attention, whereas the native graph absorbs the
key projection into Q and uses a compressed KV cache. Tests compare logits for
single-token and batched prefill, a subsequent cached decode, dense and sparse
output requests, and flash attention on/off. Variants include tied embeddings,
YaRN, per-layer top-k and Q8_0 key expansion weights.

Counterfactual reference graphs omit gating, use a single residual stream or
make the FFN sequential; each must differ measurably from the correct oracle.
Additional tests cover converter guards and tensor mapping, invalid GGUFs,
tensor-split rejection, the authenticated memory endpoint and SSE generation.
The converter guard test executes the actual method with a stub base; it does
not replace a full HF checkpoint conversion test.

Reference sources for equations:

- [AMD model implementation](https://huggingface.co/amd/Instella-MoE-16B-A3B-Think/blob/74d28c1a8425583d7805b58e2caa9c13d6b62929/modeling_instella_moe.py)
- [AMD configuration](https://huggingface.co/amd/Instella-MoE-16B-A3B-Think/blob/74d28c1a8425583d7805b58e2caa9c13d6b62929/config.json)
- [Transformers DeepseekV3 v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)
- [Transformers RoPE v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py)

Before qualifying real-model support, convert/load the intended checkpoint,
verify tokenizer/chat-template behavior and coherent text, compare an agreed
prompt with the reference, and repeat on each distributed backend/quantization.
Do not treat synthetic test vocabulary output as coherent-model validation.
