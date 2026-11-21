# TLlama native Inkling 2 Mini backport

Base: `e148b6dbe6e08911e59b60fefd8562fcf73e80a5`, the published
`1.0.0~dev+b12359.2-1` runtime. The build number stays 12359; the commit
fingerprint changes. Rebuild the core and every backend from this snapshot.

## Selected upstream changes

| Commit | Purpose |
| --- | --- |
| `5c008e00085f4d3e394f9a1be649284999de1326` (#32217) | Inkling 2 text, tokenizer, still-image encoder, conversion and per-head CUDA attention masks |
| `8af68d486a9473820d24d00af8f33334ca80124d` (#32429) | Relative-position masks from the same cache cells, hybrid ISWA graph reuse |
| `a1f4222608594c57202fdaa6a45dca1ef05abc62` (#32657) | Merged gate/up expert scales |
| `29b52d398b294509746ec0a3f28948a3107c0213` (#32694) | Mini loop graph, controllers and physical-layer GGUF layout |
| `690b87d88151dbd775bb6eff5581663688785928` (#32704) | Share completed physical weights after optional scales are attached |
| `e1925b3695b8d2f2acd162fb10ad0748da0d921b` (#32366) | Sorted JSON in the original tool template |
| `dd099a6bf30c50627f6ae18915fbb687539a704d` (#32294) | Move image markers out of text blocks |
| `4ddf2b9470b44b122e8a5404ae199f04cd03bcec` (#32381) | Dedicated content, reasoning and tool parser with grammar triggers |

The backport resolves registry context around existing Instella-MoE and retains
the v1.0 graph node budget implementation. Both Inkling architectures receive
the upstream 32-nodes-per-tensor budget without the later generic model budget
refactor. No later scheduler, audio, public API or SDK upgrade is included.

## Supported input

Use the physical-layer `inkling2_loop` layout from
<https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF> for the initial qualification.
The IQ4_NL checkpoint uses 24 stored layers and 48 visits. KV and convolution
state are separate for each visit; repeated visits share their physical weights.
Use the matching `mmproj-Inkling-2-Mini-{BF16,Q8_0}.gguf` for images.

Do not assume compatibility with a converter that declares 48 stored blocks but
contains only 24 physical weight blocks. Draft PR #32695 uses a different loop
layout; it is not part of this backport. Do not change GGUF metadata by hand.
The merged scale fix alone does not translate another converter's layout.

The architecture also implements the upstream varying-loop metadata, but the
research checkpoint is outside the initial Mini qualification. An MTP/DSpark
drafter, maximum context, and performance claims require separate validation.

## Local checks

Build `llama-server`, `test-llama-archs`, `test-mtmd-impl`, and
`test-backend-ops`. Run:

```sh
./build/bin/test-llama-archs -a '^inkling2(_loop)?$' -b CPU
./build/bin/test-mtmd-impl
./build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CPU
```

The loop fixture uses three passes over a two-layer core, and includes merged
expert weights with per-expert scales. It checks scale and weight identity in
all extra slots before inference. Moving weight sharing before the generic
scale pass must fail this test. Backend comparisons also exercise independent convolution and KV state,
and verify rejection of unsupported mixed token/embedding batches.

The upstream parser tests cover text, reasoning, partial streaming, parallel and
forced tools, JSON responses, and image marker placement. The generic template
analyzer alone is insufficient for this format: it can mistake a sample tool
name for a literal grammar trigger. Use the specialized Inkling parser.

Actual Mini IQ4_NL text, streaming, image input, tools, reasoning parsing and
GPU inference remain deployment qualification. Passing a synthetic graph does
not establish model quality or support on every hardware backend. Preserve the
Memory API V1 compatibility checks when adopting the new fingerprint in TLlama.
