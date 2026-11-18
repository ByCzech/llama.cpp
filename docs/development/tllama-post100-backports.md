# Selected post-v1.0.0 backports

Base: TLlama native `c0819bbbaeafa81e506eb83cf1b43f9f6106d307`, with official
llama.cpp `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (build 12359).
These are selected backports, not a merge of the current upstream branch.
Keep the packaging build number at 12359 and advance the TLlama snapshot
revision when pinning this tree.

| Upstream commit | PR | Change |
| --- | --- | --- |
| `b79696f1f556cb30b0c184838115007d929baca8` | #30021 | GCN MMQ configurations |
| `d20426bbbe32dde9c352e954d305105b0e3dd1f8` | #31990 | Narrow-N Vulkan cooperative matrix tiles |
| `d8f30a5b6db907110b6f8d67c2f84c1443b871ef` | #31323 | RDNA3/RDNA4 Vulkan prefill attention |
| `cf9a9957dae4801dde95d174ec1ac1b4cac84bf8` | #30492 | HTTP connection reuse after streaming |
| `05db09a4d3c3c1caa7cda31f98eca6ae28d92eb5` | #32344 | HIP MoE MMQ scheduling regression |
| `f477c1b90adb1f27e80276c0ebeed4d04812e605` | #32385 | AMD odd-column MUL_MAT_ID fallback |
| `301db633702f1d2c107d4feaa4268f07c0b508c5` | #32244 | Vulkan FP32 attention shared-memory limit |
| `47fe1333180210d01b000f9da614ea272f2f04c3` | #31327 | CDNA GDN MMA kernel |
| `dde9de82cbb9b1050ae75c95ab9a9f955340647f` | #31536 | SYCL attention mask broadcast indexing |
| `8f647f1fdb0c3bf72bf36e23641da2a28cea993f` | #32415 | Disable the new Vulkan flip shader on AMD Windows |
| `2e53a0bab751a288d7e48e642827cb4773b7b9c9` | #30022 | GCN Stream-K selection and configurations |
| `0a76757469ea94ad5f88d14e18cc7370b9931f99` | #31913 | Clear recurrent state with explicit stores |

The CDNA GDN backport retains the base's `__shfl_sync` and
`__shfl_up_sync` spelling at three call sites instead of importing the
unrelated repository-wide shuffle wrapper refactor from #31617. In this
base, the HIP aliases already dispatch to `__shfl` and `__shfl_up` and pass
the explicit wave width. Apart from these three equivalent spellings, the
GDN file matches upstream #31327.

The other eleven changes match the upstream patches. The Windows guard is
included as a correction to #31323 even though this package targets Linux.
Upstream authorship is retained in the backport commits.

The native memory API remains schema version 1. Buffer inventory, weight
storage, assignment and allocation-locality collectors are retained. The
new internal tensor enumerator from #31913 supplements the existing buffer
enumerator; it does not replace the TLlama memory visitor.

The offline tests additionally exercise a real tiny Qwen3.5 hybrid model
with poisoned recurrent state, compare the next request's logits with a
clean reference, and reuse one authenticated HTTP connection for two
streamed completions and Memory API reads. These cover failures reproduced
with the corresponding fixes absent.

CPU and host-glue checks do not qualify GPU compilation or execution.
Validate CUDA, ROCm, SYCL and Vulkan with their actual SDKs and hardware.
For legacy GCN qualification, retain the known mmap workaround; this patch
does not change load-mode defaults or fix kernel KFD behavior.

Audio/model additions, the DFlash refactor, slot scheduler refactoring and
batch API migration are outside this selected set.
