# Model weight storage accounting

## Implementation stages

The first stage adds a private owner-thread visitor in `src/llama-memory-weight-storage.h`. It does not add a JSON section or change existing `/tllama/memory` fields. Buffer capacities and mmap spans retain their current meaning. The collector and interval deduplication are a subsequent change.

The visitor preserves the tensor identity, model part, native layer, buffer and checked loader assignment from placement inspection. It emits exact, sorted, disjoint byte ranges relative to that buffer. Padding and holes between strided rows are excluded. Quantized dimensions use storage blocks, including their scale/format bytes. Dense and transposed layouts do not require an element scan. A fragmented layout exceeding 4096 intermediate ranges is unavailable with `tensor_range_limit`; it never falls back to a dense span.

Ranges describe used tensor storage, not physical RAM/VRAM residency. The buffer object is the storage identity boundary in this stage. Two buffers with the same base address, offsets, names or content are separate storage. Different views of the same buffer can overlap; the collector must union their intervals and preserve each role only on the bytes that it covers. Allocation-locality providers are not required for this inspection.

`missing_reason == nullptr` proves the ranges, including an empty range list for an empty tensor. A non-NULL reason invalidates the entire range list. Placement and assignment coverage remain independent: a tensor can have exact storage ranges but an unknown role or assignment. `llama_model_visit_weight_storage()` returns false if any of these dimensions is incomplete, and may still emit usable records. A null/no-alloc/vocab-only model can return false without records. No-alloc is not a known zero-weight model.

CPU tensor extras can describe a repacked layout. Only a full dense tensor whose allocation-size contract preserves its storage size is counted by the generic resolver. Expanded repacks and views of repacked weights return `transformed_tensor_storage`; they need a repacker-specific mapping. The resolver does not mistake an original quantized layout for the storage of an expanded AMX/KleidiAI representation.

The resolver checks view chains, buffer association, quantization block shape, arithmetic overflow and buffer bounds. It rejects opaque/meta buffers before reading their base. Supported ordinary CPU, CUDA/HIP, Vulkan and SYCL buffers use buffer-relative tensor addresses; Vulkan's synthetic base is valid within one buffer and is never a cross-buffer identity. Opaque or split layouts require a proved tensor-to-shard mapping before they can be counted. A layout result is not evidence of physical aliasing or native allocator overhead.

Callbacks borrow tensor/buffer metadata at a safe point on the owner thread. They must not retain pointers beyond the capture or export addresses. Tensor data, backend allocation inspection and SDK queries are not used. The HTTP handler must continue to read an immutable snapshot rather than invoke this visitor.

## Planned additive JSON contract

This section defines the intended collector contract; it is not yet emitted by this first-stage patch.

- Keep schema version 1 and all existing buffer/placement/locality meanings. Add optional `weight_storage` with independent status, scope and `missing_components`. Absence means unsupported, never zero bytes.
- Initial scope is `model_weights`: main model, independently loaded draft model, and a model shared by main/MTP. Projector, LoRA, context state and MoE runtime cache/replicas are outside this scope. The inventory continues to describe them independently.
- Emit non-overlapping intervals per proved storage identity. Use snapshot-local opaque IDs and `buffer_id`, `offset_bytes`, `size_bytes`, checked assignment and owner/component/native-layer relationships. Names may be used only in separately bounded diagnostics, never as identity.
- Split intervals at every relationship boundary. Input [0,8) and output [4,12) require three intervals: [0,4) input, [4,8) input+output, [8,12) output. Their union is 12 bytes; excluding input-only intervals leaves 8 bytes. Role/owner subtotals need not be additive.
- Keep all model parts: input/output/repeating/MTP/other. Do not implement Ollama display policy or filter input roles in native code. Main/MTP sharing adds relationships rather than duplicate bytes. Separate actual copies remain separate.
- Preserve known intervals with explicit incomplete coverage. Unknown sizes or assignments must not become zero. Conflicting assignments on the same storage bytes require an explicit reason, not double counting as CPU and GPU.
- Deduplication across different buffer objects needs separate, explicit storage evidence. Do not infer it from allocation locality, pointer equality across devices, names, contents or host visibility.
- Cache static model metadata at the existing safe capture point. Reset it before owner destruction/replacement, including sleep, reload and failed load. IDs and intervals must not be joined across snapshots.

The collector contract uses this shape (example only; not emitted by this patch):

```json
{
  "weight_storage": {
    "scope": "model_weights",
    "deduplication": "buffer_ranges",
    "status": "known",
    "missing_components": [],
    "intervals": [{
      "id": "w1",
      "buffer_id": "b1",
      "offset_bytes": 64,
      "size_bytes": 128,
      "assignment": {
        "status": "known",
        "device_ids": ["d1"],
        "sources": ["loader_selection"],
        "reasons": []
      },
      "relationships": [{
        "owner_id": "o1",
        "component": "repeating_layers",
        "layer_index": 0
      }]
    }]
  }
}
```

`layer_index` is null for non-layer weights and a nonnegative native block index otherwise. Assignment reuses the existing placement-assignment contract, including partial/unavailable and empty device IDs where proof is missing. All IDs refer only to this snapshot; interval IDs are unique. Ranges are nonempty, bounded by the referenced buffer, sorted and disjoint within it. A known empty tensor contributes no interval. Relationships and missing-component records are unique.

Each missing-component record contains `owner_id` (nullable), `component`, `layer_index` (nullable), and `reason`. For example, a layer with an opaque storage mapping has a concrete native layer and `opaque_tensor_storage`; model inspection unavailable before load has a model-level reason. Known status requires complete storage, roles and assignment in scope, with no missing components. Partial retains proved intervals with explicit gaps in coverage. Unavailable has no usable storage evidence and an explicit reason; a proved unloaded state is known with an empty interval list. Assignment conflicts over one interval must remain explicit and cannot yield two copies of that interval. `deduplication = buffer_ranges` deliberately makes no claim about physical aliasing between buffer objects.

The collector patch must update the native HTTP test that currently asserts an exact top-level key set and qualify this shape against the client. Old schema-V1 responses remain valid. The client needs a separate adapter change to validate/use the new section and to avoid adding model buffer capacities to the same payload again. An unknown-field-tolerant reader alone does not use the new accounting.

## Compatibility audit

The TLlama reader at `a7cf2e970e4999a6c0fd28293fd2ad1329969732` explicitly ignores unknown fields in `_record()` and only recognizes current extensions in `_extensions()`. Its existing `test_v1_additions_are_ignored_and_new_role_component_reason_names_survive` covers this behavior. An optional top-level `weight_storage` is compatible with that reader, but is discarded until the dependent client update. `_native_memory_sizes()` still uses buffer totals for mixed placement at that revision.

This audit does not authorize changing the TLlama repository in this native patch. Before rollout, test the finalized new section with both the old reader and the updated adapter, including absent, partial and unavailable sections.

## Validation

The range tests use synthetic buffer-relative addresses and do not allocate or read the declared storage. They cover distant 1 MiB tensors in a 1 GiB span, quantized/strided/transposed/overlapping layouts, views, independent placement coverage, matching addresses in distinct buffers, opaque buffers, overflow and bounded enumeration. An independent byte-set oracle checks small randomized layouts.

Real CPU loader tests cover exact payload ranges for generated models, mmap on/off parity, tied input/output roles, MTP and no-alloc inspection. These tests do not establish hardware qualification for GPU backends, full cross-tensor interval deduplication, or the P100 mixed-placement case. Those remain required in subsequent stages.
