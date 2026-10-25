# Model weight storage accounting

## Capture and ranges

The private owner-thread visitor is in `src/llama-memory-weight-storage.h`. The server collector publishes its deduplicated ranges in the optional `weight_storage` section of `/tllama/memory`. Existing fields, buffer capacities and mmap spans retain their current meaning.

The visitor preserves the tensor identity, model part, native layer, buffer and checked loader assignment from placement inspection. It emits exact, sorted, disjoint byte ranges relative to that buffer. Padding and holes between strided rows are excluded. Quantized dimensions use storage blocks, including their scale/format bytes. Dense and transposed layouts do not require an element scan. A fragmented layout exceeding 4096 intermediate ranges is unavailable with `tensor_range_limit`; it never falls back to a dense span.

Ranges describe used tensor storage, not physical RAM/VRAM residency. The buffer object is the storage identity boundary in this stage. Two buffers with the same base address, offsets, names or content are separate storage. Different views of the same buffer can overlap; the collector must union their intervals and preserve each role only on the bytes that it covers. Allocation-locality providers are not required for this inspection.

`missing_reason == nullptr` proves the ranges, including an empty range list for an empty tensor. A non-NULL reason invalidates the entire range list. Placement and assignment coverage remain independent: a tensor can have exact storage ranges but an unknown role or assignment. `llama_model_visit_weight_storage()` returns false if any of these dimensions is incomplete, and may still emit usable records. A null/no-alloc/vocab-only model can return false without records. No-alloc is not a known zero-weight model.

CPU tensor extras can describe a repacked layout. Only a full dense tensor whose allocation-size contract preserves its storage size is counted by the generic resolver. Expanded repacks and views of repacked weights return `transformed_tensor_storage`; they need a repacker-specific mapping. The resolver does not mistake an original quantized layout for the storage of an expanded AMX/KleidiAI representation.

The resolver checks view chains, buffer association, quantization block shape, arithmetic overflow and buffer bounds. It rejects opaque/meta buffers before reading their base. Supported ordinary CPU, CUDA/HIP, Vulkan and SYCL buffers use buffer-relative tensor addresses; Vulkan's synthetic base is valid within one buffer and is never a cross-buffer identity. Opaque or split layouts require a proved tensor-to-shard mapping before they can be counted. A layout result is not evidence of physical aliasing or native allocator overhead.

Callbacks borrow tensor/buffer metadata at a safe point on the owner thread. They must not retain pointers beyond the capture or export addresses. Tensor data, backend allocation inspection and SDK queries are not used. The HTTP handler must continue to read an immutable snapshot rather than invoke this visitor.

## Additive JSON contract

New native snapshots include this section; it remains optional for clients reading older V1 runtimes.

- Keep schema version 1 and all existing buffer/placement/locality meanings. Add optional `weight_storage` with independent status, scope and `missing_components`. Absence means unsupported, never zero bytes.
- Initial scope is `model_weights`: main model, independently loaded draft model, and a model shared by main/MTP. Projector, LoRA, context state and MoE runtime cache/replicas are outside this scope. The inventory continues to describe them independently.
- Emit non-overlapping intervals per proved storage identity. Use snapshot-local opaque IDs and `buffer_id`, `offset_bytes`, `size_bytes`, checked assignment and owner/component/native-layer relationships. Names may be used only in separately bounded diagnostics, never as identity.
- Split intervals at every relationship boundary. Input [0,8) and output [4,12) require three intervals: [0,4) input, [4,8) input+output, [8,12) output. Their union is 12 bytes; excluding input-only intervals leaves 8 bytes. Role/owner subtotals need not be additive.
- Keep all model parts: input/output/repeating/MTP/other. Do not implement Ollama display policy or filter input roles in native code. Main/MTP sharing adds relationships rather than duplicate bytes. Separate actual copies remain separate.
- Preserve known intervals with explicit incomplete coverage. Unknown sizes or assignments must not become zero. Conflicting assignments on the same storage bytes require an explicit reason, not double counting as CPU and GPU.
- Deduplication across different buffer objects needs separate, explicit storage evidence. Do not infer it from allocation locality, pointer equality across devices, names, contents or host visibility.
- Cache static model metadata at the existing safe capture point. Reset it before owner destruction/replacement, including sleep, reload and failed load. IDs and intervals must not be joined across snapshots.

The collector uses this shape (IDs and numbers below are illustrative):

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

Unclassified tensors, missing buffer references and unavailable layouts produce missing-component records rather than invented relationships or byte counts. Exact geometry with an unknown assignment can remain in `intervals` with an explicit unavailable/partial assignment. Overlapping contradictory device assignments clear the interval's device IDs, add `assignment_conflict` and mark coverage partial. They never produce two copies of the same interval. A model owner whose weights were not inspected is unavailable, not a known empty model.

Static weight metadata is captured with the model inventory. A model shared by main/MTP is visited once while its owner keeps both roles. The interval sweep is cached; collector copies share this value state until a write. The state contains copied IDs, relationships and ranges, not live tensor pointers. Owner-thread reset before destruction invalidates it for unloading, sleeping, failed load and replacement. The immutable HTTP snapshot is independent of the inference task queue. An empty proved unloaded state has known coverage and no intervals; transitions and failures have unavailable coverage with a reason.

Old schema-V1 responses remain valid. The native HTTP test accepts the section as optional and validates its interval bounds and references when present. The client needs a separate adapter change to validate/use the new section and to avoid adding model buffer capacities to the same payload again. An unknown-field-tolerant reader alone does not use the new accounting.

## Optional tensor diagnostics

Start the worker with `--memory-tensor-diagnostics N` (environment: `LLAMA_ARG_MEMORY_TENSOR_DIAGNOSTICS`). The default is 0, disabled. Values outside 0-4096 are rejected. The setting is retained across sleep/reload. No new endpoint or authentication path is added.

When enabled, `weight_storage.tensor_diagnostics` adds `status`, `record_limit`, `range_limit`, `truncated`, `name_encoding`, `missing_components` and `records`. Record and range limits apply across all model owners in one snapshot. There are at most N records and 16384 copied ranges. A record is one tensor-role visit, not a unique storage allocation; tied input/output roles can produce multiple records for the same bytes. Do not add these sizes to the interval total.

Each record has `id`, `tensor_name`, `owner_id`, `component`, `layer_index`, `buffer_id`, `storage_status`, `storage_reason`, `size_bytes`, `ranges_complete`, `ranges` and `assignment`. IDs and names are copied metadata, not addresses. Names are bounded by the native GGML name array. `name_encoding = byte_escape` preserves printable ASCII, doubles a literal backslash and writes other bytes as backslash-x followed by two hexadecimal digits. This also makes malformed/non-UTF8 names safe to serialize. A missing tensor has a null name; an unproved role has a null component; a missing buffer reference has a null buffer_id.

`storage_status = unavailable` has a reason, null size, no ranges and ranges_complete false. Known geometry keeps the full per-tensor payload size even if the diagnostic range budget is exhausted; the stored ranges are then an explicitly incomplete prefix with ranges_complete false. Each range has offset_bytes and size_bytes relative to the record's buffer. Assignment is independent of geometry and uses the same checked loader evidence as the main section.

Limits add diagnostic_record_limit or diagnostic_range_limit to diagnostic missing_components and set truncated true. Collection failure adds diagnostic_collection_failed. These affect diagnostic coverage only, not the main weight_storage status, intervals or missing_components. Underlying storage/assignment failures remain visible in both coverages. A truncated=false flag alone does not prove completeness: inspect status and individual record fields too.

Capture runs with the static model inventory on the owner thread, never in the HTTP handler. Names are not copied and no diagnostic JSON field is emitted when disabled. Unload/failure clears the records before owner destruction; a proved unloaded state has known empty diagnostics, while loading/failure is unavailable with a reason. Changing the owner-thread setting invalidates the native model cache; the next capture/transition publishes the new setting.

For the P100 case, keep the same model/projector/draft, context, KV types, flash attention and offload parameters, and add `--memory-tensor-diagnostics 4096`. Save the complete raw endpoint and loader log using the same authenticated method as the original capture. Inspect the specific native layers without assuming what their CPU tensors are:

```bash
jq '.weight_storage.tensor_diagnostics.records[]
    | select(.layer_index == 39 or .layer_index == 40)
    | {tensor_name, component, layer_index, buffer_id, size_bytes,
       storage_status, storage_reason, ranges_complete, ranges, assignment}' memory.json
```

Resolve assignment device_ids against the same snapshot's devices. Use the deduplicated main intervals for totals, not the sum of diagnostic rows. Compare actual tensor evidence with the old mmap span; do not infer payload from the number of offloaded layers or from nvidia-smi. Diagnostic capture does not measure physical residency. This feature does not itself qualify the P100 case or any GPU backend.

## Client compatibility audit

The TLlama reader at `a7cf2e970e4999a6c0fd28293fd2ad1329969732` explicitly ignores unknown fields in `_record()` and only recognizes current extensions in `_extensions()`. Its existing `test_v1_additions_are_ignored_and_new_role_component_reason_names_survive` covers this behavior. An optional top-level `weight_storage` is compatible with that reader, but is discarded until the dependent client update. `_native_memory_sizes()` still uses buffer totals for mixed placement at that revision.

This native patch does not change the TLlama repository. The old reader is checked against native output; before rollout the updated adapter also needs tests for absent, partial and unavailable sections. No supported fingerprint should be added solely from the presence of this optional section.

## Validation

The range tests use synthetic buffer-relative addresses and do not allocate or read the declared storage. They cover distant 1 MiB tensors in a 1 GiB span, quantized/strided/transposed/overlapping layouts, views, independent placement coverage, matching addresses in distinct buffers, opaque buffers, overflow and bounded enumeration. An independent byte-set oracle checks small randomized layouts.

Collector tests cover the 1 GiB sparse span, byte-accurate role boundaries, shared owners, separate copies, conflicting/unknown assignments, invalid ranges and copy-on-write cache isolation. A second independent byte-set oracle checks randomized role overlaps. Real CPU loader tests cover mmap on/off, an independent draft, shared main/MTP owners and replacement with a different-size model. HTTP tests cover streaming reads and sleep/reload. Reporter tests cover failure after a previous snapshot, weight-only changes and concurrent snapshot reads. These tests do not establish GPU hardware qualification or explain the real P100 mixed-placement case; those remain required before release.
