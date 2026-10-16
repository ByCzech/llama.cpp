# TLlama worker memory API, version 1

`GET /tllama/memory` reports allocated backend buffers and actual model-weight
placement in a single-model worker. It uses the worker's existing API-key
authentication (Bearer or X-Api-Key). The router does not expose this endpoint.
Responses contain `Cache-Control: no-store`.

## Availability and snapshots

The endpoint reads a copied, immutable snapshot, including during streaming
inference. It does not enqueue a measurement task, synchronize a device, or wake
a sleeping model. `snapshot_age_ms` indicates how old the snapshot is.

The first loaded snapshot is published before the worker becomes ready. Model
weights and placement are cached until unload/reload. Context and projector
buffers are inspected at safe batch, projector, request and reconfiguration
boundaries. Unchanged batch inventories do not publish another snapshot; request
completion and successful load publish one even when the inventory is unchanged.
These are allocation snapshots, not peak or continuously sampled measurements.

Before freeing owners, the inventory is invalidated. After unloading, a sleeping
or unloaded worker reports a known empty inventory. Collection failure reports
`unavailable` with `collection_failed`, rather than retaining a stale known report.

| HTTP status | Meaning |
| --- | --- |
| 200 | A V1 report, including partial or unavailable inventories. |
| 401 | Missing or invalid API key when authentication is configured. |
| 503 | Worker not ready, or no snapshot published yet. Existing readiness middleware runs before authentication. |
| 404 | Endpoint unsupported on a ready worker, or request sent to the router. |

To detect support, request this endpoint on a ready worker and verify
`schema_version == 1`. Authentication failure and temporary unavailability do not
mean the feature is unsupported. `runtime_fingerprint` identifies the executing
native build, independently of package versions.

## Report fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `runtime_fingerprint` | Native `llama_build_info()`, for example `b11552-<commit>`. |
| `scope` | `backend_buffers`. |
| `precision` | `exact_bytes`; non-null sizes come directly from native buffer sizes. |
| `status` | `known`, `partial`, or `unavailable` for backend-buffer coverage. |
| `missing_components` | Objects with `owner_id` (nullable), `component`, and `reason`. |
| `model_state` | `loading`, `loaded`, `unloading`, `unloaded`, `sleeping`, or `failed`. |
| `snapshot_sequence` | Increasing integer for the worker lifetime; preserved across sleep/reload. |
| `snapshot_stage` | `before_load`, `after_load`, `after_batch`, `after_projector`, `after_request`, `after_reconfigure`, `before_unload`, `after_unload`, or `after_load_failure`. |
| `snapshot_age_ms` | Nonnegative age using a monotonic clock. |
| `owners` | Objects with opaque `id`, `kind`, and `roles`. |
| `devices` | Objects with opaque `id`, `backend`, `name`, and `type`. Names/backend may be null. |
| `buffers` | Deduplicated leaf-buffer inventory, described below. |
| `placement` | Independent coverage status, missing components, and model-weight layer groups. |
| `allocation_locality` | Optional V1 extension for native allocation relationships and evidence, with independent coverage. |

Owner kinds are `model`, `context`, `projector`, and `adapter`. Roles identify
`main`, `draft`, `mtp`, `vision`, `audio`, `audio_generation`, or `lora`. One
model owner can serve both main and MTP contexts; its weight buffers appear once.
Loaded LoRA weights are counted even when the adapter is not enabled for a request.
Devices have type `cpu`, `gpu`, `igpu`, `accel`, `meta`, or `unknown`.

All IDs are opaque and local to a snapshot. They expose no native addresses and
must not be used to join separate snapshots. Distinct devices with identical
names remain distinct.

Each buffer has exactly these fields:

| Field | Meaning |
| --- | --- |
| `id` | Opaque buffer ID. |
| `owners` | Unique `{owner_id, component}` relationships. |
| `device_ids` | Native provider devices, not a claim about physical memory heaps. |
| `buffer_type` | Native buffer-type name, or null. |
| `memory_kind` | `host`, positively identified `host_pinned`, `device`, or `unknown`. |
| `backing` | `allocated`, `mmap`, or `unknown`; `imported` is reserved. |
| `size_bytes` | Native declared buffer capacity/span in bytes, or null when unavailable. |

Components include `model_weights`, `kv_cache`, `recurrent_state`, `state_other`,
`compute`, `output`, `moe_cache`, `control_vector`, `adapter_weights`,
`projector_weights`, and `projector_compute`. Saved on-device context states are
included as `state_other`. A legitimately absent component, such as KV cache in
an embeddings-only context, does not make coverage partial.

Generic multi-buffers are flattened and shared leaves counted once. Backend
composites whose allocations cannot be enumerated retain null sizes and make
coverage incomplete. Do not substitute zero or sum a buffer once per owner.
Typical missing reasons are `collector_not_implemented`, `opaque_composite`,
`size_unavailable`, `collection_failed`, and `transition_in_progress`.

## Weight placement

`placement` has `status`, `missing_components`, and `layer_groups`. Each group has
`owner_id`, `component`, `layer_indices`, `location`, and `device_ids`. Groups may
also contain the optional `assignment` object described below; V1 readers must
accept its absence in older workers. Group `device_ids` still identify buffer
providers.

Components are `input_weights`, `output_weights`, `repeating_layers`, `mtp_layers`,
or `other_weights`. Non-layer groups have an empty `layer_indices` array.
`location` is `cpu`, `accelerator`, `mixed`, or `unknown`, based on the final
loader assignment validated against actual tensor buffers. Older paths without
loader evidence retain the unambiguous native-buffer classification. The
requested GPU-layer count is never placement evidence. Unknown placement takes
precedence over a partial observation. Additional placement reasons include
`tensor_unclassified`, `device_unknown`, `opaque_distribution`, and
`assignment_unknown`.

The optional `assignment` object has these fields:

| Field | Meaning |
| --- | --- |
| `status` | `known` if all tensor roles in the group have proved assignments, `partial` if only some do, or `unavailable` if none do. |
| `device_ids` | Deduplicated proved assignment devices, referencing the same snapshot `devices` array. Can include known devices in a partial group. |
| `sources` | Deduplicated evidence sources: `loader_selection`, `loader_override`, `loader_mmap`, `loader_lazy`, or `native_buffer`. |
| `reasons` | Deduplicated reasons for missing assignment evidence; empty when known. |

Loader sources identify decisions retained for each tensor role. A source can
also describe an unresolved decision, such as an ambiguous raw host-buffer
override; consult `status` and `reasons`. A loader decision that became invalid
is not replaced by a native-buffer guess. `native_buffer` applies only when the
path supplied no loader evidence and the provider/type gives an unambiguous
classification. Opaque composites remain unknown even with a loader device.
Groups merge only when provider IDs, assignment IDs, sources, reasons and
coverage agree.

For example, CPU-assigned input weights can reside in a GPU-provided pinned
host buffer. Their group has `location=cpu`, provider `device_ids` referencing
the GPU, and `assignment.device_ids` referencing the CPU. The buffer retains
`memory_kind=host_pinned`, its GPU provider and its original `size_bytes`.
Devices used only as assignment evidence can therefore appear in `devices`
without providing a buffer. Neither assignment IDs nor provider IDs identify a
physical memory heap. A mixed group can have one provider and multiple proved
assignment devices.

Placement describes weight storage/backend assignment, not where every operation
executes. CPU context state and projector buffers remain visible independently;
weight-layer placement does not justify ignoring them.

## Allocation locality

Current workers also publish `allocation_locality`; older V1 workers can omit
it. Readers must tolerate both forms and additional optional fields. This
extension does not change `schema_version`, existing required fields, buffer
capacities, `memory_kind`, provider IDs or weight-assignment semantics.

| Field | Meaning |
| --- | --- |
| `status` | `known`, `partial`, or `unavailable` for native allocation-relationship coverage. |
| `scope` | `backend_buffer_allocations`; internal pools, library/driver workspaces and physical residency are outside this scope. |
| `physical_aliasing` | `unknown`: different native import/registration objects need not have disjoint physical backing. |
| `missing_components` | Deduplicated `{buffer_id, reason}` objects. Null buffer ID means some source inventory could not be inspected. |
| `allocations` | Deduplicated native allocation/import/registration objects with snapshot-local IDs. |
| `bindings` | Proved buffer-to-allocation relationships, including only available ranges. |

Coverage describes enumeration of relationships, not availability of every
attribute. A known inventory can include unknown sizes, device association,
locality or native properties. `partial` retains a proved subset; `unavailable`
means none of the missing inventory can be complemented by usable observations.
A known empty inventory after unload has `status=known` and empty arrays. A
loaded CPU or other unsupported backend instead reports missing entries and
`unavailable` (or `partial` alongside supported buffers). Missing locality never
changes valid buffer sizes or weight placement. Opaque buffer capacity can
coexist with complete native allocation relationships.

Each allocation contains:

| Field | Meaning |
| --- | --- |
| `id` | Opaque allocation ID local to this snapshot. |
| `device_ids` | Proved allocation association in the shared `devices` catalog; empty if unknown. This can differ from buffer providers and weight assignment. |
| `kind` | `device`, `host`, `host_pinned`, `managed`, `shared`, `system`, or `unknown`; this is a new metadata category, not the existing buffer `memory_kind`. |
| `origin` | `allocated`, `imported`, `registered`, or `unknown`. Registration is not additional allocated capacity. |
| `native_api` | Successful allocation/import/registration API name, or null. |
| `size_bytes` | Known native allocation span, or null. It can include padding beyond buffer capacity. |
| `locality` | `device_local`, `non_device_local`, or `unknown`, relative to the associated device. This does not measure physical residency. |
| `native_properties` | Known unsigned native attributes; absent keys are unknown, while zero values are valid. |

Published attribute names are `vulkan.memory_type_index`, `vulkan.heap_index`,
`vulkan.memory_property_flags`, `vulkan.heap_flags`, and
`level_zero.memory_ordinal`. They keep their native meanings. Unrecognized
native attributes are ignored rather than exposing arbitrary handles/tokens.
Vulkan DEVICE_LOCAL and HOST_VISIBLE bits are independent. CUDA/HIP ordinary
allocations and SYCL USM need no Vulkan heap/type index. Managed/shared/system
kind alone does not establish locality or CPU offload.

Each binding contains `buffer_id`, `allocation_id`, `buffer_offset_bytes`,
`allocation_offset_bytes`, and `size_bytes`. IDs reference this report's buffer
and allocation arrays. Each range field is independently nullable. Native
ranges are checked for overflow and against known buffer/allocation spans.
Unknown offsets or lengths must not be used to distribute buffer capacity among
heaps. Multiple buffers can bind one allocation, and a buffer can have several
bindings. Shared owners do not repeat the buffer or its binding.

The collector deduplicates by native allocation domain and identity, never by
provider, same-named devices or heap index. All native pointers/handles remain
inside owner-thread capture state; HTTP receives copied descriptions and opaque
IDs. Inconsistent descriptions for one identity withdraw that allocation and
all its bindings, with `conflicting_allocation_metadata` on affected buffers.
No first-observation-wins estimate is published. Other missing reasons include
`unsupported_backend`, `allocation_metadata_unavailable`,
`incomplete_allocation_metadata`, `invalid_metadata`, `collection_failed`, and
source-inventory/transition reasons. Failed or unsupported exporter results are
discarded even if they emitted callbacks.

Do not add allocation spans to the existing buffer total, or add bound ranges
to their backing allocation. The extension intentionally provides no total of
physical memory. Separate imports can alias external storage without a proved
shared identity. Native allocation metadata also says nothing about current
page migration, driver budgets or GPU execution time.

Static model metadata is cached until reset before unload/reload. Context and
projector metadata refresh with their buffer capture. Changes in locality alone
participate in snapshot equality and can publish a new sequence. Optional
exporter failures affect only locality; no device synchronization, SDK pointer
query or model wake-up is introduced. HTTP reads only the copied snapshot.

## Measurement scope

Buffer capacity, tensor payload, mapped span and physical residency are different
quantities. This API reports the first, with mmap backing explicitly identified.
It does not measure RSS, resident mmap pages, driver overhead, device-memory
budgets, arbitrary non-backend host allocations, or physical VRAM consumption.

An iGPU device buffer is accounted as a device buffer just like a dGPU buffer.
Using shared system RAM does not reclassify it as CPU memory or discount its size.
Likewise, GPU use of host-accessible heaps does not by itself establish CPU
execution. This interface does not infer physical heap ownership from device
type or buffer names.

## Tests

With CPU, tools, server, tests and mtmd enabled, build `llama-server`,
`test-memory-vulkan`, `test-memory-allocations`, `test-memory-buffers`, `test-memory-report`, and
`test-memory-snapshot`, then run:

```sh
ctest --test-dir build -R '^test-memory-(vulkan|allocations|buffers|report|snapshot|http)$' --output-on-failure
```

The HTTP test uses Python's standard library and a generated CPU GGUF model; it
downloads no model. It tests authentication, schema, mmap/allocated backing,
reads during an unbounded stream, request snapshots, sleep/reload, and router
exclusion. Native tests separately compare sizes with live buffer APIs and cover
shared weights, output growth, hybrid state, LoRA, missing and opaque buffers.
A CPU-backed test provider exercises the real loader's host assignment, partial
offload, overrides, fallback, mmap and tied-weight paths through JSON collection;
a 40-layer fixture checks the 0-20 CPU / 21-39 GPU boundary and GPU output.
Actual GPU inference and vision/projector runs require separate hardware checks.

The optional native allocation accessor is specified in
[the native ABI contract](../../docs/development/tllama-memory-allocations.md).
Its metadata/coverage fixtures need no GPU SDK or hardware. Report tests cover
native aliases, coverage independence, conflicts and exporter failures. Snapshot
and real-CPU registry fixtures cover metadata-only refresh, static cache reset,
sleep/reload, concurrent reads and copied data after native owners are released.
