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
`owner_id`, `component`, `layer_indices`, `location`, and `device_ids`.

Components are `input_weights`, `output_weights`, `repeating_layers`, `mtp_layers`,
or `other_weights`. Non-layer groups have an empty `layer_indices` array.
`location` is `cpu`, `accelerator`, `mixed`, or `unknown`, based on actual tensor
buffers, never the requested GPU-layer count. Unknown placement takes precedence
over a partial observation. Additional placement reasons include
`tensor_unclassified`, `device_unknown`, and `opaque_distribution`.

Placement describes weight storage/backend assignment, not where every operation
executes. CPU context state and projector buffers remain visible independently;
weight-layer placement does not justify ignoring them.

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
`test-memory-buffers`, `test-memory-report`, and `test-memory-snapshot`, then run:

```sh
ctest --test-dir build -R '^test-memory-(buffers|report|snapshot|http)$' --output-on-failure
```

The HTTP test uses Python's standard library and a generated CPU GGUF model; it
downloads no model. It tests authentication, schema, mmap/allocated backing,
reads during an unbounded stream, request snapshots, sleep/reload, and router
exclusion. Native tests separately compare sizes with live buffer APIs and cover
shared weights, output growth, hybrid state, LoRA, missing and opaque buffers.
Actual GPU inference and vision/projector runs require separate hardware checks.
