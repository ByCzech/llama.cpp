# TLlama native allocation inspection, ABI v1

`ggml-backend-memory.h` provides optional, read-only allocation metadata through
`ggml_backend_memory_buffer_allocations()`. Backends can export registry procedure
`tllama_buffer_memory_allocations_v1` with type
`ggml_backend_memory_buffer_allocations_t`. Missing procedures are normal. The
versioned name fixes the C ABI; incompatible layouts need another procedure name.
No mandatory backend/device/buffer vtable entry is added.

This interface supplies native evidence for a later snapshot collector. Adding
it does not implement a backend exporter or change the V1 HTTP report, buffer
sizes, `memory_kind`, providers, weight assignments or inference behavior.

## Capture and coverage

Call only at safe owner-thread capture points while owners cannot free or
reallocate buffers. The exporter reads metadata saved by successful native
allocation/import/registration paths. It must not allocate backend memory,
query an SDK, synchronize work, change a current device/context or log. Backend
failures must return a coverage status, without throwing across the interface.
Callbacks must also return without throwing. HTTP handlers must read copied
snapshots, never call this live interface.

| Result | Meaning |
| --- | --- |
| `UNSUPPORTED` | No compatible exporter for this buffer. Discard any callback records. |
| `COMPLETE` | All allocation relationships of this buffer were enumerated. Individual attributes can still be unknown. No records is valid for a known empty buffer. |
| `PARTIAL` | Only a proved subset is available, possibly empty. Retain the emitted subset and mark this dimension incomplete. |
| `UNAVAILABLE` | Inspection failed or cannot supply a usable result. Discard any callback records. |

A null buffer or callback returns `UNAVAILABLE`. The dispatcher makes no
allocation inference from zero sizes, names, host visibility or memory flags.
A generic multi-buffer returns `UNSUPPORTED` without calling its first child's
exporter: its context has another layout. Use `ggml_backend_buffer_visit()` and
inspect each leaf, retaining every leaf's coverage. Existing leaf enumeration
and capacities are unchanged. Backend-specific composites may export proved
relationships and return `PARTIAL`; opaque capacity does not prevent inspection.

Coverage here is independent of backend-buffer capacity and weight placement.
A known capacity with unsupported allocation metadata remains a known capacity.
Exporters with complete relationship enumeration may omit heap, size, device or
range details without inventing values. Consumers must track attribute
availability separately from relationship coverage.

## Allocation identity and evidence

Each callback receives an allocation descriptor and one binding to the buffer
being inspected. Both records, strings and property arrays are borrowed only for
the callback. Copy values needed afterward. Native identities and device
pointers remain usable only while their owners live; expose neither addresses
nor native handles in JSON. Assign snapshot-local IDs when publishing copies.

`identity_domain` and `identity` are non-null stable native identities, compared
as a pair. A domain identifies an allocation namespace/context; an identity
identifies the owned allocation object within it. Use the same pair and consistent
allocation metadata for all bindings to that allocation, including different
buffers or providers. Never deduplicate by provider, device name, heap index or
the identity token alone. Distinct same-named devices and equal heap indices do
not imply equal allocations. Reset all such keys before owners are freed.

These identities describe native allocation/import/registration objects, not
physical payload equivalence. Separate imports or registrations can alias
external storage without proving the same identity. Do not infer physical-byte
totals or disjoint physical backing from different pairs.

- `device` is a proved allocation association, or null. It can differ from the
  buffer provider and says nothing about execution or current page residency.
- `kind` distinguishes `UNKNOWN`, `DEVICE`, `HOST`, `HOST_PINNED`, `MANAGED`,
  `SHARED`, and `SYSTEM` allocations. These are native metadata categories,
  independent of the existing HTTP `memory_kind` enum.
- `origin` distinguishes `UNKNOWN`, `ALLOCATED`, `IMPORTED`, and `REGISTERED`.
  Host registration is not another allocated capacity; describe it through
  `REGISTERED` with the proved host kind.
- `native_api` names the successful native API, or is null. Report the final
  path, including managed-to-device or pinned-to-CPU fallback, rather than an
  environment request or a failed attempt.
- `size_bytes` is valid only with `size_known=true`. A known zero differs from
  unknown. This native allocation span can differ from ggml buffer capacity.
- `locality` is `UNKNOWN`, `DEVICE_LOCAL`, or `NON_DEVICE_LOCAL`, relative to a
  proved `device`. Non-unknown locality requires that device association and
  native evidence. It is not physical residency, a performance prediction, or
  a reason to reclassify CPU/GPU weight assignment. Managed/shared/system kinds
  alone do not establish locality.

## Optional native attributes

`properties` is an optional array of unique, non-null names and unsigned 64-bit
values. `property_count=0` allows a null array. Names are scoped to the native
API; presence means the value is known, including zero. Unknown names can be
ignored. Missing values are not zero. V1 attribute conventions are:

| Name | Native evidence |
| --- | --- |
| `vulkan.memory_type_index` | Type accepted by the successful allocation/import. |
| `vulkan.heap_index` | Heap selected by that memory type on that device. |
| `vulkan.memory_property_flags` | Raw property flags of that selected type. |
| `vulkan.heap_flags` | Raw flags of the selected heap. |
| `level_zero.memory_ordinal` | Ordinal accepted by the direct Level Zero allocation path. |

Vulkan `DEVICE_LOCAL` and `HOST_VISIBLE` bits are independent. Preserve raw flags
and their native meaning. CUDA/HIP ordinary allocations have no required Vulkan
heap/type equivalent; kind, API, device association and span are useful without
these properties. SYCL USM kinds also need no heap index. A Level Zero ordinal
is not a Vulkan heap index. Future unsigned native properties can use new
names without changing this ABI; other value types require a new ABI.

## Buffer bindings

`known_fields` independently marks `BUFFER_OFFSET`, `ALLOCATION_OFFSET`, and
`SIZE`. All values are unsigned byte offsets/lengths. Unknown fields must not be
consumed, even if nonzero. Zero known fields still describes a relationship
without a proved range. Emit each distinct relationship once; an allocation can
have several bindings or buffer aliases.

Only publish or use ranges that native ownership proves. Consumers must check
additions for overflow and bounds against known allocation/buffer spans before
using ranges. An unknown range does not justify distributing capacity among
heaps. Allocation spans and bound subranges are not extra buffer capacity and
must not be added to the existing backend-buffer total. Virtual reservations,
scratch pools and library workspaces outside ggml buffers require another scope.
