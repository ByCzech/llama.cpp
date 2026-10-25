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
| `cuda.device_integrated` | Saved cudaDeviceProp::integrated, normalized to 0/1, for the proved allocation device. |
| `hip.device_integrated` | Saved hipDeviceProp_t::integrated, normalized to 0/1, for the proved allocation device. |
| `level_zero.memory_ordinal` | Ordinal accepted by the direct Level Zero allocation path. |

Vulkan `DEVICE_LOCAL` and `HOST_VISIBLE` bits are independent. Preserve raw flags
and their native meaning. CUDA/HIP ordinary allocations have no required Vulkan
heap/type equivalent; kind, API, device association and span are useful without
these properties. SYCL USM kinds also need no heap index. A Level Zero ordinal
is not a Vulkan heap index. Future unsigned native properties can use new
names without changing this ABI; other value types require a new ABI.

## Shared metadata helpers

`ggml/src/ggml-backend-memory-record.h` is an internal SDK-free helper, not a
new public ABI. The Vulkan adapter uses it; CUDA/HIP and SYCL can use the same
rules when their exporters are attached.

`ggml_backend_memory_record_attempt` clears the selected metadata before the
native attempt. It commits a copy of the candidate only when the caller's
success predicate accepts the actual result. The record has a `known` member;
it must contain the metadata for that specific attempt. Failed managed or
pinned attempts must not survive a later fallback. Native calls and any advice
or fallback decisions remain in the backend allocation path, never inspection.
The helper supports status-returning APIs and nullable/move-only handles.

`ggml_backend_memory_emit_range` accepts an already proved allocation descriptor
with a known span and a bounded buffer-to-allocation relationship. It validates
both domains' offsets using subtraction before emitting a binding, including
zero ranges and UINT64_MAX boundaries. It requires native identity pairs and
rejects non-unknown locality without a proved device. A managed/shared kind
does not create a locality claim. APIs without a known span can use the ABI's
independently known fields directly; this bounded helper returns UNAVAILABLE.

Its COMPLETE result validates one binding, not coverage of the entire buffer
or all backend allocations. Providers remain responsible for whole-buffer
coverage and partial/unsupported status. A throwing callback produces UNAVAILABLE;
the collector must discard all callback results, as required by the ABI.

These helpers own no native memory, identity registry or SDK context. Owners
must keep identity pairs stable for a live allocation, share them across views,
clear records at free/reuse and invalidate cached keys before owner destruction.
They must retain the actual successful allocation path. Tests of the helpers do
not qualify the owner lifecycle or any native API on hardware.

`test-memory-record` checks retry/fallback provenance, clearing on failure or
exception, handle success predicates, alias-safe metadata candidates, bounded
views, equal identity tokens in different domains, unknown association/locality,
known zero, overflow rejection before callback and callback exceptions.
`test-memory-vulkan` also exercises the production adapter using these helpers.

## Exporter coverage work list

Keep source implementation, SDK compilation and hardware tests separate. The
following records source coverage after the initial CUDA/HIP exporter step,
not a hardware support matrix. SDK-free tests do not qualify GPU execution.

| Backend / path | Source status | Remaining qualification |
| --- | --- | --- |
| Vulkan ordinary, pinned host and successful host imports | Existing exporter, now using shared attempt/range helpers | Preserve SDK-free regression; native allocation/import/inference tests remain separate. |
| CUDA ordinary device and managed allocations | Exported; shared helper and native host glue tested with fake SDK | Compile CUDA 12.4/12.9/13.3; test applicable paths on P100 and available modern GPU. |
| HIP ordinary device and managed fallback | Exported; actual successful path retained, shared source with CUDA | Compile ROCm 10.1; qualify gfx906 and gfx90c UMA. |
| CUDA/HIP host allocations | Exported; CPU context/fallback preserved; known/unproven pinning distinguished | Native host allocation/free, multi-device and low-memory qualification. |
| CUDA/HIP registered host views and raw CPU aliases | Not exported in this step | Prove registration-to-CPU-buffer bindings and lifecycle before enabling this path. |
| CUDA/HIP split, VMM and pools | Audit required in L1/L2 | Prove shard/range mapping or expose explicit incomplete coverage; reserved virtual space and idle/internal pools are not automatically backend-buffer allocations. |
| SYCL device/host/shared/system USM and async paths | Planned L3 | Capture actual USM/context metadata, host fallback and free/reuse; compile oneAPI and test available Intel GPU. |
| CPU/system allocation exporter | No new exporter in this step | Preserve existing unsupported/unknown behavior. |

## CUDA/HIP exporter

The common CUDA/HIP translation unit exports the existing v1 procedure for
ordinary backend-owned device buffers, successful managed allocations and
host allocation buffers. It records the successful API only. HIP's existing
hipMallocManaged-not-supported fallback records hipMalloc as DEVICE; a failed
attempt/fallback publishes no successful metadata. Advice and error clearing
remain in allocation paths, never inspection. MUSA does not advertise this
exporter.

Device records live in their owning buffer contexts. Their identity domain is
the associated registry device and their identity is the owned allocation
context, not the numeric CUDA/HIP pointer. The actual SDK integrated property
is saved during existing discovery separately from the integrated optimization
flag, which is deliberately disabled elsewhere. This does not enable that
optimization. DEVICE allocations on a proved non-integrated device are
DEVICE_LOCAL in the API sense; integrated or unknown devices and MANAGED
allocations keep UNKNOWN locality. Neither result measures page residency or
proves dedicated physical VRAM. The device association is the device selected
for allocation, not a statement that other devices cannot access the memory.

Host allocation records use a separate registry keyed by the live buffer.
The CPU buffer's raw-pointer context and get/clear/copy operations remain intact.
Only metadata is retained; free erases the record before freeing native memory.
Inspection obtains an immutable retained metadata record and releases its lock
before the callback. Its domain is this backend's host record registry and its
identity is the live record. No GPU0 association is inferred from the host
buffer type's provider device. Failed metadata storage produces unavailable
coverage while preserving the successful native allocation.

CUDA documents that cudaMallocHost may not page-lock on systems where
pageableMemoryAccessUsesHostPageTables is true. Because this host API names no
device, discovery proves pinning only if every visible physical CUDA device
reports that property false; otherwise the allocation is conservatively HOST,
with native_api=cudaMallocHost. HIP's documented host allocation path is
HOST_PINNED. The existing HOST_PINNED buffer flag now follows the saved proof,
including absent metadata, rather than the destructor/name alone. Failed or
disabled pinned attempts return the actual CPU buffer with no failed CUDA/HIP
metadata. No extra discovery query is added: existing physical-device queries
supply the cached evidence, and inspection never reads SDK state.

The reported span is the logical range accepted by the native allocation API.
It is not allocator rounding, driver reserved capacity, RSS or a physical byte
sum. A buffer binding covers its proved part of that span. Known empty probes
without a native allocation emit no allocation. Missing or invalid metadata
returns UNAVAILABLE; unrelated private contexts return UNSUPPORTED.

This step does not export cudaHostRegister/hipHostRegister ranges attached to
CPU_Mapped buffers or arbitrary CPU aliases over a host allocation. Backend
meta/split objects remain unsupported without proved shard bindings; generic
multi-buffers still require leaf traversal. VMM reservations, idle/internal
pools, graph and cuBLAS workspaces are not separately exported as backend-buffer
allocations. They must not be added to this scope by guessing from addresses.
Existing per-buffer device allocations used by model/KV/compute allocators
remain covered when represented by the supported native buffer type.

`test-memory-cuda` tests the production metadata/control-flow helpers, native
API naming, managed fallback, integrated/unknown locality, host record lifetime
and concurrent access for distinct owners. `test-memory-cuda-native.py` extracts
the actual host allocation, discovery-policy, free and inspection functions
from ggml-cuda.cu and compiles them against a fake SDK on Linux. It checks CPU
buffer behavior and fallback, free hooks, pinning proof, managed fallback,
empty probes and missing metadata. Counters establish that inspection performs
no native call, device selection, SDK property query, environment read or log.
Its MUSA variant checks only that this exporter is not advertised. It does not
compile GPU kernels or establish actual CUDA/HIP SDK/runtime compatibility.

Primary API references:

- [CUDA 12.4.1 memory API](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA 13.3.0 device properties](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/structcudaDeviceProp.html)
- [HIP memory management](https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html)
- [HIP integrated property](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/doxygen/html/structhip_device_prop__t.html)

## Vulkan exporter

Vulkan exports this procedure for its ordinary device buffers, pinned host
buffers and successful external-host imports. It saves type, heap, raw type/heap
flags and native allocation size only after `allocateMemory()` succeeds. A
failed candidate contributes no metadata. The existing retry order, requested
sizes, mmap policy and allocation behavior stay unchanged.

The identity domain is the live native Vulkan device object; allocation identity
is the owned native buffer/allocation object. Device buffers use their existing
private context. Pinned host buffers use their CPU pointer to look up the
existing pinned map under its read lock, retaining the native allocation while
copying metadata. No context cast from a CPU pointer, SDK query or device
initialization is performed during inspection. Missing metadata or invalid
ranges return `UNAVAILABLE` without publishing a binding.

Pinned allocation fallback returns an actual CPU buffer, so inspection uses the
CPU provider and publishes no metadata from a failed Vulkan attempt. Zero-size
probes without native memory return `COMPLETE` with no allocations. Ordinary
native spans include Vulkan-required padding, while ggml buffer capacity keeps
its original meaning. Pinned subranges carry their proved allocation offsets.

Imported host memory uses `origin=IMPORTED` and `kind=HOST`. Its metadata is
available only if that native path succeeds; this exporter does not enable host
imports where the device capabilities currently disable them. Separate imports
can alias the same external memory and must not imply separate physical payloads.
Device-local and host-visible flags remain independent for pinned, imported and
device buffers alike. An iGPU does not discount bytes or change assignment.

`test-memory-vulkan` exercises the same allocation-call wrapper and metadata
translation without a Vulkan SDK/device. It injects a failed first candidate,
verifies the successful second type/heap and tests import failure, pinned ranges,
local/nonlocal flags, CPU fallback dispatch and empty results. Native pinned-map
lookup, actual import/allocation and inference still require Vulkan hardware
checks; the CPU fixture does not replace them.

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
