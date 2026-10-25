#!/usr/bin/env python3
"""Compile actual CUDA/HIP host allocation/inspection functions with a fake SDK.

This verifies native glue and lifecycle without compiling GPU kernels or claiming
SDK/hardware qualification. Only the enumerated source regions are compiled.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

PREFIX = r'''
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda/ggml-cuda-memory.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define GGML_CUDA_NAME "fixture"
#define CUDA_CHECK(call) GGML_ASSERT((call) == 0)
using cudaError_t = int;
constexpr int cudaSuccess = 0, hipErrorNotSupported = 2, hipMemAdviseSetCoarseGrain = 1;
struct counters {
    int device = 0, managed = 0, host = 0, free_device = 0, free_host = 0;
    int select = 0, errors = 0, advice = 0, env = 0, info = 0, logs = 0, properties = 0;
    bool operator==(const counters & b) const {
        return device == b.device && managed == b.managed && host == b.host &&
            free_device == b.free_device && free_host == b.free_host && select == b.select &&
            errors == b.errors && advice == b.advice && env == b.env && info == b.info && logs == b.logs && properties == b.properties;
    }
} calls;
static bool unified = false, no_pinned = false;
static int managed_status = 0, host_status = 0, device_status = 0;
static int allocate(void ** ptr, size_t size, int status) {
    *ptr = nullptr;
    if (status) { return status; }
    if (size) { *ptr = std::aligned_alloc(128, (size + 127) / 128 * 128); GGML_ASSERT(*ptr); }
    return 0;
}
static int cudaMalloc(void ** p, size_t n) { ++calls.device; return allocate(p, n, device_status); }
static int cudaMallocManaged(void ** p, size_t n) { ++calls.managed; return allocate(p, n, managed_status); }
static int cudaMallocHost(void ** p, size_t n) { ++calls.host; return allocate(p, n, host_status); }
static int cudaFree(void * p) { ++calls.free_device; std::free(p); return 0; }
static int cudaFreeHost(void * p) { ++calls.free_host; std::free(p); return 0; }
static int cudaGetLastError() { ++calls.errors; return 0; }
static int hipGetLastError() { ++calls.errors; return 0; }
static int cudaMemAdvise(void *, size_t, int, int) { ++calls.advice; return 1; } // Hint failure is ignored.
static const char * cudaGetErrorString(int) { return "injected allocation failure"; }
static void ggml_cuda_set_device(int) { ++calls.select; }
static bool ggml_cuda_unified_memory_enabled() { ++calls.env; return unified; }
static bool ggml_cuda_pinned_memory_disabled() { ++calls.env; return no_pinned; }
struct fixture_info { int physical_device_count = 2; bool host_api_pinning_proven = true; struct { bool allocation_integrated = false; } devices[2]; } info;
static const fixture_info & ggml_cuda_info() { ++calls.info; return info; }
struct cudaDeviceProp { size_t totalGlobalMem = 1024; int pageableMemoryAccessUsesHostPageTables = 0; } properties[2];
static int cudaGetDeviceProperties(cudaDeviceProp * prop, int id) { ++calls.properties; *prop = properties[id]; return 0; }

'''

MAIN = r'''
static void collect(const ggml_backend_memory_allocation * a, const ggml_backend_memory_binding * b, void * data) {
    auto & out = *static_cast<std::vector<ggml_backend_memory_allocation> *>(data);
    GGML_ASSERT(b->known_fields == 7 && b->buffer_offset == 0 && b->allocation_offset == 0);
    GGML_ASSERT(b->size_bytes <= a->size_bytes && a->size_known);
    out.push_back(*a); // Only scalar/identity fields are used after the callback.
    out.back().properties = nullptr;
    out.back().native_api = nullptr;
}
int main() {
    ggml_log_set([](ggml_log_level, const char *, void *) { ++calls.logs; }, nullptr);
    GGML_ASSERT(!probe_pinning_policy(0));
    GGML_ASSERT(probe_pinning_policy(2));
    properties[1].pageableMemoryAccessUsesHostPageTables = 1;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_ASSERT(probe_pinning_policy(2));
#else
    GGML_ASSERT(!probe_pinning_policy(2)); // Any visible device prevents a universal CUDA pinning proof.
#endif
    properties[1].pageableMemoryAccessUsesHostPageTables = 0;
    ggml_backend_reg reg = {};
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface.get_proc_address = ggml_backend_cuda_reg_get_proc_address;
    ggml_backend_device device = {};
    device.reg = &reg;
    ggml_backend_cuda_buffer_type_context type_ctx = {0, "fixture"};
    ggml_backend_buffer_type buft = {};
    buft.context = &type_ctx;
    buft.device = &device;
    buft.iface.alloc_buffer = ggml_backend_cuda_buffer_type_alloc_buffer;
    buft.iface.get_name = ggml_backend_cuda_buffer_type_get_name;
    buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 128; };
    auto inspect = reinterpret_cast<ggml_backend_memory_buffer_allocations_t>(
        ggml_backend_cuda_reg_get_proc_address(&reg, "tllama_buffer_memory_allocations_v1"));
#if defined(GGML_USE_MUSA)
    GGML_ASSERT(!inspect); // This patch must not advertise an unqualified MUSA exporter.
#else
    GGML_ASSERT(inspect);
    for (bool managed : {false, true}) {
        unified = managed;
        auto * buffer = ggml_backend_buft_alloc_buffer(&buft, 512);
        GGML_ASSERT(buffer);
        const auto before = calls;
        std::vector<ggml_backend_memory_allocation> records;
        for (int i = 0; i < 20; ++i) {
            GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer, collect, &records) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
        }
        GGML_ASSERT(calls == before && records.size() == 20);
        GGML_ASSERT(records[0].device == &device && records[0].size_bytes == 512);
        GGML_ASSERT(records[0].identity == records[19].identity);
        GGML_ASSERT(records[0].kind == (managed ? GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED : GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE));
        ggml_backend_buffer_free(buffer);
    }
    unified = true;
    managed_status = 2;
    auto * buffer = ggml_backend_buft_alloc_buffer(&buft, 512);
#if defined(GGML_USE_HIP)
    GGML_ASSERT(buffer);
    std::vector<ggml_backend_memory_allocation> fallback;
    GGML_ASSERT(inspect(buffer, collect, &fallback) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(fallback[0].kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE);
    ggml_backend_buffer_free(buffer);
#else
    GGML_ASSERT(!buffer); // CUDA keeps the existing no-fallback behavior.
#endif
    device_status = 3;
    GGML_ASSERT(!ggml_backend_buft_alloc_buffer(&buft, 512));
    managed_status = 0;
    device_status = 0;
    unified = false;
    info.devices[0].allocation_integrated = true;
    buffer = ggml_backend_buft_alloc_buffer(&buft, 512);
    std::vector<ggml_backend_memory_allocation> apu;
    GGML_ASSERT(inspect(buffer, collect, &apu) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(apu[0].locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
    ggml_backend_buffer_free(buffer);
    buffer = ggml_backend_buft_alloc_buffer(&buft, 0);
    apu.clear();
    const auto before_zero = calls;
    GGML_ASSERT(inspect(buffer, collect, &apu) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE && apu.empty());
    GGML_ASSERT(calls == before_zero);
    ggml_backend_buffer_free(buffer);

    ggml_backend_buffer_type host = {};
    host.device = &device; // Provider device 0 is not a proved host allocation association.
    host.iface.alloc_buffer = ggml_backend_cuda_host_buffer_type_alloc_buffer;
    host.iface.get_name = ggml_backend_cuda_host_buffer_type_name;
    host.iface.get_alignment = buft.iface.get_alignment;
    buffer = ggml_backend_buft_alloc_buffer(&host, 512);
    GGML_ASSERT(buffer && buffer->context == ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_cuda_memory_buffer_flags(buffer) & GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED);
    auto * key = buffer;
    auto * pointer = buffer->context;
    ggml_backend_buffer_clear(buffer, 0x5a);
    GGML_ASSERT(static_cast<unsigned char *>(pointer)[511] == 0x5a);
    const auto before_host = calls;
    std::vector<ggml_backend_memory_allocation> pinned;
    GGML_ASSERT(inspect(buffer, collect, &pinned) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(calls == before_host && !pinned[0].device);
    GGML_ASSERT(pinned[0].kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED);
    ggml_cuda_host_memory_records().forget(buffer);
    GGML_ASSERT(inspect(buffer, collect, &pinned) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(ggml_backend_cuda_memory_buffer_flags(buffer) == 0);
    ggml_backend_buffer_free(buffer);
    GGML_ASSERT(!ggml_cuda_host_memory_records().find(key));
    buffer = ggml_backend_buft_alloc_buffer(&host, 512);
    key = buffer;
    GGML_ASSERT(ggml_cuda_host_memory_records().find(key));
    ggml_backend_buffer_free(buffer); // The production free hook must erase it without test assistance.
    GGML_ASSERT(!ggml_cuda_host_memory_records().find(key));
    info.host_api_pinning_proven = false;
    buffer = ggml_backend_buft_alloc_buffer(&host, 512);
    pinned.clear();
    const auto before_unproven = calls;
    GGML_ASSERT(inspect(buffer, collect, &pinned) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(calls == before_unproven && pinned[0].kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST);
    GGML_ASSERT(ggml_backend_cuda_memory_buffer_flags(buffer) == 0);
    ggml_backend_buffer_free(buffer);
    info.host_api_pinning_proven = true;
    host_status = 1;
    buffer = ggml_backend_buft_alloc_buffer(&host, 512);
    GGML_ASSERT(buffer && buffer->buft != &host);
    pinned.clear();
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer, collect, &pinned) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(pinned.empty());
    ggml_backend_buffer_free(buffer);
    host_status = 0;
    no_pinned = true;
    const auto before_disabled = calls.host;
    buffer = ggml_backend_buft_alloc_buffer(&host, 512);
    GGML_ASSERT(buffer && buffer->buft != &host && calls.host == before_disabled);
    ggml_backend_buffer_free(buffer);
#endif
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--ggml", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
    def part(begin, end):
        start = source.index(begin)
        return source[start:source.index(end, start)]
    discovery = part("    int64_t total_vram = 0;", '    GGML_LOG_INFO("%s: found %d ')
    discovery = "static bool probe_pinning_policy(int count) { fixture_info info; info.physical_device_count = count;\n" + discovery + "return info.host_api_pinning_proven; }\n"
    allocation = part("#if defined(GGML_USE_HIP)\nstatic constexpr auto ggml_cuda_native_memory_api", "\n#if defined(GGML_USE_HIP)\nstatic int ggml_cuda_parse_id")
    device = part("struct ggml_backend_cuda_buffer_context {", "static enum ggml_status ggml_backend_cuda_buffer_init_tensor")
    iface = r'''
static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = [] {
    ggml_backend_buffer_i value = {};
    value.free_buffer = ggml_backend_cuda_buffer_free_buffer;
    value.get_base = ggml_backend_cuda_buffer_get_base;
    return value;
}();
'''
    buffer = part("struct ggml_backend_cuda_buffer_type_context {", "static size_t ggml_backend_cuda_buffer_type_get_alignment")
    host = part("static const char * ggml_backend_cuda_host_buffer_type_name", "static size_t ggml_backend_cuda_host_buffer_type_get_max_size")
    inspection = part("static uint32_t ggml_backend_cuda_memory_buffer_flags", "static const ggml_backend_reg_i ggml_backend_cuda_reg_interface")
    # Existing unrelated procedure addresses are never invoked by this host-only fixture.
    stubs = "\n".join("static void " + name + "() {}" for name in (
        "ggml_backend_cuda_comm_init", "ggml_backend_cuda_comm_free", "ggml_backend_cuda_comm_allreduce_tensor",
        "ggml_backend_cuda_register_host_buffer", "ggml_backend_cuda_unregister_host_buffer", "ggml_backend_cuda_get_features"))
    with tempfile.TemporaryDirectory(prefix="cuda-native-metadata-") as tmp:
        root = Path(tmp)
        cpp = root / "native.cpp"
        cpp.write_text(PREFIX + discovery + allocation + device + iface + buffer + host + stubs + inspection + MAIN)
        for name, defines in (("cuda", []), ("hip", ["-DGGML_USE_HIP"]), ("musa-guard", ["-DGGML_USE_MUSA"])):
            binary = root / name
            sanitizer = ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else []
            command = [args.compiler, "-std=c++17", "-O1", "-pthread", *sanitizer, *defines,
                       "-I" + str(ROOT / "ggml/include"), "-I" + str(ROOT / "ggml/src"), str(cpp),
                       str(Path(args.ggml).resolve()), str(Path(args.base).resolve()), "-Wl,-rpath," + str(Path(args.base).resolve().parent), "-o", str(binary)]
            subprocess.run(command, check=True, timeout=60)
            environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=0") if args.sanitize else None
            subprocess.run([str(binary)], check=True, timeout=15, env=environment)
            print(name + ": production host allocation/inspection glue passed with fake SDK", flush=True)


if __name__ == "__main__":
    main()
