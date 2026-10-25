#!/usr/bin/env python3
"""Compile actual SYCL allocation/inspection host functions with a fake SDK.

The fixture scales alignment/USM-system thresholds for small CPU allocations.
It compiles no kernels and does not qualify oneAPI or a real Level Zero driver.
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
#include "ggml-sycl/ggml-sycl-memory.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define GGML_SYCL_NAME "fixture"
#define GGML_SYCL_DEBUG(...) ((void)0)
#define SYCL_BUFFER_ALIGNMENT 128
#define MEM_SIZE_2M 128
#define MEM_SIZE_1G 32
#define GGML_SYCL_MEM_DIRECT 0
#define GGML_SYCL_MEM_BUFFER 1
#define CHECK_TRY_ERROR(...) (__VA_ARGS__, 0)
#define SYCL_CHECK(call) GGML_ASSERT((call) == 0)
using ggml_sycl_mem_type = int;
struct counters {
    int allocations = 0, frees = 0, context = 0, device = 0, backend = 0, native = 0;
    int check = 0, select = 0, info = 0, trace = 0, wait = 0, logs = 0;
    bool operator==(const counters & b) const {
        return allocations == b.allocations && frees == b.frees && context == b.context &&
            device == b.device && backend == b.backend && native == b.native && check == b.check &&
            select == b.select && info == b.info && trace == b.trace && wait == b.wait && logs == b.logs;
    }
} calls;
struct native_allocation { int context, kind; };
static std::map<void *, native_allocation> native_allocations;
static bool fail_device = false, fail_host = false, fail_system = false;
static bool fail_context = false, throw_host = false;
static void * allocate_native(size_t size, int context, int kind, bool fail) {
    ++calls.allocations;
    if (fail) { return nullptr; }
    void * ptr = std::aligned_alloc(128, (std::max(size, size_t(1)) + 127) / 128 * 128);
    GGML_ASSERT(ptr && native_allocations.emplace(ptr, native_allocation{context, kind}).second);
    return ptr;
}
static void free_native(void * ptr, int context, int kind) {
    if (!ptr) { return; }
    ++calls.frees;
    const auto found = native_allocations.find(ptr);
    GGML_ASSERT(found != native_allocations.end() && found->second.context == context && found->second.kind == kind);
    native_allocations.erase(found);
    std::free(ptr);
}
namespace sycl {
struct exception : std::runtime_error { using std::runtime_error::runtime_error; };
enum class backend { ext_oneapi_level_zero, other };
struct context { int id = 1; };
struct device { int id = 1; bool is_gpu() const { ++calls.device; return true; } };
struct property_list {};
struct event { void wait_and_throw() { ++calls.wait; } };
struct queue {
    int context_id = 1;
    backend kind = backend::ext_oneapi_level_zero;
    context get_context() const { ++calls.context; if (fail_context) { throw exception("context"); } return {context_id}; }
    device get_device() const { ++calls.device; return {1}; }
    backend get_backend() const { ++calls.backend; return kind; }
};
template<backend B> int get_native(const context & c) { ++calls.native; return c.id; }
template<backend B> int get_native(const device & d) { ++calls.native; return d.id; }
static void * malloc_device(size_t size, queue & q) { return allocate_native(size, q.context_id, 1, fail_device); }
static void * malloc_host(size_t size, queue & q, property_list = {}) {
    if (throw_host) { throw exception("host allocation"); }
    return allocate_native(size, q.context_id, 2, fail_host);
}
static void free(void * ptr, const context & c) {
    const auto found = native_allocations.find(ptr);
    GGML_ASSERT(found != native_allocations.end() && found->second.kind != 3);
    free_native(ptr, c.id, found->second.kind);
}
static void free(void * ptr, queue & q) { free(ptr, context{q.context_id}); }
}
using queue_ptr = sycl::queue *;
namespace dpct {
struct fixture_device {
    sycl::queue q;
    sycl::queue & default_queue() { ++calls.device; return q; }
};
struct dev_mgr {
    fixture_device devices[2];
    static dev_mgr & instance() { static dev_mgr manager; return manager; }
    fixture_device & get_device(int id) { ++calls.device; return devices[id]; }
};
}
static int g_ggml_sycl_use_level_zero_api = 1;
static bool g_ggml_sycl_usm_system = false, g_ggml_sycl_enable_host_pinned_mem = true;
struct optimize_feature {};
struct fixture_info {
    struct { bool usm_system_support = true, l0_device_type_valid = true, l0_discrete_gpu = true; optimize_feature opt_feature; } devices[2];
} info;
static const fixture_info & ggml_sycl_info() { ++calls.info; return info; }
static void ggml_check_sycl() { ++calls.check; }
static void ggml_sycl_set_device(int) { ++calls.select; }
static void check_allow_gpu_index(int i) { GGML_ASSERT(i >= 0 && i < 2); }
struct ggml_tensor_extra_gpu {};
static void release_extra_gpu(ggml_tensor_extra_gpu * p) { delete p; }
static void ggml_sycl_memtrace_add(int, void *, size_t) { ++calls.trace; }
static void ggml_sycl_memtrace_fail(int, size_t) { ++calls.trace; }
static void ggml_sycl_memtrace_del(void *) { ++calls.trace; }
using ze_result_t = int;
constexpr int ZE_RESULT_SUCCESS = 0, ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 1;
constexpr int ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 2, ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE = 1;
struct ze_device_mem_alloc_desc_t { int stype; const void * pNext; unsigned flags, ordinal; };
struct ze_relaxed_allocation_limits_exp_desc_t { int stype; const void * pNext; unsigned flags; };
static int zeMemAllocDevice(int context, const ze_device_mem_alloc_desc_t *, size_t size, size_t, int, void ** ptr) {
    *ptr = allocate_native(size, context, 3, fail_device); return *ptr ? 0 : 1;
}
static int zeMemFree(int context, void * ptr) { free_native(ptr, context, 3); return 0; }
static void free_aligned_mem_host(void * ptr) { free_native(ptr, 0, 4); }
static void * fixture_aligned_alloc(size_t, size_t size) { return allocate_native(size, 0, 4, fail_system); }
#define aligned_alloc fixture_aligned_alloc
'''
MAIN = r'''
struct observation {
    ggml_backend_memory_allocation a;
    ggml_backend_memory_binding b;
    std::string api;
    bool ordinal;
};
static void collect(const ggml_backend_memory_allocation * a, const ggml_backend_memory_binding * b, void * data) {
    auto & out = *static_cast<std::vector<observation> *>(data);
    observation value = {*a, *b, a->native_api, false};
    for (size_t i = 0; i < a->property_count; ++i) {
        if (std::strcmp(a->properties[i].name, "level_zero.memory_ordinal") == 0) {
            GGML_ASSERT(a->properties[i].value == 0); value.ordinal = true;
        }
    }
    value.a.native_api = nullptr; value.a.properties = nullptr;
    out.push_back(std::move(value));
}
int main() {
    ggml_log_set([](ggml_log_level, const char *, void *) { ++calls.logs; }, nullptr);
    ggml_backend_reg reg = {};
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface.get_proc_address = ggml_backend_sycl_reg_get_proc_address;
    ggml_backend_sycl_device_context dev_ctx = {0, "fixture", "fixture", 0};
    ggml_backend_device device = {};
    device.reg = &reg; device.context = &dev_ctx;
    auto * q = &dpct::dev_mgr::instance().devices[0].q;
    ggml_backend_sycl_buffer_type_context type_ctx = {0, "fixture", q};
    ggml_backend_buffer_type buft = {};
    buft.context = &type_ctx; buft.device = &device;
    buft.iface.alloc_buffer = ggml_backend_sycl_buffer_type_alloc_buffer;
    buft.iface.get_name = ggml_backend_sycl_buffer_type_get_name;
    buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 128; };
    auto inspect = reinterpret_cast<ggml_backend_memory_buffer_allocations_t>(
        ggml_backend_sycl_reg_get_proc_address(&reg, "tllama_buffer_memory_allocations_v1"));
    GGML_ASSERT(inspect);
    for (int native : {0, 1}) {
        g_ggml_sycl_use_level_zero_api = native;
        auto * buffer = ggml_backend_buft_alloc_buffer(&buft, 512);
        GGML_ASSERT(buffer);
        const auto before = calls;
        std::vector<observation> out;
        for (int i = 0; i < 20; ++i) {
            GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
        }
        GGML_ASSERT(calls == before && out.size() == 20 && out[0].a.device == &device);
        GGML_ASSERT(out[0].a.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE && out[0].a.size_bytes == 512);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_ASSERT(out[0].ordinal == bool(native));
        GGML_ASSERT(out[0].api == (native ? "zeMemAllocDevice" : "sycl::malloc_device"));
#else
        GGML_ASSERT(!out[0].ordinal && out[0].api == "sycl::malloc_device");
#endif
        // Free must follow the saved successful API/context, even if the selector changed.
        g_ggml_sycl_use_level_zero_api = !native;
        q->context_id = 2;
        ggml_backend_buffer_free(buffer);
        q->context_id = 1;
    }
    fail_device = true;
    GGML_ASSERT(!ggml_backend_buft_alloc_buffer(&buft, 512));
    fail_device = false;
    g_ggml_sycl_use_level_zero_api = 0;
    fail_context = true;
    auto * buffer = ggml_backend_buft_alloc_buffer(&buft, 512);
    GGML_ASSERT(buffer); // Diagnostic context failure cannot turn native allocation success into failure.
    std::vector<observation> out;
    GGML_ASSERT(inspect(buffer, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE && out.empty());
    fail_context = false;
    ggml_backend_buffer_free(buffer);
    g_ggml_sycl_usm_system = true;
    buffer = ggml_backend_buft_alloc_buffer(&buft, 64);
    const auto before_system = calls;
    GGML_ASSERT(inspect(buffer, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(calls == before_system && out[0].a.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM);
    GGML_ASSERT(out[0].a.size_bytes == MEM_SIZE_2M && out[0].b.size_bytes == 64 && !out[0].a.device);
    ggml_backend_buffer_free(buffer);
    fail_system = true;
    GGML_ASSERT(!ggml_backend_buft_alloc_buffer(&buft, 64));
    fail_system = false; g_ggml_sycl_usm_system = false;
    buffer = ggml_backend_buft_alloc_buffer(&buft, 0);
    out.clear();
    const auto before_empty = calls;
    GGML_ASSERT(inspect(buffer, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE && out.empty());
    GGML_ASSERT(calls == before_empty);
    ggml_backend_buffer_free(buffer);

    ggml_backend_buffer_type host = {};
    host.device = &device; host.iface.get_name = ggml_backend_sycl_host_buffer_type_name;
    host.iface.alloc_buffer = ggml_backend_sycl_host_buffer_type_alloc_buffer;
    host.iface.get_alignment = buft.iface.get_alignment;
    for (bool usm : {false, true}) {
        g_ggml_sycl_enable_host_pinned_mem = usm;
        buffer = ggml_backend_buft_alloc_buffer(&host, 512);
        GGML_ASSERT(buffer && buffer->context == ggml_backend_buffer_get_base(buffer));
        ggml_backend_buffer_clear(buffer, 0x5a);
        GGML_ASSERT(static_cast<unsigned char *>(buffer->context)[511] == 0x5a);
        out.clear();
        const auto before = calls;
        GGML_ASSERT(inspect(buffer, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
        GGML_ASSERT(calls == before && !out[0].a.device);
        GGML_ASSERT(out[0].a.kind == (usm ? GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST : GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM));
        GGML_ASSERT(!(ggml_backend_sycl_memory_buffer_flags(buffer) & GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED));
        const void * key = buffer;
        g_ggml_sycl_enable_host_pinned_mem = !usm;
        q->context_id = 2;
        ggml_backend_buffer_free(buffer);
        GGML_ASSERT(!ggml_sycl_host_memory_records().find(key));
        q->context_id = 1;
    }
    g_ggml_sycl_enable_host_pinned_mem = true;
    for (bool throwing : {false, true}) {
        throw_host = throwing; fail_host = !throwing;
        buffer = ggml_backend_buft_alloc_buffer(&host, 512);
        GGML_ASSERT(buffer && buffer->buft != &host);
        out.clear();
        GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
        ggml_backend_buffer_free(buffer);
    }
    throw_host = false; fail_host = false;
    GGML_ASSERT(native_allocations.empty());
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--ggml", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
    common = (ROOT / "ggml/src/ggml-sycl/common.cpp").read_text()
    def part(text, begin, end):
        start = text.index(begin)
        return text[start:text.index(end, start)]
    use_l0 = part(common, "#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API\nstatic bool ggml_sycl_use_level_zero_device_alloc", "// Use Level Zero zeMemAllocDevice")
    malloc_free = part(common, "void * ggml_sycl_malloc_device", "void release_extra_gpu")
    device = part(source, "struct ggml_backend_sycl_buffer_context {", "static enum ggml_status\nggml_backend_sycl_buffer_init_tensor")
    iface = r'''
static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = [] {
    ggml_backend_buffer_i result = {};
    result.free_buffer = ggml_backend_sycl_buffer_free_buffer;
    result.get_base = ggml_backend_sycl_buffer_get_base;
    return result;
}();
'''
    factory = part(source, "struct ggml_backend_sycl_buffer_type_context {", "static size_t ggml_backend_sycl_buffer_type_get_alignment")
    host = part(source, "struct ggml_backend_sycl_device_context {", "static size_t ggml_backend_sycl_host_buffer_type_get_max_size")
    proc = part(source, "static uint32_t ggml_backend_sycl_memory_buffer_flags", "static const ggml_backend_reg_i ggml_backend_sycl_reg_interface")
    stubs = 'static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t) { return "split"; }\n'
    stubs += "\n".join("static void " + name + "() {}" for name in (
        "ggml_backend_sycl_split_buffer_type", "ggml_backend_sycl_comm_init", "ggml_backend_sycl_comm_free",
        "ggml_backend_sycl_comm_allreduce_tensor"))
    with tempfile.TemporaryDirectory(prefix="sycl-native-metadata-") as tmp:
        root = Path(tmp)
        cpp = root / "native.cpp"
        cpp.write_text(PREFIX + use_l0 + malloc_free + device + iface + factory + host + stubs + proc + MAIN)
        for name, flags in (("sycl", []), ("level-zero", ["-DGGML_SYCL_SUPPORT_LEVEL_ZERO_API"]),
                            ("level-zero-relaxed", ["-DGGML_SYCL_SUPPORT_LEVEL_ZERO_API", "-DZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME=1"])):
            binary = root / name
            sanitizer = ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else []
            subprocess.run([args.compiler, "-std=c++17", "-O1", "-pthread", *sanitizer, *flags,
                            "-I" + str(ROOT / "ggml/include"), "-I" + str(ROOT / "ggml/src"), str(cpp),
                            str(Path(args.ggml).resolve()), str(Path(args.base).resolve()),
                            "-Wl,-rpath," + str(Path(args.base).resolve().parent), "-o", str(binary)], check=True, timeout=60)
            environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=0") if args.sanitize else None
            subprocess.run([str(binary)], check=True, timeout=15, env=environment)
            print(name + ": production SYCL host glue passed with fake SDK", flush=True)


if __name__ == "__main__":
    main()
