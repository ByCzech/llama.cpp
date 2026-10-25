#include "../ggml/src/ggml-sycl/ggml-sycl-memory.hpp"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct context { int id; };
struct queue {
    int id;
    bool fail = false;
    context get_context() const {
        if (fail) { throw std::runtime_error("diagnostic context unavailable"); }
        return {id};
    }
};
struct observation {
    ggml_backend_memory_allocation a;
    ggml_backend_memory_binding b;
    std::string api;
    std::map<std::string, uint64_t> properties;
};
static void collect(const ggml_backend_memory_allocation * a, const ggml_backend_memory_binding * b, void * opaque) {
    auto & out = *static_cast<std::vector<observation> *>(opaque);
    observation value = {*a, *b, a->native_api, {}};
    for (size_t i = 0; i < a->property_count; ++i) { value.properties.emplace(a->properties[i].name, a->properties[i].value); }
    value.a.native_api = nullptr;
    value.a.properties = nullptr;
    out.push_back(std::move(value));
}

static void test_contexts_and_spans() {
    ggml_backend_device device = {};
    int identity = 0;
    queue first{1}, second{2};
    auto record = ggml_sycl_memory_candidate(GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE, "zeMemAllocDevice", 4096);
    record.known = true;
    record.device = &device;
    record.ordinal_known = true;
    record.ordinal = 0;
    record.device_type_known = true;
    record.discrete = true;
    ggml_sycl_memory_capture_context(&record, first);
    GGML_ASSERT(record.context && static_cast<const context *>(record.context.get())->id == 1);
    std::vector<observation> out;
    GGML_ASSERT(ggml_sycl_memory_visit(record, &identity, 1024, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].a.size_bytes == 4096 && out[0].b.size_bytes == 1024 && out[0].b.known_fields == 7);
    GGML_ASSERT(out[0].properties.at("level_zero.memory_ordinal") == 0);
    GGML_ASSERT(out[0].properties.at("level_zero.device_integrated") == 0);
    GGML_ASSERT(out[0].a.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL);
    auto alias = record;
    GGML_ASSERT(ggml_sycl_memory_visit(alias, &identity, 2048, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].a.identity_domain == out[1].a.identity_domain && out[0].a.identity == out[1].a.identity);
    ggml_sycl_memory_capture_context(&alias, second);
    GGML_ASSERT(ggml_sycl_memory_visit(alias, &identity, 1024, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[2].a.identity_domain != out[0].a.identity_domain && out[2].a.identity == out[0].a.identity);
    out.clear();
    record.discrete = false;
    GGML_ASSERT(ggml_sycl_memory_visit(record, &identity, 1024, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].a.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
    GGML_ASSERT(out[0].properties.at("level_zero.device_integrated") == 1);
    record.device_type_known = false;
    record.ordinal_known = false;
    out.clear();
    GGML_ASSERT(ggml_sycl_memory_visit(record, &identity, 1024, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].properties.empty() && out[0].a.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
    out.clear();
    GGML_ASSERT(ggml_sycl_memory_visit(record, &identity, 4097, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(out.empty());
    first.fail = true;
    ggml_sycl_memory_capture_context(&record, first);
    GGML_ASSERT(record.known && !record.context); // Allocation success survives a diagnostic failure.
    GGML_ASSERT(ggml_sycl_memory_visit(record, &identity, 1024, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(out.empty());
}

static void test_host_and_system() {
    ggml_sycl_memory_host_records records;
    queue q{7};
    int buffer = 0;
    auto host = ggml_sycl_memory_candidate(GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST, "sycl::malloc_host", 8192);
    host.known = true;
    ggml_sycl_memory_capture_context(&host, q);
    records.remember(&buffer, host);
    auto saved = records.find(&buffer);
    GGML_ASSERT(saved && saved->context && !saved->device);
    std::vector<observation> out;
    GGML_ASSERT(ggml_sycl_memory_visit(*saved, saved.get(), 4096, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].a.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST && !out[0].a.device);
    GGML_ASSERT(out[0].a.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN && out[0].properties.empty());
    records.forget(&buffer);
    GGML_ASSERT(!records.find(&buffer));
    records.remember(&buffer, host);
    GGML_ASSERT(records.find(&buffer).get() != saved.get());
    auto system = ggml_sycl_memory_candidate(GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM, "aligned_alloc", 2 * 1024 * 1024);
    system.known = true;
    records.remember(&buffer, system);
    const auto aligned = records.find(&buffer);
    out.clear();
    GGML_ASSERT(ggml_sycl_memory_visit(*aligned, aligned.get(), 64, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].a.identity_domain == ggml_sycl_memory_system_domain());
    GGML_ASSERT(out[0].a.size_bytes == 2 * 1024 * 1024 && out[0].b.size_bytes == 64);
    GGML_ASSERT(out[0].a.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM && !out[0].a.device);
    system.known = false;
    records.remember(&buffer, system);
    GGML_ASSERT(!records.find(&buffer));
}

int main() {
    test_contexts_and_spans();
    test_host_and_system();
    std::puts("SYCL context identities, native spans, host/system metadata and lifetime passed (SDK-free)");
}
