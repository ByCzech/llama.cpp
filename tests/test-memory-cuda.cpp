#include "../ggml/src/ggml-cuda/ggml-cuda-memory.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

struct observation {
    ggml_backend_memory_allocation value;
    ggml_backend_memory_binding binding;
    std::string api;
    std::map<std::string, uint64_t> properties;
};
static void collect(const ggml_backend_memory_allocation * a, const ggml_backend_memory_binding * b, void * data) {
    auto & out = *static_cast<std::vector<observation> *>(data);
    observation value = {*a, *b, a->native_api, {}};
    for (size_t i = 0; i < a->property_count; ++i) { value.properties.emplace(a->properties[i].name, a->properties[i].value); }
    value.value.properties = nullptr;
    value.value.native_api = nullptr;
    out.push_back(std::move(value));
}

static void test_paths() {
    for (auto api : {ggml_cuda_memory_api::cuda, ggml_cuda_memory_api::hip}) {
        ggml_cuda_memory_record selected;
        size_t device_calls = 0, managed_calls = 0, advice_calls = 0, fallback_checks = 0;
        auto run = [&](bool unified, int managed_status, int device_status) {
            return ggml_cuda_memory_allocate(selected, api, 4096, unified,
                [&] { ++device_calls; return device_status; },
                [&] { ++managed_calls; return managed_status; }, [](int result) { return result == 0; },
                [&] { ++advice_calls; },
                [&](int status) { ++fallback_checks; return api == ggml_cuda_memory_api::hip && status == 2; });
        };
        GGML_ASSERT(run(false, 0, 0) == 0 && selected.known);
        GGML_ASSERT(selected.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE && managed_calls == 0);
        GGML_ASSERT(run(true, 0, 0) == 0 && selected.known);
        GGML_ASSERT(selected.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED && advice_calls == 1);
        GGML_ASSERT(run(true, 1, 0) == 1 && !selected.known && selected.size == 0);
        const auto old_device_calls = device_calls;
        const auto status = run(true, 2, 0);
        if (api == ggml_cuda_memory_api::hip) {
            GGML_ASSERT(status == 0 && selected.known && selected.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE);
            GGML_ASSERT(device_calls == old_device_calls + 1 && std::strcmp(ggml_cuda_memory_native_api(selected), "hipMalloc") == 0);
        } else {
            GGML_ASSERT(status == 2 && !selected.known && device_calls == old_device_calls);
        }
        GGML_ASSERT(run(true, 2, 3) != 0 && !selected.known && selected.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_UNKNOWN);
        GGML_ASSERT(advice_calls == 1 && fallback_checks == 3 && managed_calls == 4);
    }
}

static void test_metadata() {
    ggml_backend_device device = {};
    int identity = 0;
    for (auto api : {ggml_cuda_memory_api::cuda, ggml_cuda_memory_api::hip}) {
        for (auto kind : {GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE, GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED}) {
            auto record = ggml_cuda_memory_candidate(api, kind, 8192);
            record.known = true;
            record.device = &device;
            record.integrated_known = true;
            std::vector<observation> out;
            GGML_ASSERT(ggml_cuda_memory_visit(record, &device, &identity, 4096, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
            GGML_ASSERT(out.size() == 1 && out[0].value.size_bytes == 8192 && out[0].binding.size_bytes == 4096);
            GGML_ASSERT(out[0].value.device == &device && out[0].value.kind == kind);
            GGML_ASSERT(out[0].properties.at(api == ggml_cuda_memory_api::hip ? "hip.device_integrated" : "cuda.device_integrated") == 0);
            GGML_ASSERT(out[0].value.locality == (kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE ?
                GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL : GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN));
            record.integrated = true;
            out.clear();
            GGML_ASSERT(ggml_cuda_memory_visit(record, &device, &identity, 4096, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
            GGML_ASSERT(out[0].value.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
            GGML_ASSERT(out[0].properties.begin()->second == 1);
            out.clear();
            GGML_ASSERT(ggml_cuda_memory_visit(record, &device, &identity, 8193, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
            GGML_ASSERT(out.empty());
            record.integrated_known = false;
            out.clear();
            GGML_ASSERT(ggml_cuda_memory_visit(record, &device, &identity, 4096, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
            GGML_ASSERT(out[0].properties.empty() && out[0].value.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
        }
    }
}

static void test_host_lifetime() {
    ggml_cuda_memory_host_records first, second;
    int buffer = 0;
    auto record = ggml_cuda_memory_candidate(ggml_cuda_memory_api::hip, GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED, 4096);
    record.known = true;
    GGML_ASSERT(!first.find(&buffer));
    first.remember(&buffer, record);
    const auto old = first.find(&buffer);
    GGML_ASSERT(old && !old->device && !old->integrated_known);
    std::vector<observation> out;
    GGML_ASSERT(ggml_cuda_memory_visit(*old, &first, old.get(), 4096, collect, &out) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(out[0].value.kind == GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED);
    GGML_ASSERT(out[0].api == "hipHostMalloc" && !out[0].value.device && out[0].properties.empty());
    first.forget(&buffer);
    GGML_ASSERT(!first.find(&buffer));
    first.remember(&buffer, record);
    const auto replacement = first.find(&buffer);
    GGML_ASSERT(replacement && replacement.get() != old.get()); // Retained old identity cannot be recycled.
    second.remember(&buffer, record);
    GGML_ASSERT(second.find(&buffer) && second.find(&buffer).get() != replacement.get());
    record.known = false;
    first.remember(&buffer, record);
    GGML_ASSERT(!first.find(&buffer)); // Failed/unknown metadata cannot leave a previous record behind.
    second.forget(&buffer);
    GGML_ASSERT(!second.find(&buffer));

    // Different owners can update the registry concurrently with a stable owner's read.
    record.known = true;
    first.remember(&buffer, record);
    int other_buffer = 0;
    std::thread writer([&] {
        for (int i = 0; i < 2000; ++i) {
            first.remember(&other_buffer, record);
            first.forget(&other_buffer);
        }
    });
    for (int i = 0; i < 2000; ++i) {
        const auto stable = first.find(&buffer);
        GGML_ASSERT(stable && stable->size == 4096 && !stable->device);
    }
    writer.join();
    first.forget(&buffer);
    GGML_ASSERT(!first.find(&buffer) && !first.find(&other_buffer));
}

int main() {
    test_paths();
    test_metadata();
    test_host_lifetime();
    std::puts("CUDA/HIP allocation paths, API metadata, UMA and host lifetime passed (SDK-free)");
}
