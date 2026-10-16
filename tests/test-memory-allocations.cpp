#include "ggml-backend-memory.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Metadata-only providers: no SDK calls or GPU allocation/inference.
struct allocation_provider {
    ggml_backend_reg reg = {};
    ggml_backend_device device = {};
    ggml_backend_buffer_type type = {};
    ggml_backend_memory_allocation allocation = {};
    ggml_backend_memory_binding binding = {};
    std::vector<ggml_backend_memory_allocation_property> properties;
    ggml_backend_memory_allocation_status status = GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE;
    bool supported = true, old_version_only = false, emit = true, opaque = false;
    size_t calls = 0;
    int domain = 0, identity = 0;

    allocation_provider() {
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.context = this;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "same-backend-name"; };
        reg.iface.get_proc_address = [](ggml_backend_reg_t reg, const char * name) -> void * {
            auto & self = *static_cast<allocation_provider *>(reg->context);
            if (std::strcmp(name, "tllama_buffer_memory_flags") == 0) {
                static ggml_backend_memory_buffer_flags_t flags = [](ggml_backend_buffer_t buffer) -> uint32_t {
                    return static_cast<allocation_provider *>(buffer->context)->opaque ? GGML_BACKEND_MEMORY_BUFFER_OPAQUE : 0;
                };
                return reinterpret_cast<void *>(flags);
            }
            const char * procedure = self.old_version_only ? "tllama_buffer_memory_allocations" : "tllama_buffer_memory_allocations_v1";
            if (!self.supported || std::strcmp(name, procedure) != 0) { return nullptr; }
            static ggml_backend_memory_buffer_allocations_t inspect = [](
                    ggml_backend_buffer_t buffer, ggml_backend_memory_allocation_callback callback, void * opaque) {
                auto & self = *static_cast<allocation_provider *>(buffer->context);
                ++self.calls;
                if (self.emit) {
                    auto allocation = self.allocation;
                    allocation.properties = self.properties.data();
                    allocation.property_count = self.properties.size();
                    callback(&allocation, &self.binding, opaque);
                }
                return self.status;
            };
            return reinterpret_cast<void *>(inspect);
        };
        device.reg = &reg;
        device.iface.get_name = [](ggml_backend_dev_t) { return "same-device-name"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        type.device = &device;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "same-buffer-name"; };
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        allocation.identity_domain = &domain;
        allocation.identity = &identity;
        allocation.device = &device;
        allocation.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE;
        allocation.origin = GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED;
        allocation.size_known = true;
        allocation.size_bytes = 8192;
        binding = {GGML_BACKEND_MEMORY_BINDING_BUFFER_OFFSET | GGML_BACKEND_MEMORY_BINDING_ALLOCATION_OFFSET |
                   GGML_BACKEND_MEMORY_BINDING_SIZE, 0, 1024, 4096};
    }

    ggml_backend_buffer_t buffer(size_t size = 4096) {
        return ggml_backend_buffer_init(&type, {}, this, size);
    }
};

struct observation {
    ggml_backend_memory_allocation allocation;
    ggml_backend_memory_binding binding;
    std::string native_api;
    std::map<std::string, uint64_t> properties;
};
using observations = std::vector<observation>;

static void collect(const ggml_backend_memory_allocation * allocation, const ggml_backend_memory_binding * binding, void * opaque) {
    auto & result = *static_cast<observations *>(opaque);
    observation copied = {*allocation, *binding, allocation->native_api ? allocation->native_api : "", {}};
    for (size_t i = 0; i < allocation->property_count; ++i) {
        GGML_ASSERT(copied.properties.emplace(allocation->properties[i].name, allocation->properties[i].value).second);
    }
    copied.allocation.native_api = nullptr;
    copied.allocation.properties = nullptr;
    result.push_back(std::move(copied));
}

static void test_coverage() {
    allocation_provider provider;
    ggml_backend_buffer_ptr buffer(provider.buffer());
    observations result;
    const auto size = ggml_backend_buffer_get_size(buffer.get());
    provider.supported = false;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(result.empty() && provider.calls == 0);
    provider.supported = true;
    provider.old_version_only = true;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(provider.calls == 0);
    provider.old_version_only = false;
    provider.type.device = nullptr;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    provider.type.device = &provider.device;
    provider.device.reg = nullptr;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    provider.device.reg = &provider.reg;
    GGML_ASSERT(provider.calls == 0 && result.empty());
    provider.opaque = true; // a composite may return a proved subset without claiming full coverage
    provider.status = GGML_BACKEND_MEMORY_ALLOCATION_PARTIAL;
    GGML_ASSERT(ggml_backend_memory_buffer_flags(buffer.get()) & GGML_BACKEND_MEMORY_BUFFER_OPAQUE);
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_PARTIAL);
    GGML_ASSERT(result.size() == 1 && provider.calls == 1);
    provider.status = GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE;
    result.clear();
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(result.size() == 1); // callbacks alone do not establish successful coverage; discard this result
    provider.emit = false;
    result.clear();
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(result.empty());
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == size);
    ggml_backend_buffer_ptr empty(provider.buffer(0));
    provider.status = GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(empty.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(result.empty());
    provider.supported = false;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(empty.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(nullptr, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), nullptr, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE);
    ggml_backend_buffer_ptr cpu(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 256));
    GGML_ASSERT(cpu && ggml_backend_buffer_get_size(cpu.get()) == 256);
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(cpu.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(ggml_backend_buffer_get_size(cpu.get()) == 256);
}

static void test_identity_and_bindings() {
    allocation_provider first, second;
    first.properties = {{"vulkan.memory_type_index", 0}, {"vulkan.heap_index", 0}, {"vulkan.memory_property_flags", 3}};
    first.allocation.native_api = "vkAllocateMemory";
    first.allocation.locality = GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL;
    second.properties = first.properties;
    // An identical token/heap on another context/device is not the same allocation.
    second.allocation.identity = first.allocation.identity;
    ggml_backend_buffer_ptr a(first.buffer()), b(first.buffer()), c(second.buffer());
    observations result;
    for (auto * buffer : {a.get(), b.get(), c.get()}) {
        GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    }
    GGML_ASSERT(result.size() == 3);
    GGML_ASSERT(result[0].allocation.identity == result[1].allocation.identity);
    GGML_ASSERT(result[0].allocation.identity_domain == result[1].allocation.identity_domain);
    GGML_ASSERT(result[0].allocation.identity == result[2].allocation.identity);
    GGML_ASSERT(result[0].allocation.identity_domain != result[2].allocation.identity_domain);
    GGML_ASSERT(result[0].allocation.device != result[2].allocation.device);
    GGML_ASSERT(result[0].properties == result[2].properties);
    GGML_ASSERT(result[0].properties.size() == 3 && result[0].properties.at("vulkan.memory_type_index") == 0);
    GGML_ASSERT(result[0].properties.at("vulkan.memory_property_flags") == 3);
    GGML_ASSERT(result[0].native_api == "vkAllocateMemory");
    GGML_ASSERT(result[0].allocation.size_known && result[0].allocation.size_bytes == 8192);
    GGML_ASSERT(result[0].allocation.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL);
    GGML_ASSERT(result[0].binding.known_fields == 7 && result[0].binding.allocation_offset == 1024 && result[0].binding.size_bytes == 4096);

    // A different provider can refer to the same native allocation and its proved association.
    second.allocation.identity_domain = first.allocation.identity_domain;
    second.allocation.device = &first.device;
    result.clear();
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(c.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(result[0].allocation.identity_domain == first.allocation.identity_domain);
    GGML_ASSERT(result[0].allocation.identity == first.allocation.identity);
    GGML_ASSERT(result[0].allocation.device != ggml_backend_memory_buffer_device(c.get()));

    // Generic containers borrow their first child's type: do not dispatch its private context as a leaf.
    ggml_backend_buffer_t children[] = {a.release(), b.release()};
    ggml_backend_buffer_ptr multi(ggml_backend_multi_buffer_alloc_buffer(children, 2));
    const auto calls = first.calls;
    result.clear();
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(multi.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED);
    GGML_ASSERT(first.calls == calls && result.empty());
    std::vector<ggml_backend_buffer_t> leaves;
    ggml_backend_buffer_visit(multi.get(), [](ggml_backend_buffer_t buffer, void * opaque) {
        static_cast<std::vector<ggml_backend_buffer_t> *>(opaque)->push_back(buffer);
    }, &leaves);
    GGML_ASSERT(leaves.size() == 2 && leaves[0] == children[0] && leaves[1] == children[1]);
    for (auto * leaf : leaves) {
        GGML_ASSERT(ggml_backend_memory_buffer_allocations(leaf, collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    }
    GGML_ASSERT(result.size() == 2 && first.calls == calls + 2);
    GGML_ASSERT(ggml_backend_buffer_get_size(multi.get()) == 8192);
}

static void test_backend_neutral_metadata() {
    allocation_provider provider;
    ggml_backend_buffer_ptr buffer(provider.buffer());
    const struct {
        const char * api;
        ggml_backend_memory_allocation_kind kind;
    } cases[] = {
        {"cudaMallocManaged", GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED},
        {"hipMallocManaged", GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED},
        {"sycl::malloc_shared", GGML_BACKEND_MEMORY_ALLOCATION_KIND_SHARED},
        {"aligned_alloc", GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM},
    };
    for (const auto & item : cases) {
        provider.allocation.native_api = item.api;
        provider.allocation.kind = item.kind;
        provider.allocation.device = nullptr; // association cannot be inferred from the GPU provider
        provider.allocation.size_known = false;
        provider.allocation.size_bytes = 999; // invalid unless size_known
        provider.binding.known_fields = GGML_BACKEND_MEMORY_BINDING_SIZE;
        provider.binding.size_bytes = 0; // a proved zero range differs from an unknown range
        observations result;
        GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
        GGML_ASSERT(result.size() == 1 && result[0].native_api == item.api && result[0].allocation.kind == item.kind);
        GGML_ASSERT(!result[0].allocation.device && !result[0].allocation.size_known);
        GGML_ASSERT(result[0].allocation.locality == GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN);
        GGML_ASSERT(result[0].properties.empty() && result[0].binding.known_fields == GGML_BACKEND_MEMORY_BINDING_SIZE);
        GGML_ASSERT(result[0].binding.size_bytes == 0);
    }
    provider.allocation.kind = GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE;
    provider.allocation.native_api = "zeMemAllocDevice";
    provider.properties = {{"level_zero.memory_ordinal", 0}};
    observations result;
    GGML_ASSERT(ggml_backend_memory_buffer_allocations(buffer.get(), collect, &result) == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE);
    GGML_ASSERT(result[0].properties.count("level_zero.memory_ordinal") == 1 && result[0].properties.count("vulkan.heap_index") == 0);
}

int main() {
    ggml_backend_load_all();
    test_coverage();
    test_identity_and_bindings();
    test_backend_neutral_metadata();
    std::puts("memory allocations PASSED");
}
