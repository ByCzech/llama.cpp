#include "server-memory-locality.h"

#include <exception>
#include <limits>

namespace {
using json = common_json;

const char * kind_name(ggml_backend_memory_allocation_kind kind) {
    switch (kind) {
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_DEVICE: return "device";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST: return "host";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_HOST_PINNED: return "host_pinned";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_MANAGED: return "managed";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_SHARED: return "shared";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_SYSTEM: return "system";
        case GGML_BACKEND_MEMORY_ALLOCATION_KIND_UNKNOWN: return "unknown";
    }
    return "unknown";
}

const char * origin_name(ggml_backend_memory_allocation_origin origin) {
    switch (origin) {
        case GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_ALLOCATED: return "allocated";
        case GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_IMPORTED: return "imported";
        case GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_REGISTERED: return "registered";
        case GGML_BACKEND_MEMORY_ALLOCATION_ORIGIN_UNKNOWN: return "unknown";
    }
    return "unknown";
}

const char * locality_name(ggml_backend_memory_allocation_locality locality) {
    switch (locality) {
        case GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_DEVICE_LOCAL: return "device_local";
        case GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_NON_DEVICE_LOCAL: return "non_device_local";
        case GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN: return "unknown";
    }
    return "unknown";
}

bool published_property(const std::string & name) {
    // Publish only attributes with defined numeric semantics, never arbitrary native tokens/handles.
    return name == "vulkan.memory_type_index" || name == "vulkan.heap_index" ||
        name == "vulkan.memory_property_flags" || name == "vulkan.heap_flags" || name == "level_zero.memory_ordinal";
}

bool valid_range(bool offset_known, uint64_t offset, bool size_known, uint64_t size, bool span_known, uint64_t span) {
    if (offset_known && size_known && size > std::numeric_limits<uint64_t>::max() - offset) { return false; }
    if (!span_known) { return true; }
    if (offset_known && offset > span) { return false; }
    return !size_known || size <= span - (offset_known ? offset : 0);
}

struct record {
    const void * domain;
    const void * identity;
    ggml_backend_dev_t device;
    json description;
    json range;
};

struct probe {
    bool buffer_size_known;
    uint64_t buffer_size;
    bool invalid = false;
    bool failed = false;
    std::vector<record> records;
};

void collect(const ggml_backend_memory_allocation * a, const ggml_backend_memory_binding * b, void * opaque) noexcept {
    auto & result = *static_cast<probe *>(opaque);
    if (result.failed) { return; }
    try {
        if (!a || !b || !a->identity_domain || !a->identity || (a->property_count && !a->properties) ||
            (b->known_fields & ~uint32_t(7)) ||
            (a->locality != GGML_BACKEND_MEMORY_ALLOCATION_LOCALITY_UNKNOWN && !a->device)) {
            result.invalid = true;
            return;
        }
        const bool length = (b->known_fields & GGML_BACKEND_MEMORY_BINDING_SIZE) != 0;
        const bool buffer_offset = (b->known_fields & GGML_BACKEND_MEMORY_BINDING_BUFFER_OFFSET) != 0;
        const bool allocation_offset = (b->known_fields & GGML_BACKEND_MEMORY_BINDING_ALLOCATION_OFFSET) != 0;
        if (!valid_range(buffer_offset, b->buffer_offset, length, b->size_bytes, result.buffer_size_known, result.buffer_size) ||
            !valid_range(allocation_offset, b->allocation_offset, length, b->size_bytes, a->size_known, a->size_bytes)) {
            result.invalid = true;
            return;
        }
        json properties = json::object();
        std::set<std::string> names;
        for (size_t i = 0; i < a->property_count; ++i) {
            const auto & property = a->properties[i];
            if (!property.name || !*property.name || !names.insert(property.name).second) {
                result.invalid = true;
                return;
            }
            if (published_property(property.name)) { properties[property.name] = property.value; }
        }
        result.records.push_back({a->identity_domain, a->identity, a->device,
            {{"kind", kind_name(a->kind)}, {"origin", origin_name(a->origin)},
             {"native_api", a->native_api ? json(a->native_api) : json()},
             {"size_bytes", a->size_known ? json(a->size_bytes) : json()},
             {"locality", locality_name(a->locality)}, {"native_properties", std::move(properties)}},
            {{"buffer_offset_bytes", buffer_offset ? json(b->buffer_offset) : json()},
             {"allocation_offset_bytes", allocation_offset ? json(b->allocation_offset) : json()},
             {"size_bytes", length ? json(b->size_bytes) : json()}}});
    } catch (const std::exception &) {
        // The native callback ABI must not unwind through a backend. Discard this probe on return.
        result.failed = true;
    }
}
} // namespace

void server_memory_locality::missing(const std::string & buffer_id, const char * reason) {
    failures[buffer_id].insert(reason);
}

void server_memory_locality::buffer(ggml_backend_buffer_t buffer, const common_json & description,
        const std::function<common_json(ggml_backend_dev_t)> & device) {
    const auto id = description["id"].get<std::string>();
    const bool sized = !description["size_bytes"].is_null();
    probe result {sized, sized ? description["size_bytes"].get<uint64_t>() : 0, false, false, {}};
    ggml_backend_memory_allocation_status status;
    try {
        status = ggml_backend_memory_buffer_allocations(buffer, collect, &result);
    } catch (const std::exception &) {
        missing(id, "collection_failed");
        return;
    }
    if (result.failed) { missing(id, "collection_failed"); return; }
    switch (status) {
        case GGML_BACKEND_MEMORY_ALLOCATION_UNSUPPORTED: missing(id, "unsupported_backend"); return;
        case GGML_BACKEND_MEMORY_ALLOCATION_UNAVAILABLE: missing(id, "allocation_metadata_unavailable"); return;
        case GGML_BACKEND_MEMORY_ALLOCATION_PARTIAL: missing(id, "incomplete_allocation_metadata"); break;
        case GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE: break;
        default: missing(id, "invalid_metadata"); return;
    }
    if (result.invalid) { missing(id, "invalid_metadata"); }
    if (status == GGML_BACKEND_MEMORY_ALLOCATION_COMPLETE && !result.invalid && result.records.empty()) {
        empty_buffers.insert(id);
    }
    for (auto & entry : result.records) {
        try {
            entry.description["device_ids"] = device(entry.device);
        } catch (const std::exception &) {
            missing(id, "collection_failed");
            continue;
        }
        auto & domain = indices[entry.domain];
        auto found = domain.find(entry.identity);
        size_t index;
        if (found == domain.end()) {
            index = allocations.size();
            allocations.push_back({std::move(entry.description), false});
            domain.emplace(entry.identity, index);
        } else {
            index = found->second;
            if (allocations[index].description != entry.description) { allocations[index].invalid = true; }
        }
        bool duplicate = false;
        for (const auto & binding : bindings) {
            if (binding.buffer_id == id && binding.allocation_index == index && binding.range == entry.range) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) { bindings.push_back({id, index, std::move(entry.range)}); }
    }
}

common_json server_memory_locality::finish() const {
    auto missing = failures;
    common_json result = {{"status", "known"}, {"scope", "backend_buffer_allocations"},
        {"physical_aliasing", "unknown"}, {"missing_components", common_json::array()},
        {"allocations", common_json::array()}, {"bindings", common_json::array()}};
    bool usable = !empty_buffers.empty();
    for (const auto & binding : bindings) {
        const auto & allocation = allocations[binding.allocation_index];
        if (allocation.invalid) {
            missing[binding.buffer_id].insert("conflicting_allocation_metadata");
            continue;
        }
        auto item = binding.range;
        item["buffer_id"] = binding.buffer_id;
        item["allocation_id"] = "a" + std::to_string(binding.allocation_index + 1);
        result["bindings"].push_back(std::move(item));
        usable = true;
    }
    for (size_t i = 0; i < allocations.size(); ++i) {
        if (allocations[i].invalid) { continue; }
        auto item = allocations[i].description;
        item["id"] = "a" + std::to_string(i + 1);
        result["allocations"].push_back(std::move(item));
    }
    for (const auto & buffer : missing) {
        for (const auto & reason : buffer.second) {
            result["missing_components"].push_back({{"buffer_id", buffer.first.empty() ? common_json() : common_json(buffer.first)}, {"reason", reason}});
        }
    }
    result["status"] = missing.empty() ? "known" : usable ? "partial" : "unavailable";
    return result;
}
