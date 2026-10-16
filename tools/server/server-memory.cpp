#include "server-memory.h"
#include "../mtmd/mtmd-memory.h"

#include <limits>

namespace {

constexpr size_t no_owner = std::numeric_limits<size_t>::max();

void append_unique(json & array, const json & value) {
    for (const auto & entry : array) {
        if (entry == value) {
            return;
        }
    }
    array.push_back(value);
}

const char * component_name(llama_memory_component component) {
    switch (component) {
        case LLAMA_MEMORY_COMPONENT_MODEL_WEIGHTS: return "model_weights";
        case LLAMA_MEMORY_COMPONENT_KV_CACHE: return "kv_cache";
        case LLAMA_MEMORY_COMPONENT_RECURRENT_STATE: return "recurrent_state";
        case LLAMA_MEMORY_COMPONENT_STATE_OTHER: return "state_other";
        case LLAMA_MEMORY_COMPONENT_COMPUTE: return "compute";
        case LLAMA_MEMORY_COMPONENT_OUTPUT: return "output";
        case LLAMA_MEMORY_COMPONENT_MOE_CACHE: return "moe_cache";
        case LLAMA_MEMORY_COMPONENT_CONTROL_VECTOR: return "control_vector";
        case LLAMA_MEMORY_COMPONENT_ADAPTER_WEIGHTS: return "adapter_weights";
        case LLAMA_MEMORY_COMPONENT_PROJECTOR_WEIGHTS: return "projector_weights";
        case LLAMA_MEMORY_COMPONENT_PROJECTOR_COMPUTE: return "projector_compute";
    }
    return "state_other";
}

const char * part_name(llama_memory_model_part part) {
    switch (part) {
        case LLAMA_MEMORY_MODEL_PART_INPUT: return "input_weights";
        case LLAMA_MEMORY_MODEL_PART_OUTPUT: return "output_weights";
        case LLAMA_MEMORY_MODEL_PART_REPEATING: return "repeating_layers";
        case LLAMA_MEMORY_MODEL_PART_MTP: return "mtp_layers";
        case LLAMA_MEMORY_MODEL_PART_OTHER: return "other_weights";
    }
    return "other_weights";
}

const char * device_type(ggml_backend_dev_t native) {
    switch (ggml_backend_dev_type(native)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        case GGML_BACKEND_DEVICE_TYPE_META: return "meta";
    }
    return "unknown";
}

const char * assignment_source(llama_memory_assignment_source source) {
    switch (source) {
        case LLAMA_MEMORY_ASSIGNMENT_SELECTION: return "loader_selection";
        case LLAMA_MEMORY_ASSIGNMENT_OVERRIDE: return "loader_override";
        case LLAMA_MEMORY_ASSIGNMENT_MMAP: return "loader_mmap";
        case LLAMA_MEMORY_ASSIGNMENT_LAZY: return "loader_lazy";
        case LLAMA_MEMORY_ASSIGNMENT_UNKNOWN: return nullptr;
    }
    return nullptr;
}

const char * backing_name(llama_memory_backing backing) {
    switch (backing) {
        case LLAMA_MEMORY_BACKING_ALLOCATED: return "allocated";
        case LLAMA_MEMORY_BACKING_MMAP: return "mmap";
        case LLAMA_MEMORY_BACKING_UNKNOWN: return "unknown";
    }
    return "unknown";
}

} // namespace

server_memory_collector::server_memory_collector() {
    data = {{"status", "known"}, {"missing_components", json::array()},
            {"owners", json::array()}, {"devices", json::array()}, {"buffers", json::array()},
            {"placement", {{"status", "known"}, {"missing_components", json::array()}, {"layer_groups", json::array()}}}};
}

size_t server_memory_collector::owner(const void * identity, const char * kind, const char * role) {
    auto & indices = owner_indices[kind];
    auto found = indices.find(identity);
    size_t index;
    if (found != indices.end()) {
        index = found->second;
    } else {
        index = data["owners"].size();
        indices.emplace(identity, index);
        data["owners"].push_back({{"id", "o" + std::to_string(index + 1)}, {"kind", kind}, {"roles", json::array()}});
    }
    append_unique(data["owners"][index]["roles"], role);
    return index;
}

json server_memory_collector::device(ggml_backend_dev_t native) {
    if (!native) {
        return json::array();
    }
    auto found = device_indices.find(native);
    size_t index;
    if (found != device_indices.end()) {
        index = found->second;
    } else {
        index = data["devices"].size();
        device_indices.emplace(native, index);
        auto * reg = ggml_backend_dev_backend_reg(native);
        const auto * name = ggml_backend_dev_name(native);
        const auto * backend = reg ? ggml_backend_reg_name(reg) : nullptr;
        data["devices"].push_back({{"id", "d" + std::to_string(index + 1)},
            {"backend", backend ? json(backend) : json()}, {"name", name ? json(name) : json()}, {"type", device_type(native)}});
    }
    return json::array({data["devices"][index]["id"]});
}

void server_memory_collector::missing(size_t owner_index, const char * component, const char * reason, bool placement) {
    json id = owner_index == no_owner ? json() : data["owners"][owner_index]["id"];
    auto & entries = placement ? data["placement"]["missing_components"] : data["missing_components"];
    append_unique(entries, {{"owner_id", id}, {"component", component}, {"reason", reason}});
}

void server_memory_collector::known_source() {
    usable = true;
}

void server_memory_collector::buffer(size_t owner_index, const llama_memory_buffer_view & view) {
    if (!view.buffer) {
        missing(owner_index, component_name(view.component), "size_unavailable");
        return;
    }
    auto found = buffer_indices.find(view.buffer);
    size_t index;
    if (found != buffer_indices.end()) {
        index = found->second;
    } else {
        index = data["buffers"].size();
        buffer_indices.emplace(view.buffer, index);
        backings.emplace_back();
        auto * type = ggml_backend_buffer_get_type(view.buffer);
        auto * native = ggml_backend_memory_buffer_device(view.buffer);
        const auto flags = ggml_backend_memory_buffer_flags(view.buffer);
        const bool opaque = (flags & GGML_BACKEND_MEMORY_BUFFER_OPAQUE) != 0;
        const bool host = ggml_backend_buffer_is_host(view.buffer) || (native && ggml_backend_dev_type(native) == GGML_BACKEND_DEVICE_TYPE_CPU);
        const char * kind = "unknown";
        if (!opaque) {
            if (flags & GGML_BACKEND_MEMORY_BUFFER_HOST_PINNED) {
                kind = "host_pinned";
            } else if (host) {
                kind = "host";
            } else if (native && ggml_backend_dev_type(native) != GGML_BACKEND_DEVICE_TYPE_META) {
                kind = "device";
            }
            usable = true;
        }
        const auto * name = type ? ggml_backend_buft_name(type) : nullptr;
        data["buffers"].push_back({{"id", "b" + std::to_string(index + 1)}, {"owners", json::array()},
            {"device_ids", device(native)}, {"buffer_type", name ? json(name) : json()},
            {"memory_kind", kind}, {"backing", "unknown"},
            {"size_bytes", opaque ? json() : json(uint64_t(ggml_backend_buffer_get_size(view.buffer)))}});
    }
    auto & item = data["buffers"][index];
    append_unique(item["owners"], {{"owner_id", data["owners"][owner_index]["id"]}, {"component", component_name(view.component)}});
    if (item["size_bytes"].is_null()) {
        missing(owner_index, component_name(view.component), "opaque_composite");
    }
    if (view.backing != LLAMA_MEMORY_BACKING_UNKNOWN) {
        backings[index].insert(view.backing);
    }
    item["backing"] = backings[index].size() == 1 ? backing_name(*backings[index].begin()) : "unknown";
    if (backings[index].size() > 1) {
        for (const auto & relation : item["owners"]) {
            append_unique(data["missing_components"], {{"owner_id", relation["owner_id"]},
                {"component", relation["component"]}, {"reason", "collection_failed"}});
        }
    }
}

void server_memory_collector::placement(size_t owner_index, const llama_memory_placement_view & view) {
    auto & current = layers[{owner_index, view.part, view.layer}];
    auto * native = view.buffer ? ggml_backend_memory_buffer_device(view.buffer) : nullptr;
    for (const auto & id : device(native)) {
        current.devices.insert(id.get<std::string>());
    }
    const auto & assignment = view.assignment;
    const char * source = assignment_source(assignment.source);
    ggml_backend_dev_t assigned = nullptr;
    const char * reason = nullptr;
    if (!view.known || !view.buffer) {
        reason = "tensor_unclassified";
    } else if ((ggml_backend_memory_buffer_flags(view.buffer) & GGML_BACKEND_MEMORY_BUFFER_OPAQUE) ||
               (native && ggml_backend_dev_type(native) == GGML_BACKEND_DEVICE_TYPE_META)) {
        reason = "opaque_distribution";
    } else if (assignment.source != LLAMA_MEMORY_ASSIGNMENT_UNKNOWN) {
        // A failed or invalidated loader decision must not become a buffer-based guess.
        assigned = source ? assignment.device : nullptr;
        if (!assigned) { reason = "assignment_unknown"; }
    } else if (!native) {
        reason = "device_unknown";
    } else if (ggml_backend_dev_type(native) != GGML_BACKEND_DEVICE_TYPE_CPU && ggml_backend_buffer_is_host(view.buffer)) {
        reason = "opaque_distribution";
    } else {
        assigned = native;
        source = "native_buffer";
    }
    if (assigned && ggml_backend_dev_type(assigned) == GGML_BACKEND_DEVICE_TYPE_META) {
        assigned = nullptr;
        reason = "opaque_distribution";
    }
    if (source) { current.sources.insert(source); }
    if (assigned) {
        for (const auto & id : device(assigned)) { current.assigned_devices.insert(id.get<std::string>()); }
        if (ggml_backend_dev_type(assigned) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            current.cpu = true;
        } else {
            current.accelerator = true;
        }
        placement_usable = true;
    }
    if (reason) {
        current.unknown = true;
        current.reasons.insert(reason);
        missing(owner_index, part_name(view.part), reason, true);
    }
}

json server_memory_collector::finish() const {
    json result = data;
    result["status"] = data["missing_components"].empty() ? "known" : usable ? "partial" : "unavailable";
    auto & placement = result["placement"];
    placement["status"] = placement["missing_components"].empty() ? "known" : placement_usable ? "partial" : "unavailable";
    for (const auto & entry : layers) {
        const auto owner_index = std::get<0>(entry.first);
        const auto part = std::get<1>(entry.first);
        const auto layer_index = std::get<2>(entry.first);
        const auto & current = entry.second;
        const char * location = current.unknown ? "unknown" : current.cpu && current.accelerator ? "mixed" : current.cpu ? "cpu" : current.accelerator ? "accelerator" : "unknown";
        json ids = json::array();
        for (const auto & id : current.devices) {
            ids.push_back(id);
        }
        json group = {{"owner_id", data["owners"][owner_index]["id"]}, {"component", part_name(part)},
            {"layer_indices", json::array()}, {"location", location}, {"device_ids", ids},
            {"assignment", {{"status", current.unknown ? (current.cpu || current.accelerator ? "partial" : "unavailable") : "known"},
                {"device_ids", current.assigned_devices}, {"sources", current.sources}, {"reasons", current.reasons}}}};
        bool merged = false;
        for (auto & existing : placement["layer_groups"]) {
            if (existing["owner_id"] == group["owner_id"] && existing["component"] == group["component"] &&
                existing["location"] == group["location"] && existing["device_ids"] == ids &&
                existing["assignment"] == group["assignment"]) {
                if (layer_index >= 0) {
                    existing["layer_indices"].push_back(layer_index);
                }
                merged = true;
                break;
            }
        }
        if (!merged) {
            if (layer_index >= 0) {
                group["layer_indices"].push_back(layer_index);
            }
            placement["layer_groups"].push_back(std::move(group));
        }
    }
    return result;
}

void server_memory_native::reset() {
    base.reset();
    cached = {};
}

json server_memory_native::capture(const server_memory_sources & sources) {
    if (!sources.main) {
        reset();
        server_memory_collector empty;
        empty.missing(no_owner, "inventory", "transition_in_progress");
        empty.missing(no_owner, "inventory", "transition_in_progress", true);
        return empty.finish();
    }
    if (!base || cached.main != sources.main || cached.draft != sources.draft || cached.draft_is_mtp != sources.draft_is_mtp) {
        reset();
        auto next = std::make_unique<server_memory_collector>();
        auto add_model = [&](const llama_context * ctx, const char * role) {
            if (!ctx) { return; }
            const auto * model = llama_get_model(ctx);
            const size_t owner = next->owner(model, "model", role);
            next->owner(ctx, "context", role);
            try {
                if (!llama_model_visit_buffers(model, [&](const llama_memory_buffer_view & view) { next->buffer(owner, view); })) {
                    next->missing(owner, "model_weights", "collector_not_implemented");
                } else { next->known_source(); }
            } catch (const std::exception &) { next->missing(owner, "model_weights", "collection_failed"); }
            try {
                if (!llama_model_visit_placement(model, [&](const llama_memory_placement_view & view) { next->placement(owner, view); })) {
                    next->missing(owner, "model_weights", "tensor_unclassified", true);
                }
            } catch (const std::exception &) { next->missing(owner, "model_weights", "collection_failed", true); }
        };
        add_model(sources.main, "main");
        add_model(sources.draft, sources.draft_is_mtp ? "mtp" : "draft");
        cached = sources;
        base = std::move(next);
    }
    auto current = *base;
    auto add_context = [&](const llama_context * ctx, const char * role) {
        if (!ctx) { return; }
        const size_t owner = current.owner(ctx, "context", role);
        try {
            if (!llama_context_visit_buffers(ctx, [&](const llama_memory_buffer_view & view) { current.buffer(owner, view); })) {
                current.missing(owner, "state_other", "collector_not_implemented");
            } else { current.known_source(); }
        } catch (const std::exception &) { current.missing(owner, "state_other", "collection_failed"); }
        try {
            if (!llama_model_visit_loras(llama_get_model(ctx), [&](const llama_adapter_lora * adapter) {
                const auto adapter_owner = current.owner(adapter, "adapter", "lora");
                if (!llama_adapter_lora_visit_buffers(adapter, [&](const llama_memory_buffer_view & view) { current.buffer(adapter_owner, view); })) {
                    current.missing(adapter_owner, "adapter_weights", "collector_not_implemented");
                }
            })) {
                current.missing(owner, "adapter_weights", "collector_not_implemented");
            }
        } catch (const std::exception &) { current.missing(owner, "adapter_weights", "collection_failed"); }
    };
    add_context(sources.main, "main");
    add_context(sources.draft, sources.draft_is_mtp ? "mtp" : "draft");
    if (sources.projector) {
        try {
            if (!mtmd_visit_projectors(sources.projector, [&](const clip_ctx * projector, mtmd_memory_role role) {
                const auto owner = current.owner(projector, "projector", role == MTMD_MEMORY_ROLE_VISION ? "vision" : role == MTMD_MEMORY_ROLE_AUDIO ? "audio" : "audio_generation");
                if (!clip_visit_buffers(projector, [&](const llama_memory_buffer_view & view) { current.buffer(owner, view); })) {
                    current.missing(owner, "projector_weights", "collector_not_implemented");
                    current.missing(owner, "projector_compute", "collector_not_implemented");
                } else { current.known_source(); }
            })) {
                current.missing(no_owner, "projector_weights", "collector_not_implemented");
                current.missing(no_owner, "projector_compute", "collector_not_implemented");
            }
        } catch (const std::exception &) {
            current.missing(no_owner, "projector_weights", "collection_failed");
            current.missing(no_owner, "projector_compute", "collection_failed");
        }
    }
    return current.finish();
}
