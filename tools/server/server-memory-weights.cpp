#include "server-memory-weights.h"

#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using json = common_json;

void add_missing(std::map<std::string, json> & entries, const json & relationship, const char * reason) {
    json entry = {{"owner_id", relationship["owner_id"]}, {"component", relationship["component"]},
        {"layer_index", relationship["layer_index"]}, {"reason", reason}};
    entries.emplace(entry.dump(), entry);
}

struct claim {
    uint64_t begin;
    uint64_t end;
    json relationship;
    json assignment;
};

} // namespace

struct server_memory_weights::state {
    std::map<std::string, std::vector<claim>> buffers;
    std::map<std::string, json> missing;
    bool usable = false;
    json snapshot;
};

server_memory_weights::server_memory_weights() : data(std::make_shared<state>()) {}

server_memory_weights::state & server_memory_weights::write() {
    if (!data.unique()) { data = std::make_shared<state>(*data); }
    data->snapshot = json();
    return *data;
}

void server_memory_weights::missing(const json & relationship, const char * reason) {
    add_missing(write().missing, relationship, reason);
}

void server_memory_weights::tensor(const json & buffer, const llama_memory_weight_storage_view & storage,
                                   const json & relationship, const json & assignment) {
    if (storage.missing_reason) {
        missing(relationship, storage.missing_reason);
        return;
    }
    if (buffer["size_bytes"].is_null()) {
        missing(relationship, "buffer_bounds_unavailable");
        return;
    }
    const auto capacity = buffer["size_bytes"].get<uint64_t>();
    for (const auto range : storage.ranges) {
        if (range.size == 0 || range.offset > capacity || range.size > capacity - range.offset ||
                uint64_t(range.size) > std::numeric_limits<uint64_t>::max() - uint64_t(range.offset)) {
            missing(relationship, "invalid_storage_range");
            return;
        }
    }
    auto & current = write();
    current.usable = true; // a proved empty tensor is usable evidence too
    if (assignment["status"] != "known") {
        if (assignment["reasons"].empty()) { add_missing(current.missing, relationship, "assignment_unknown"); }
        for (const auto & reason : assignment["reasons"]) {
            add_missing(current.missing, relationship, reason.get<std::string>().c_str());
        }
    }
    auto & claims = current.buffers[buffer["id"].get<std::string>()];
    for (const auto range : storage.ranges) {
        claims.push_back({uint64_t(range.offset), uint64_t(range.offset) + uint64_t(range.size), relationship, assignment});
    }
}

const json & server_memory_weights::finish() const {
    if (!data->snapshot.is_null()) { return data->snapshot; }
    auto missing = data->missing;
    json intervals = json::array();
    for (const auto & buffer : data->buffers) {
        const auto & claims = buffer.second;
        struct event { size_t index; bool begin; };
        std::map<uint64_t, std::vector<event>> events;
        for (size_t i = 0; i < claims.size(); ++i) {
            events[claims[i].begin].push_back({i, true});
            events[claims[i].end].push_back({i, false});
        }
        std::set<size_t> active;
        for (auto position = events.begin(); position != events.end(); ++position) {
            for (const auto event : position->second) {
                if (event.begin) { active.insert(event.index); }
                else { active.erase(event.index); }
            }
            const auto next = std::next(position);
            if (active.empty() || next == events.end()) { continue; }
            std::map<std::string, json> relationships;
            std::set<std::string> devices, sources, reasons;
            bool unknown = false;
            for (const auto index : active) {
                const auto & entry = claims[index];
                relationships.emplace(entry.relationship.dump(), entry.relationship);
                const auto & assignment = entry.assignment;
                unknown |= assignment["status"] != "known";
                for (const auto & id : assignment["device_ids"]) { devices.insert(id.get<std::string>()); }
                for (const auto & source : assignment["sources"]) { sources.insert(source.get<std::string>()); }
                for (const auto & reason : assignment["reasons"]) { reasons.insert(reason.get<std::string>()); }
            }
            if (devices.size() > 1) {
                reasons.insert("assignment_conflict");
                devices.clear(); // conflicting candidates are not a proved CPU/GPU distribution
                unknown = true;
                for (const auto & relationship : relationships) {
                    add_missing(missing, relationship.second, "assignment_conflict");
                }
            }
            json relations = json::array();
            for (const auto & relationship : relationships) { relations.push_back(relationship.second); }
            json assignment = {{"status", unknown ? (devices.empty() ? "unavailable" : "partial") : "known"},
                {"device_ids", devices}, {"sources", sources}, {"reasons", reasons}};
            const uint64_t begin = position->first, size = next->first - begin;
            // Tensor boundaries need not split adjacent intervals with identical evidence and relationships.
            if (!intervals.empty()) {
                auto & previous = intervals[intervals.size() - 1];
                if (previous["buffer_id"] == buffer.first && previous["assignment"] == assignment &&
                        previous["relationships"] == relations &&
                        previous["offset_bytes"].get<uint64_t>() + previous["size_bytes"].get<uint64_t>() == begin) {
                    previous["size_bytes"] = previous["size_bytes"].get<uint64_t>() + size;
                    continue;
                }
            }
            intervals.push_back({{"id", "w" + std::to_string(intervals.size() + 1)}, {"buffer_id", buffer.first},
                {"offset_bytes", begin}, {"size_bytes", size}, {"assignment", assignment}, {"relationships", relations}});
        }
    }
    json failures = json::array();
    for (const auto & entry : missing) { failures.push_back(entry.second); }
    data->snapshot = {{"scope", "model_weights"}, {"deduplication", "buffer_ranges"},
        {"status", missing.empty() ? "known" : data->usable ? "partial" : "unavailable"},
        {"missing_components", failures}, {"intervals", intervals}};
    return data->snapshot;
}
