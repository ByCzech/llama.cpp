#include "server-memory-snapshot.h"
#include "build-info.h"

#include <atomic>
#include <limits>

server_memory_reporter::server_memory_reporter(std::string fingerprint) :
    fingerprint(fingerprint.empty() ? llama_build_info() : std::move(fingerprint)) {}

void server_memory_reporter::capture(const server_memory_sources & sources, const char * stage, bool force) {
    publish(native.capture(sources), "loaded", stage, force);
}

void server_memory_reporter::publish(json inventory, const char * model_state, const char * stage, bool force) {
    const auto previous = std::atomic_load_explicit(&current, std::memory_order_acquire);
    if (!force && previous && previous->model_state == model_state && previous->inventory == inventory) {
        return;
    }
    std::shared_ptr<const snapshot> next = std::make_shared<snapshot>(snapshot {
        std::move(inventory), model_state, stage, ++sequence, std::chrono::steady_clock::now()
    });
    std::atomic_store_explicit(&current, std::move(next), std::memory_order_release);
}

void server_memory_reporter::transition(const char * model_state, const char * stage, bool known_empty, const char * reason) {
    // Invalidate cached buffer identities before any native object can be freed.
    native.reset();
    server_memory_collector empty;
    if (!known_empty) {
        const auto no_owner = std::numeric_limits<size_t>::max();
        empty.missing(no_owner, "inventory", reason);
        empty.missing(no_owner, "inventory", reason, true);
    }
    publish(empty.finish(), model_state, stage);
}

json server_memory_reporter::read() const {
    const auto saved = std::atomic_load_explicit(&current, std::memory_order_acquire);
    if (!saved) {
        return json();
    }
    json result = saved->inventory;
    result["schema_version"] = 1;
    result["runtime_fingerprint"] = fingerprint;
    result["scope"] = "backend_buffers";
    result["precision"] = "exact_bytes";
    result["model_state"] = saved->model_state;
    result["snapshot_sequence"] = saved->sequence;
    result["snapshot_stage"] = saved->stage;
    result["snapshot_age_ms"] = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - saved->captured).count());
    return result;
}
