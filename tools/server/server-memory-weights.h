#pragma once

#include "json.h"
#include "../../src/llama-memory-weight-storage.h"

#include <memory>

// Owner-thread value state. Only copied IDs, evidence and byte ranges are retained.
// Copies share immutable captured data until a write; finish() caches the interval sweep.
class server_memory_weights {
public:
    server_memory_weights();
    void tensor(const common_json & buffer, const llama_memory_weight_storage_view & storage,
                const common_json & relationship, const common_json & assignment);
    void missing(const common_json & relationship, const char * reason);
    const common_json & finish() const;

private:
    struct state;
    std::shared_ptr<state> data;
    state & write();
};
