#pragma once

#include "server-memory.h"

#include <chrono>

// Only capture(), publish(), and transition() touch owner-thread state. read()
// copies an immutable snapshot and never consults the inference task queue.
class server_memory_reporter {
public:
    explicit server_memory_reporter(std::string fingerprint = {});
    void capture(const server_memory_sources & sources, const char * stage, bool force = true);
    void publish(json inventory, const char * model_state, const char * stage, bool force = true);
    void transition(const char * model_state, const char * stage, bool known_empty = false,
                    const char * reason = "transition_in_progress");
    json read() const; // null until the first snapshot is published

private:
    struct snapshot {
        json inventory;
        std::string model_state;
        std::string stage;
        uint64_t sequence;
        std::chrono::steady_clock::time_point captured;
    };
    const std::string fingerprint;
    server_memory_native native;
    uint64_t sequence = 0;
    std::shared_ptr<const snapshot> current;
};
