#pragma once

#include "json.h"
#include "ggml-backend-memory.h"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

// Owner-thread state. Native keys live only until the owning collector is reset.
class server_memory_locality {
public:
    void buffer(ggml_backend_buffer_t buffer, const common_json & description,
        const std::function<common_json(ggml_backend_dev_t)> & device);
    void missing(const std::string & buffer_id, const char * reason);
    common_json finish() const;

private:
    struct allocation {
        common_json description;
        bool invalid = false;
    };
    struct binding {
        std::string buffer_id;
        size_t allocation_index;
        common_json range;
    };
    std::map<const void *, std::map<const void *, size_t>> indices;
    std::vector<allocation> allocations;
    std::vector<binding> bindings;
    std::set<std::string> empty_buffers;
    std::map<std::string, std::set<std::string>> failures;
};
