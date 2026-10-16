#pragma once

#include "json.h"
#include "../../src/llama-memory-placement.h"

#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <string>
#include <vector>

using json = common_json;

struct mtmd_context;

struct server_memory_sources {
    const llama_context * main = nullptr;
    const llama_context * draft = nullptr;
    const mtmd_context * projector = nullptr;
    bool draft_is_mtp = false;
};

// Owner-thread collector. No live native pointer is stored in its JSON result.
class server_memory_collector {
public:
    server_memory_collector();
    size_t owner(const void * identity, const char * kind, const char * role);
    void buffer(size_t owner_index, const llama_memory_buffer_view & view);
    void placement(size_t owner_index, const llama_memory_placement_view & view);
    void missing(size_t owner_index, const char * component, const char * reason, bool placement = false);
    void known_source();
    json finish() const;

private:
    struct layer {
        std::set<std::string> devices; // buffer providers
        std::set<std::string> assigned_devices;
        std::set<std::string> sources;
        std::set<std::string> reasons;
        bool cpu = false;
        bool accelerator = false;
        bool unknown = false;
    };
    json data;
    bool usable = false;
    bool placement_usable = false;
    std::map<std::string, std::map<const void *, size_t>> owner_indices;
    std::map<ggml_backend_dev_t, size_t> device_indices;
    std::map<ggml_backend_buffer_t, size_t> buffer_indices;
    std::vector<std::set<llama_memory_backing>> backings;
    std::map<std::tuple<size_t, llama_memory_model_part, int32_t>, layer> layers;
    json device(ggml_backend_dev_t native);
};

// Cache only static model metadata. reset() must precede destruction or replacement of native owners.
class server_memory_native {
public:
    json capture(const server_memory_sources & sources);
    void reset();

private:
    server_memory_sources cached;
    std::unique_ptr<server_memory_collector> base;
};
