#include "../tools/server/server-memory-snapshot.h"
#include "build-info.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

static void test_lifecycle() {
    server_memory_reporter reporter;
    GGML_ASSERT(reporter.read().is_null());
    server_memory_collector empty;
    auto inventory = empty.finish();
    reporter.publish(inventory, "loaded", "after_load");
    auto first = reporter.read();
    GGML_ASSERT(first.size() == 15);
    GGML_ASSERT(first["allocation_locality"]["status"] == "known");
    GGML_ASSERT(first["schema_version"].get<int>() == 1);
    GGML_ASSERT(first["runtime_fingerprint"] == llama_build_info());
    GGML_ASSERT(first["scope"] == "backend_buffers");
    GGML_ASSERT(first["precision"] == "exact_bytes");
    GGML_ASSERT(first["snapshot_sequence"].get<uint64_t>() == 1);
    GGML_ASSERT(first["snapshot_stage"] == "after_load");
    GGML_ASSERT(first["model_state"] == "loaded");
    reporter.publish(inventory, "loaded", "after_batch", false);
    GGML_ASSERT(reporter.read()["snapshot_sequence"].get<uint64_t>() == 1);
    reporter.publish(inventory, "loaded", "after_request");
    const auto completed = reporter.read();
    GGML_ASSERT(completed["snapshot_sequence"].get<uint64_t>() == 2);
    GGML_ASSERT(completed["snapshot_stage"] == "after_request");
    // Mutating a returned copy cannot change a published snapshot.
    first["status"] = "corrupted";
    GGML_ASSERT(reporter.read()["status"] == "known");
    reporter.transition("unloading", "before_unload");
    const auto unloading = reporter.read();
    GGML_ASSERT(unloading["status"] == "unavailable");
    GGML_ASSERT(unloading["buffers"].empty());
    GGML_ASSERT(unloading["snapshot_sequence"].get<uint64_t>() == 3);
    reporter.transition("sleeping", "after_unload", true);
    const auto sleeping = reporter.read();
    GGML_ASSERT(sleeping["status"] == "known");
    GGML_ASSERT(sleeping["model_state"] == "sleeping");
    GGML_ASSERT(sleeping["buffers"].empty());
    GGML_ASSERT(sleeping["allocation_locality"]["status"] == "known");
    GGML_ASSERT(sleeping["allocation_locality"]["allocations"].empty());
    GGML_ASSERT(sleeping["snapshot_sequence"].get<uint64_t>() == 4);
    const auto age = sleeping["snapshot_age_ms"].get<uint64_t>();
    GGML_ASSERT(reporter.read()["snapshot_age_ms"].get<uint64_t>() >= age);
    reporter.transition("failed", "after_load_failure");
    GGML_ASSERT(reporter.read()["status"] == "unavailable");
    reporter.transition("loaded", "after_batch", false, "collection_failed");
    const auto failed_collection = reporter.read();
    GGML_ASSERT(failed_collection["status"] == "unavailable");
    GGML_ASSERT(failed_collection["buffers"].empty());
    GGML_ASSERT(failed_collection["missing_components"][0]["reason"] == "collection_failed");
}

static void test_concurrent_reads() {
    server_memory_reporter reporter("b-test");
    std::mutex mutex;
    std::condition_variable cv;
    bool published = false, release = false;
    std::atomic<bool> finished{false};
    std::thread owner([&] {
        server_memory_collector empty;
        auto inventory = empty.finish();
        inventory["buffers"].push_back({{"id", "b1"}, {"size_bytes", uint64_t(1)}});
        inventory["allocation_locality"]["allocations"].push_back({{"id", "a1"}, {"size_bytes", uint64_t(1)}});
        reporter.publish(inventory, "loaded", "after_load");
        {
            std::unique_lock<std::mutex> lock(mutex);
            published = true;
            cv.notify_one();
            // Simulates an owner occupied by inference. The reader must not need this lock.
            cv.wait(lock, [&] { return release; });
        }
        for (uint64_t i = 2; i <= 200; ++i) {
            inventory["buffers"][0]["size_bytes"] = i;
            inventory["allocation_locality"]["allocations"][0]["size_bytes"] = i;
            reporter.publish(inventory, "loaded", "after_batch", false);
        }
        finished.store(true);
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return published; });
    }
    GGML_ASSERT(reporter.read()["snapshot_sequence"].get<uint64_t>() == 1);
    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
    }
    cv.notify_one();
    uint64_t previous = 0;
    do {
        const auto report = reporter.read();
        const auto sequence = report["snapshot_sequence"].get<uint64_t>();
        GGML_ASSERT(sequence >= previous);
        GGML_ASSERT(sequence == report["buffers"][0]["size_bytes"].get<uint64_t>());
        GGML_ASSERT(sequence == report["allocation_locality"]["allocations"][0]["size_bytes"].get<uint64_t>());
        GGML_ASSERT(report["runtime_fingerprint"] == "b-test");
        previous = sequence;
    } while (!finished.load());
    owner.join();
    GGML_ASSERT(reporter.read()["snapshot_sequence"].get<uint64_t>() == 200);
}

static void test_locality_only_updates() {
    server_memory_reporter reporter("test-locality");
    server_memory_collector empty;
    auto inventory = empty.finish();
    inventory["allocation_locality"]["allocations"].push_back({{"id", "a1"}, {"size_bytes", 4096}});
    reporter.publish(inventory, "loaded", "after_load");
    auto saved = reporter.read();
    reporter.publish(inventory, "loaded", "after_batch", false);
    GGML_ASSERT(reporter.read()["snapshot_sequence"] == 1);
    inventory["allocation_locality"]["allocations"][0]["size_bytes"] = 8192;
    reporter.publish(inventory, "loaded", "after_batch", false);
    GGML_ASSERT(reporter.read()["snapshot_sequence"] == 2);
    GGML_ASSERT(saved["allocation_locality"]["allocations"][0]["size_bytes"] == 4096);
    saved["allocation_locality"]["allocations"].clear();
    GGML_ASSERT(reporter.read()["allocation_locality"]["allocations"][0]["size_bytes"] == 8192);
    reporter.transition("unloading", "before_unload");
    GGML_ASSERT(reporter.read()["allocation_locality"]["status"] == "unavailable");
    reporter.transition("sleeping", "after_unload", true);
    GGML_ASSERT(reporter.read()["allocation_locality"]["allocations"].empty());
    GGML_ASSERT(reporter.read()["allocation_locality"]["bindings"].empty());
    GGML_ASSERT(reporter.read()["allocation_locality"]["status"] == "known");
    reporter.publish(inventory, "loaded", "after_load");
    GGML_ASSERT(reporter.read()["snapshot_sequence"] == 5);
}

int main() {
    test_lifecycle();
    test_concurrent_reads();
    test_locality_only_updates();
    std::puts("memory snapshots PASSED");
}
