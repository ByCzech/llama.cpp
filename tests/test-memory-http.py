#!/usr/bin/env python3
"""Offline memory-endpoint integration tests using a generated CPU model."""

import argparse
import contextlib
import http.client
import json
import pathlib
import socket
import subprocess
import tempfile
import time


KEY = "tllama-memory-test-key"


class Worker:
    def __init__(self, server, model, directory, *options):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.log = open(directory / (str(self.port) + ".log"), "w+")
        args = [server, "--host", "127.0.0.1", "--port", str(self.port),
                "--api-key", KEY, "--threads", "2", "--ctx-size", "512",
                "--parallel", "1", "--device", "none", "--n-gpu-layers", "0",
                "--no-op-offload", "--no-warmup", "--no-jinja", "--context-shift"]
        if model:
            args += ["--model", str(model)]
        args += list(options)
        self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT)

    def request(self, path, key=KEY, data=None, header="Authorization", timeout=2):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {} if key is None else {header: "Bearer " + key if header == "Authorization" else key}
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            conn.request("GET" if data is None else "POST", path,
                         body=None if data is None else json.dumps(data), headers=headers)
            response = conn.getresponse()
            raw = response.read()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw.decode()
            return response.status, dict(response.getheaders()), body
        finally:
            conn.close()

    def wait(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log.seek(0)
                raise AssertionError("worker exited:\n" + self.log.read())
            try:
                value = predicate()
                if value:
                    return value
            except (ConnectionError, TimeoutError, http.client.HTTPException):
                pass
            time.sleep(0.02)
        raise AssertionError("timed out waiting for worker")

    def ready(self):
        return self.wait(lambda: self.request("/health", key=None)[0] == 200)

    def report(self):
        status, headers, body = self.request("/tllama/memory")
        assert status == 200, body
        assert headers["Cache-Control"] == "no-store"
        validate(body)
        return body

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.log.close()


@contextlib.contextmanager
def worker(*args):
    instance = Worker(*args)
    try:
        instance.ready()
        yield instance
    finally:
        instance.close()


def validate(report):
    required = {"schema_version", "runtime_fingerprint", "scope", "precision",
                           "status", "missing_components", "model_state", "snapshot_sequence",
                           "snapshot_stage", "snapshot_age_ms", "owners", "devices", "buffers", "placement", "allocation_locality"}
    assert required <= set(report) <= required | {"weight_storage"}
    assert type(report["schema_version"]) is int and report["schema_version"] == 1
    assert report["scope"] == "backend_buffers" and report["precision"] == "exact_bytes"
    assert report["runtime_fingerprint"].startswith("b")
    assert report["status"] in ("known", "partial", "unavailable")
    assert type(report["snapshot_sequence"]) is int and report["snapshot_sequence"] > 0
    assert type(report["snapshot_age_ms"]) is int and report["snapshot_age_ms"] >= 0
    owners = {item["id"] for item in report["owners"]}
    devices = {item["id"] for item in report["devices"]}
    buffers = {item["id"] for item in report["buffers"]}
    assert len(owners) == len(report["owners"])
    assert len(devices) == len(report["devices"])
    assert len(buffers) == len(report["buffers"])
    for item in report["buffers"]:
        assert set(item) == {"id", "owners", "device_ids", "buffer_type", "memory_kind", "backing", "size_bytes"}
        assert item["memory_kind"] in ("host", "host_pinned", "device", "unknown")
        assert item["backing"] in ("allocated", "mmap", "imported", "unknown")
        assert item["size_bytes"] is None or (type(item["size_bytes"]) is int and item["size_bytes"] >= 0)
        assert set(item["device_ids"]) <= devices
        assert item["owners"]
        assert all(entry["owner_id"] in owners for entry in item["owners"])
    for group in report["placement"]["layer_groups"]:
        assert group["owner_id"] in owners
        assert set(group["device_ids"]) <= devices
        assert group["location"] in ("cpu", "accelerator", "mixed", "unknown")
        assert len(group["layer_indices"]) == len(set(group["layer_indices"]))
        if "assignment" in group:
            evidence = group["assignment"]
            assert set(evidence) == {"status", "device_ids", "sources", "reasons"}
            assert evidence["status"] in ("known", "partial", "unavailable")
            assert set(evidence["device_ids"]) <= devices
            assert set(evidence["sources"]) <= {"loader_selection", "loader_override", "loader_mmap",
                                               "loader_lazy", "native_buffer"}
            assert evidence["status"] != "known" or not evidence["reasons"]

    locality = report["allocation_locality"]
    assert set(locality) == {"status", "scope", "physical_aliasing", "missing_components", "allocations", "bindings"}
    assert locality["status"] in ("known", "partial", "unavailable")
    assert locality["scope"] == "backend_buffer_allocations" and locality["physical_aliasing"] == "unknown"
    allocations = {item["id"] for item in locality["allocations"]}
    assert len(allocations) == len(locality["allocations"])
    for item in locality["allocations"]:
        assert set(item) == {"id", "device_ids", "kind", "origin", "native_api", "size_bytes", "locality", "native_properties"}
        assert set(item["device_ids"]) <= devices
        assert item["kind"] in ("unknown", "device", "host", "host_pinned", "managed", "shared", "system")
        assert item["origin"] in ("unknown", "allocated", "imported", "registered")
        assert item["locality"] in ("unknown", "device_local", "non_device_local")
        assert item["size_bytes"] is None or (type(item["size_bytes"]) is int and item["size_bytes"] >= 0)
    for item in locality["bindings"]:
        assert set(item) == {"buffer_id", "allocation_id", "buffer_offset_bytes", "allocation_offset_bytes", "size_bytes"}
        assert item["buffer_id"] in buffers and item["allocation_id"] in allocations
        assert all(item[key] is None or (type(item[key]) is int and item[key] >= 0)
                   for key in ("buffer_offset_bytes", "allocation_offset_bytes", "size_bytes"))
    for item in locality["missing_components"]:
        assert item["buffer_id"] is None or item["buffer_id"] in buffers

    if "weight_storage" in report:
        weights = report["weight_storage"]
        required_weights = {"scope", "deduplication", "status", "missing_components", "intervals"}
        assert required_weights <= set(weights) <= required_weights | {"tensor_diagnostics"}
        assert weights["scope"] == "model_weights" and weights["deduplication"] == "buffer_ranges"
        assert weights["status"] in ("known", "partial", "unavailable")
        assert (weights["status"] == "known") == (not weights["missing_components"])
        models = {item["id"] for item in report["owners"] if item["kind"] == "model"}
        descriptions = {item["id"]: item for item in report["buffers"]}
        ids, ends = set(), {}
        components = {"input_weights", "output_weights", "repeating_layers", "mtp_layers", "other_weights"}
        for interval in weights["intervals"]:
            assert set(interval) == {"id", "buffer_id", "offset_bytes", "size_bytes", "assignment", "relationships"}
            assert interval["id"] not in ids
            ids.add(interval["id"])
            buffer_id = interval["buffer_id"]
            assert buffer_id in buffers
            offset, size = interval["offset_bytes"], interval["size_bytes"]
            assert type(offset) is int and type(size) is int and offset >= 0 and size > 0
            assert offset + size <= 2**64 - 1
            assert offset >= ends.get(buffer_id, 0)
            ends[buffer_id] = offset + size
            assert descriptions[buffer_id]["size_bytes"] is not None
            assert offset + size <= descriptions[buffer_id]["size_bytes"]
            evidence = interval["assignment"]
            assert set(evidence) == {"status", "device_ids", "sources", "reasons"}
            assert evidence["status"] in ("known", "partial", "unavailable")
            assert set(evidence["device_ids"]) <= devices
            assert evidence["status"] != "known" or (len(evidence["device_ids"]) == 1 and not evidence["reasons"])
            assert interval["relationships"]
            relations = set()
            for relation in interval["relationships"]:
                assert set(relation) == {"owner_id", "component", "layer_index"}
                assert relation["owner_id"] in models and relation["component"] in components
                layer = relation["layer_index"]
                assert layer is None or (type(layer) is int and layer >= 0)
                key = relation["owner_id"], relation["component"], layer
                assert key not in relations
                relations.add(key)
                assert any(item["owner_id"] == relation["owner_id"] and item["component"] == "model_weights"
                           for item in descriptions[buffer_id]["owners"])
        failures = set()
        for missing in weights["missing_components"]:
            assert set(missing) == {"owner_id", "component", "layer_index", "reason"}
            assert missing["owner_id"] is None or missing["owner_id"] in models
            layer = missing["layer_index"]
            assert layer is None or (type(layer) is int and layer >= 0)
            key = missing["owner_id"], missing["component"], layer, missing["reason"]
            assert key not in failures
            failures.add(key)
        assert weights["status"] != "unavailable" or not weights["intervals"]
        if "tensor_diagnostics" in weights:
            diagnostics = weights["tensor_diagnostics"]
            assert set(diagnostics) == {"status", "record_limit", "range_limit", "truncated", "name_encoding", "missing_components", "records"}
            assert diagnostics["status"] in ("known", "partial", "unavailable")
            assert type(diagnostics["record_limit"]) is int and 1 <= diagnostics["record_limit"] <= 4096
            assert diagnostics["range_limit"] == 16384 and type(diagnostics["truncated"]) is bool
            assert diagnostics["name_encoding"] == "byte_escape"
            assert (diagnostics["status"] == "known") == (not diagnostics["missing_components"])
            assert not diagnostics["truncated"] or diagnostics["status"] == "partial"
            assert len(diagnostics["records"]) <= diagnostics["record_limit"]
            record_ids, range_count = set(), 0
            for record in diagnostics["records"]:
                assert set(record) == {"id", "tensor_name", "owner_id", "component", "layer_index", "buffer_id",
                                       "storage_status", "storage_reason", "size_bytes", "ranges_complete", "ranges", "assignment"}
                assert record["id"] not in record_ids
                record_ids.add(record["id"])
                assert record["owner_id"] in models
                assert record["component"] is None or record["component"] in components
                assert record["buffer_id"] is None or record["buffer_id"] in buffers
                name = record["tensor_name"]
                assert name is None or (isinstance(name, str) and len(name) <= 512 and all(32 <= ord(c) < 127 for c in name))
                assert record["storage_status"] in ("known", "unavailable")
                assert type(record["ranges_complete"]) is bool
                assert set(record["assignment"]["device_ids"]) <= devices
                if record["storage_status"] == "unavailable":
                    assert record["size_bytes"] is None and record["storage_reason"]
                    assert record["ranges"] == [] and not record["ranges_complete"]
                else:
                    assert type(record["size_bytes"]) is int and record["size_bytes"] >= 0
                    assert record["storage_reason"] is None
                length, end = 0, 0
                for item in record["ranges"]:
                    assert set(item) == {"offset_bytes", "size_bytes"}
                    offset, size = item["offset_bytes"], item["size_bytes"]
                    assert type(offset) is int and type(size) is int and offset >= end and size > 0
                    end = offset + size
                    assert record["buffer_id"] in buffers and end <= descriptions[record["buffer_id"]]["size_bytes"]
                    length += size
                    range_count += 1
                if record["ranges_complete"]:
                    assert length == record["size_bytes"]
            assert range_count <= diagnostics["range_limit"]


def test_loaded_auth_and_stream(server, model, directory, mmap):
    options = ["--load-mode", "mmap" if mmap else "none"]
    with worker(server, model, directory, *options) as instance:
        for key in (None, "wrong-key"):
            assert instance.request("/tllama/memory", key=key)[0] == 401
        assert instance.request("/tllama/memory", header="X-Api-Key")[0] == 200
        initial = instance.report()
        assert initial["status"] == "known", initial["missing_components"]
        assert initial["model_state"] == "loaded"
        assert initial["snapshot_stage"] == "after_load"
        assert initial["placement"]["status"] == "known", initial["placement"]
        assert initial["weight_storage"]["status"] == "known", initial["weight_storage"]
        assert initial["weight_storage"]["intervals"]
        assert "tensor_diagnostics" not in initial["weight_storage"]
        cpu_devices = {item["id"] for item in initial["devices"] if item["type"] == "cpu"}
        for group in initial["placement"]["layer_groups"]:
            assert group["assignment"]["status"] == "known"
            assert group["assignment"]["device_ids"] and set(group["assignment"]["device_ids"]) <= cpu_devices
            assert group["assignment"]["sources"] == ["loader_selection"]
        assert initial["allocation_locality"]["status"] == "unavailable"  # CPU exporter is optional
        assert initial["allocation_locality"]["allocations"] == []
        assert all(item["memory_kind"] == "host" for item in initial["buffers"])
        weights = [item for item in initial["buffers"] if any(
            owner["component"] == "model_weights" for owner in item["owners"])]
        assert weights and all(item["size_bytes"] > 0 for item in weights)
        assert all(item["backing"] == ("mmap" if mmap else "allocated") for item in weights)

        stream = http.client.HTTPConnection("127.0.0.1", instance.port, timeout=5)
        try:
            stream.request("POST", "/completion", json.dumps({
                "prompt": [1, 2, 3], "stream": True, "n_predict": -1, "ignore_eos": True,
                "cache_prompt": False}), {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            response = stream.getresponse()
            assert response.status == 200
            while True:
                line = response.readline()
                assert line, "stream ended before producing a token"
                if line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    assert not event.get("stop", False), event
                    break
            # The unbounded stream remains open while separate HTTP connections
            # read snapshots. A task-queue-based collector would block here.
            for _ in range(3):
                current = instance.report()
                assert current["snapshot_sequence"] >= initial["snapshot_sequence"]
                assert current["model_state"] == "loaded"
                assert current["weight_storage"] == initial["weight_storage"]
        finally:
            stream.close()
        def completed():
            report = instance.report()
            return report if report["snapshot_stage"] == "after_request" else None
        final = instance.wait(completed)
        assert final["snapshot_sequence"] > initial["snapshot_sequence"]
        assert final["weight_storage"] == initial["weight_storage"]
        print("loaded/auth/stream/after_request", "mmap" if mmap else "allocated", "PASSED")


def test_sleep(server, model, directory):
    with worker(server, model, directory, "--sleep-idle-seconds", "1") as instance:
        before = instance.report()
        def asleep():
            report = instance.report()
            return report if report["model_state"] == "sleeping" else None
        saved = instance.wait(asleep)
        assert saved["status"] == "known" and saved["buffers"] == []
        assert saved["allocation_locality"]["status"] == "known"
        assert saved["allocation_locality"]["allocations"] == saved["allocation_locality"]["bindings"] == []
        assert saved["weight_storage"]["status"] == "known" and saved["weight_storage"]["intervals"] == []
        assert saved["snapshot_stage"] == "after_unload"
        assert saved["snapshot_sequence"] > before["snapshot_sequence"]
        assert instance.report()["snapshot_sequence"] == saved["snapshot_sequence"]
        assert instance.request("/props")[2]["is_sleeping"] is True
        assert instance.request("/completion", data={"prompt": [1, 2, 3], "n_predict": 1})[0] == 200
        after = instance.report()
        assert after["model_state"] == "loaded" and after["buffers"]
        assert after["snapshot_sequence"] > saved["snapshot_sequence"]
        assert after["weight_storage"] == before["weight_storage"]
        print("sleep/read-without-wake/reload PASSED")


def test_stream_connection_reuse(server, model, directory):
    with worker(server, model, directory) as instance:
        connection = http.client.HTTPConnection("127.0.0.1", instance.port, timeout=5)
        try:
            connection.connect()
            connection.auto_open = 0
            original_socket = connection.sock
            headers = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
            for _ in range(2):
                connection.request("POST", "/completion", json.dumps({
                    "prompt": [1, 2, 3], "n_predict": 4, "stream": True,
                    "ignore_eos": True, "cache_prompt": False}), headers)
                with connection.getresponse() as response:
                    assert response.status == 200 and response.chunked and not response.will_close
                    events = [json.loads(line[6:]) for line in response.read().splitlines()
                              if line.startswith(b"data: ")]
                    assert events and events[-1]["stop"] is True
                connection.request("GET", "/tllama/memory", headers=headers)
                with connection.getresponse() as response:
                    assert response.status == 200
                    validate(json.loads(response.read()))
                assert connection.sock is original_socket
        finally:
            connection.close()
        print("stream/Memory API/stream on one HTTP connection PASSED")


def test_tensor_diagnostics(server, model, directory):
    baseline = None
    for limit in (2, 4096):
        with worker(server, model, directory, "--memory-tensor-diagnostics", str(limit), "--sleep-idle-seconds", "1") as instance:
            assert instance.request("/tllama/memory", key=None)[0] == 401
            initial = instance.report()
            weights = initial["weight_storage"]
            diagnostics = weights.pop("tensor_diagnostics")
            assert weights["status"] == "known" and diagnostics["records"]
            if baseline is None:
                baseline = weights
            else:
                assert weights == baseline
            assert diagnostics["truncated"] == (limit == 2)
            assert diagnostics["status"] == ("partial" if limit == 2 else "known")
            saved = instance.wait(lambda: (value if (value := instance.report())["model_state"] == "sleeping" else None))
            empty = saved["weight_storage"]["tensor_diagnostics"]
            assert empty["status"] == "known" and not empty["records"] and not empty["truncated"]
            assert instance.request("/completion", data={"prompt": [1, 2, 3], "n_predict": 1})[0] == 200
            after = instance.report()
            assert after["weight_storage"]["tensor_diagnostics"] == diagnostics
    print("optional bounded tensor diagnostics/sleep/reload PASSED")


def test_router(server, directory):
    models = directory / "empty-models"
    models.mkdir()
    with worker(server, None, directory, "--models-dir", str(models), "--no-models-autoload") as instance:
        assert instance.request("/tllama/memory")[0] == 404
        print("router does not expose worker memory PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--fixture-generator", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tllama-memory-http-") as temp:
        directory = pathlib.Path(temp)
        model = directory / "model.gguf"
        subprocess.run([args.fixture_generator, "--write-model", str(model)], check=True, timeout=15)
        test_loaded_auth_and_stream(args.server, model, directory, mmap=True)
        test_loaded_auth_and_stream(args.server, model, directory, mmap=False)
        test_stream_connection_reuse(args.server, model, directory)
        test_sleep(args.server, model, directory)
        test_tensor_diagnostics(args.server, model, directory)
        test_router(args.server, directory)


if __name__ == "__main__":
    main()
