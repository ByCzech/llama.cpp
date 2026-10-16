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
    assert set(report) == {"schema_version", "runtime_fingerprint", "scope", "precision",
                           "status", "missing_components", "model_state", "snapshot_sequence",
                           "snapshot_stage", "snapshot_age_ms", "owners", "devices", "buffers", "placement"}
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
        cpu_devices = {item["id"] for item in initial["devices"] if item["type"] == "cpu"}
        for group in initial["placement"]["layer_groups"]:
            assert group["assignment"]["status"] == "known"
            assert group["assignment"]["device_ids"] and set(group["assignment"]["device_ids"]) <= cpu_devices
            assert group["assignment"]["sources"] == ["loader_selection"]
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
        finally:
            stream.close()
        def completed():
            report = instance.report()
            return report if report["snapshot_stage"] == "after_request" else None
        final = instance.wait(completed)
        assert final["snapshot_sequence"] > initial["snapshot_sequence"]
        print("loaded/auth/stream/after_request", "mmap" if mmap else "allocated", "PASSED")


def test_sleep(server, model, directory):
    with worker(server, model, directory, "--sleep-idle-seconds", "1") as instance:
        before = instance.report()
        def asleep():
            report = instance.report()
            return report if report["model_state"] == "sleeping" else None
        saved = instance.wait(asleep)
        assert saved["status"] == "known" and saved["buffers"] == []
        assert saved["snapshot_stage"] == "after_unload"
        assert saved["snapshot_sequence"] > before["snapshot_sequence"]
        assert instance.report()["snapshot_sequence"] == saved["snapshot_sequence"]
        assert instance.request("/props")[2]["is_sleeping"] is True
        assert instance.request("/completion", data={"prompt": [1, 2, 3], "n_predict": 1})[0] == 200
        after = instance.report()
        assert after["model_state"] == "loaded" and after["buffers"]
        assert after["snapshot_sequence"] > saved["snapshot_sequence"]
        print("sleep/read-without-wake/reload PASSED")


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
        test_sleep(args.server, model, directory)
        test_router(args.server, directory)


if __name__ == "__main__":
    main()
