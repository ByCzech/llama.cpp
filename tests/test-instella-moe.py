#!/usr/bin/env python3
"""Offline Instella tests: materialized K/V NumPy oracle, loader guards and native HTTP.

Reference equations: AMD Instella-MoE-16B-A3B-Think revision
74d28c1a8425583d7805b58e2caa9c13d6b62929 and Transformers v4.57.1 DeepseekV3.
No remote weights, remote code execution or torch dependency.
"""
import argparse
import ast
import importlib.util
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf

H, HEADS, NOPE, ROT, LATENT, VALUE, VOCAB = 32, 2, 96, 32, 32, 128, 32
LAYERS, EXPERTS, DENSE_FF, EXPERT_FF, SHARED = 3, 4, 64, 32, 2
EPS, THETA = 1e-6, 8000000.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def norm(x, weight):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * weight


def rope(x, yarn):
    # Adjacent pairs, materialized before attention. YaRN mscale == mscale_all_dim == 1
    # gives unit rotary amplitude; its all-dimension factor belongs to attention scaling.
    inv = THETA ** (-np.arange(0, ROT, 2, dtype=np.float64) / ROT)
    if yarn:
        low = max(0, np.floor(ROT * np.log(4096 / (32 * 2 * np.pi)) / (2 * np.log(THETA))))
        high = min(ROT - 1, np.ceil(ROT * np.log(4096 / (2 * np.pi)) / (2 * np.log(THETA))))
        ramp = np.clip((np.arange(ROT // 2) - low) / max(high - low, 0.001), 0, 1)
        inv = inv * (1 - ramp) + inv / 40 * ramp
    angle = np.arange(x.shape[0])[:, None] * inv[None, :]
    while angle.ndim < x.ndim:
        angle = angle[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * np.cos(angle) - x[..., 1::2] * np.sin(angle)
    out[..., 1::2] = x[..., 0::2] * np.sin(angle) + x[..., 1::2] * np.cos(angle)
    return out


def reference(weights, tied=False, yarn=False, variable=False, mutation=None):
    w = {k: v.astype(np.float64) for k, v in weights.items()}
    full = w["token_embd.weight"][np.arange(1, 8)]
    routed_free = full.copy()
    for layer in range(LAYERS):
        def get(name):
            return w[f"blk.{layer}.{name}"]
        x = norm(full if mutation == "single_stream" else routed_free, get("attn_norm.weight"))
        q = (x @ get("attn_q.weight").T).reshape(7, HEADS, NOPE + ROT)
        kv = x @ get("attn_kv_a_mqa.weight").T
        latent = norm(kv[:, :LATENT], get("attn_kv_a_norm.weight"))
        # Deliberately expand K/V rather than absorbing wk_b into Q as the native graph does.
        keys = np.einsum("tl,hln->thn", latent, get("attn_k_b.weight"))
        values = np.einsum("tl,hvl->thv", latent, get("attn_v_b.weight"))
        qrot, krot = rope(q[..., NOPE:], yarn), rope(kv[:, LATENT:], yarn)
        scores = np.einsum("thd,shd->hts", q[..., :NOPE], keys)
        scores += np.einsum("thr,sr->hts", qrot, krot)
        scale = (1 + 0.1 * np.log(40)) ** 2 if yarn else 1
        scores *= scale / np.sqrt(NOPE + ROT)
        scores[:, np.triu_indices(7, 1)[0], np.triu_indices(7, 1)[1]] = -np.inf
        prob = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        prob /= np.sum(prob, axis=-1, keepdims=True)
        attn = np.einsum("hts,shv->thv", prob, values).reshape(7, HEADS * VALUE)
        if mutation != "ungated":
            attn *= sigmoid(x @ get("attn_gate.weight").T)
        attn = attn @ get("attn_output.weight").T
        ffn_input = full + attn if mutation == "sequential" else full
        x = norm(ffn_input, get("ffn_norm.weight"))
        def mlp(gate, up, down):
            g = x @ gate.T
            return (g * sigmoid(g) * (x @ up.T)) @ down.T
        if layer == 0:
            full = full + attn + mlp(get("ffn_gate.weight"), get("ffn_up.weight"), get("ffn_down.weight"))
            routed_free = full.copy()
        else:
            scores = sigmoid(x @ get("ffn_gate_inp.weight").T)
            choice = scores + get("exp_probs_b.bias")
            used = 3 if variable and layer == 2 else 2
            ids = np.argsort(-choice, axis=-1, kind="stable")[:, :used]
            selected = np.take_along_axis(scores, ids, axis=-1)
            selected = selected / selected.sum(axis=-1, keepdims=True) * 2.5
            routed = np.zeros_like(x)
            for expert in range(EXPERTS):
                contribution = mlp(get("ffn_gate_exps.weight")[expert],
                                   get("ffn_up_exps.weight")[expert], get("ffn_down_exps.weight")[expert])
                factor = np.sum(selected * (ids == expert), axis=-1)
                routed += contribution * factor[:, None]
            shared = mlp(get("ffn_gate_shexp.weight"), get("ffn_up_shexp.weight"), get("ffn_down_shexp.weight"))
            routed_free = full + attn + shared
            full = routed_free + routed
    output = w["token_embd.weight"] if tied else w["output.weight"]
    return (norm(full, w["output_norm.weight"]) @ output.T).astype(np.float32)


def fixture(directory, tied=False, yarn=False, variable=False, invalid=None, q8_key=False):
    path = directory / "model.gguf"
    w = gguf.GGUFWriter(path, "instella-moe")
    w.add_context_length(163840 if yarn else 128)
    w.add_embedding_length(H)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(DENSE_FF)
    w.add_head_count(HEADS)
    w.add_head_count_kv(1)
    w.add_key_length(LATENT + ROT + (1 if invalid == "key" else 0))
    w.add_value_length(LATENT)
    w.add_key_length_mla(NOPE + ROT)
    w.add_value_length_mla(VALUE)
    w.add_rope_dimension_count(ROT)
    w.add_rope_freq_base(THETA)
    w.add_layer_norm_rms_eps(EPS)
    w.add_leading_dense_block_count(4 if invalid == "dense" else 1)
    w.add_kv_lora_rank(LATENT)
    if invalid == "query":
        w.add_q_lora_rank(8)
    w.add_expert_count(EXPERTS)
    w.add_expert_used_count([2, 2, 3] if variable else 2)
    w.add_expert_feed_forward_length(EXPERT_FF)
    w.add_expert_shared_count(SHARED)
    w.add_expert_weights_norm(True)
    w.add_expert_weights_scale(2.5)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    if yarn:
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(40)
        w.add_rope_scaling_orig_ctx_len(4096)
        w.add_rope_scaling_yarn_log_mul(0.1)
        w.add_rope_scaling_yarn_beta_fast(32)
        w.add_rope_scaling_yarn_beta_slow(1)
    w.add_tokenizer_model("test")
    w.add_token_list([f"tok_{i}" for i in range(VOCAB)])
    w.add_vocab_size(VOCAB)
    rng = np.random.default_rng(943)
    weights = {}
    def tensor(name, *shape, normal=False):
        value = rng.normal(1 if normal else 0, 0.05, shape).astype(np.float32)
        if q8_key and name.endswith("attn_k_b.weight"):
            quantized = gguf.quants.quantize(value, gguf.GGMLQuantizationType.Q8_0)
            weights[name] = gguf.quants.dequantize(quantized, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            weights[name] = value
            if not (invalid == "gate" and name == "blk.1.attn_gate.weight"):
                w.add_tensor(name, value)
    tensor("token_embd.weight", VOCAB, H)
    tensor("output_norm.weight", H, normal=True)
    if not tied:
        tensor("output.weight", VOCAB, H)
    for layer in range(LAYERS):
        def t(name, *shape, **kwargs):
            tensor(f"blk.{layer}.{name}", *shape, **kwargs)
        t("attn_norm.weight", H, normal=True)
        t("attn_kv_a_norm.weight", LATENT, normal=True)
        t("attn_q.weight", HEADS * (NOPE + ROT), H)
        t("attn_kv_a_mqa.weight", LATENT + ROT, H)
        t("attn_k_b.weight", HEADS, LATENT, NOPE)
        t("attn_v_b.weight", HEADS, VALUE, LATENT)
        t("attn_gate.weight", HEADS * VALUE, H)
        t("attn_output.weight", H, HEADS * VALUE)
        t("ffn_norm.weight", H, normal=True)
        if layer == 0:
            t("ffn_gate.weight", DENSE_FF, H)
            t("ffn_up.weight", DENSE_FF, H)
            t("ffn_down.weight", H, DENSE_FF)
        else:
            t("ffn_gate_inp.weight", EXPERTS, H)
            t("exp_probs_b.bias", EXPERTS)
            t("ffn_gate_exps.weight", EXPERTS, EXPERT_FF, H)
            t("ffn_up_exps.weight", EXPERTS, EXPERT_FF, H)
            t("ffn_down_exps.weight", EXPERTS, H, EXPERT_FF)
            t("ffn_gate_shexp.weight", EXPERT_FF * SHARED, H)
            t("ffn_up_shexp.weight", EXPERT_FF * SHARED, H)
            t("ffn_down_shexp.weight", H, EXPERT_FF * SHARED)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    expected = reference(weights, tied, yarn, variable)
    if not tied and not yarn and not variable and not invalid:
        for mutation in ("single_stream", "ungated", "sequential"):
            delta = np.max(np.abs(reference(weights, mutation=mutation) - expected))
            assert delta > 0.001, (mutation, delta)
    oracle = directory / "reference.bin"
    oracle.write_bytes(struct.pack("<II", 7, VOCAB) + expected.astype("<f4").tobytes())
    return path, oracle


def conversion_guards():
    # Exercise the real Instella guard method without importing the torch converter stack.
    module = ast.parse((ROOT / "conversion/deepseek.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "InstellaMoEModel")
    cls.decorator_list = []
    class Base:
        def set_gguf_parameters(self):
            self.accepted = True
    scope = {"gguf": gguf, "DeepseekV2Model": Base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "conversion/deepseek.py", "exec"), scope)
    model_class = scope["InstellaMoEModel"]
    params = dict(num_hidden_layers=27, farskip=True, gated_attention=True, q_lora_rank=None)
    def check(changes, accept):
        obj = model_class()
        obj.hparams = dict(params, **changes)
        try:
            obj.set_gguf_parameters()
        except NotImplementedError:
            assert not accept, changes
        else:
            assert accept and obj.accepted, changes
    check({}, True)
    for changes in ({"farskip": False}, {"farskip_start_idx": 1}, {"farskip_end_idx": 25},
                    {"attn_only_farskip": True}, {"mlp_only_farskip": True},
                    {"gated_attention": False}, {"q_lora_rank": 8}, {"rope_interleave": False},
                    {"attention_bias": True}, {"hidden_act": "gelu"}, {"moe_layer_freq": 2},
                    {"num_nextn_predict_layers": 1}, {"scoring_func": "softmax"}):
        check(changes, False)
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.INSTELLA_MOE, LAYERS)
    assert names.get_name("model.layers.1.self_attn.gate_proj.weight", try_suffixes=(".weight",)) == "blk.1.attn_gate.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--server", required=True)
    args = parser.parse_args()
    conversion_guards()
    with tempfile.TemporaryDirectory(prefix="instella-test-") as tmp:
        root = pathlib.Path(tmp)
        first = None
        for index, options in enumerate(({}, {"tied": True}, {"yarn": True}, {"variable": True}, {"q8_key": True})):
            directory = root / str(index)
            directory.mkdir()
            model, oracle = fixture(directory, **options)
            subprocess.run([args.runner, str(model), str(oracle)], check=True, timeout=45)
            if first is None:
                first = model
        for invalid in ("key", "dense", "query", "gate"):
            directory = root / invalid
            directory.mkdir()
            model, _ = fixture(directory, invalid=invalid)
            result = subprocess.run([args.server, "-m", str(model), "--device", "none", "--no-warmup"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15, text=True)
            assert result.returncode != 0, invalid
            assert "Instella-MoE:" in result.stdout or "attn_gate.weight" in result.stdout, result.stdout
        result = subprocess.run([args.server, "-m", str(first), "--device", "none", "--split-mode", "tensor"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15, text=True)
        assert result.returncode != 0 and "LLAMA_SPLIT_MODE_TENSOR not implemented" in result.stdout, result.stdout
        spec = importlib.util.spec_from_file_location("memory_http", ROOT / "tests/test-memory-http.py")
        http = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(http)
        with http.worker(args.server, first, root, "--memory-tensor-diagnostics", "4096") as worker:
            assert worker.request("/tllama/memory", key=None)[0] == 401
            report = worker.report()
            assert report["weight_storage"]["status"] == "known", report["weight_storage"]
            records = report["weight_storage"]["tensor_diagnostics"]["records"]
            assert any("attn_gate" in r["tensor_name"] for r in records), records
            status, _, body = worker.request("/completion", data={"prompt": [1, 2, 3], "n_predict": 2, "stream": True}, timeout=15)
            assert status == 200 and "data:" in body, body
            worker.report()
    print("Instella: conversion guards, 40 numeric paths, loader guards and HTTP/SSE passed")


if __name__ == "__main__":
    main()
