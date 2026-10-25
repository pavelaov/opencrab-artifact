"""Transformer tests: SecureGPT2 against the float reference, install versioning, releases.

Synthetic small models exercise both realizations of every linear site and
every tournament arity cheaply; the real openai-community/gpt2 snapshot at the
pinned revision (1 layer, 4-token prefill) checks the float argmax token and is
skipped when that snapshot is not in the local cache. A scratch cache that
holds only ``snapshots/<GPT2_REVISION>/`` (no ``refs/main``) checks that the
weight lookup and the tokenizer loads find the pinned commit. All runs use the
in-process harness of tests/test_party.py (INSECURE_TEST_ONLY_SHARED_SEED, DIAGNOSTIC).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from opencrab2.errors import ContractViolation, MaterialError
from opencrab2.highlevel import transformer as tf
from opencrab2.highlevel.party import Party
from opencrab2.highlevel.transformer import SecureGPT2, load_gpt2_weights, reference_gpt2_logits, validate_plan
from opencrab2.lowlevel import ring
from tests.test_party import TwoPartyHarness, dec, enc


# ---------------------------------------------------------------------------
# synthetic models
# ---------------------------------------------------------------------------
def synthetic_gpt2(seed: int, d: int = 32, H: int = 2, d_ff: int = 64, V: int = 50, n_ctx: int = 16,
                   n_layers: int = 2) -> Tuple[Dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    W = {"wte": rng.normal(0, 0.5, size=(V, d)).astype(np.float32),
         "wpe": rng.normal(0, 0.1, size=(n_ctx, d)).astype(np.float32),
         "ln_f.g": rng.uniform(0.8, 1.2, size=d).astype(np.float32),
         "ln_f.b": rng.normal(0, 0.1, size=d).astype(np.float32)}
    for li in range(n_layers):
        W[f"h{li}.ln_1.g"] = rng.uniform(0.8, 1.2, size=d).astype(np.float32)
        W[f"h{li}.ln_1.b"] = rng.normal(0, 0.1, size=d).astype(np.float32)
        W[f"h{li}.ln_2.g"] = rng.uniform(0.8, 1.2, size=d).astype(np.float32)
        W[f"h{li}.ln_2.b"] = rng.normal(0, 0.1, size=d).astype(np.float32)
        W[f"h{li}.qkv.w"] = rng.normal(0, 0.15, size=(d, 3 * d)).astype(np.float32)
        W[f"h{li}.qkv.b"] = rng.normal(0, 0.05, size=3 * d).astype(np.float32)
        W[f"h{li}.out.w"] = rng.normal(0, 0.15, size=(d, d)).astype(np.float32)
        W[f"h{li}.out.b"] = rng.normal(0, 0.05, size=d).astype(np.float32)
        W[f"h{li}.fc.w"] = rng.normal(0, 0.2, size=(d, d_ff)).astype(np.float32)
        W[f"h{li}.fc.b"] = rng.normal(0, 0.05, size=d_ff).astype(np.float32)
        W[f"h{li}.proj.w"] = rng.normal(0, 0.15, size=(d_ff, d)).astype(np.float32)
        W[f"h{li}.proj.b"] = rng.normal(0, 0.05, size=d).astype(np.float32)
    meta = {"model": f"synthetic-gpt2-{seed}", "revision": "synthetic", "d_model": d, "n_head": H, "d_ff": d_ff,
            "vocab": V, "ln_eps": 1e-5, "n_layers_total": n_layers, "n_layers": n_layers, "n_ctx": n_ctx}
    return W, meta


def dealer_side(W: Dict[str, np.ndarray], meta: dict, n_layers: int):
    """Encoded server-private vectors / generic weights and wmask shapes the harness dealer may serve."""
    secrets = {}
    shapes = {"wte": tuple(W["wte"].shape)}
    secrets["wte"] = enc(W["wte"])                         # generic embedding realization
    secrets["ln_f.g"] = enc(W["ln_f.g"])
    secrets["ln_f.b"] = enc(W["ln_f.b"])
    shapes["lm_head"] = (W["wte"].shape[1], W["wte"].shape[0])
    for li in range(n_layers):
        for n in ("ln_1.g", "ln_1.b", "ln_2.g", "ln_2.b"):
            secrets[f"h{li}.{n}"] = enc(W[f"h{li}.{n}"])
        for site in tf.LINEAR_SITES:
            secrets[f"h{li}.{site}.b"] = enc(W[f"h{li}.{site}.b"])
            secrets[f"h{li}.{site}"] = enc(W[f"h{li}.{site}.w"])
            shapes[f"h{li}.{site}"] = tuple(W[f"h{li}.{site}.w"].shape)
    return secrets, shapes


def _make_gpt2(h: TwoPartyHarness, W, meta, n_layers, plan):
    return [SecureGPT2(h.parties[p], W if p == 0 else None, meta, n_layers, plan) for p in (0, 1)]


def _pick_prompt(W, meta, n_layers, n, margin=0.3, seed=0):
    """A prompt whose float argmax has a clear margin (so fixed-point noise cannot flip it)."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        ids = rng.integers(0, meta["vocab"], size=n).tolist()
        logits = reference_gpt2_logits(W, meta, n_layers, ids)[-1]
        top = np.sort(logits)[::-1]
        if top[0] - top[1] > margin:
            return ids, logits
    raise AssertionError("no prompt with a clear margin found")


# ---------------------------------------------------------------------------
# plans and installs
# ---------------------------------------------------------------------------
def test_validate_plan():
    assert validate_plan({}) == tf.DEFAULT_PLAN
    assert validate_plan({"argmax_k": 8, "max_k": 4})["argmax_k"] == 8
    for bad in ({"embedding": "he"}, {"linear": "ckks"}, {"argmax_k": 3}, {"max_k": 8}, {"foo": 1}):
        with pytest.raises(ContractViolation):
            validate_plan(bad)


def test_owner_checks():
    W, meta = synthetic_gpt2(1, n_layers=1)
    h = TwoPartyHarness(seed=90)
    try:
        with pytest.raises(ContractViolation):
            SecureGPT2(h.parties[1], W, meta, 1, {})          # client must not hold weights
        with pytest.raises(ContractViolation):
            SecureGPT2(h.parties[0], None, meta, 1, {})       # owner must supply them
    finally:
        h.close()


def test_install_for_plan_versions_and_skips():
    n_layers = 1
    W, meta = synthetic_gpt2(2, n_layers=n_layers)
    secrets, shapes = dealer_side(W, meta, n_layers)
    h = TwoPartyHarness(seed=91, secrets=secrets, mask_shapes=shapes)
    try:
        plan = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
        models = _make_gpt2(h, W, meta, n_layers, plan)

        def fn(p: Party):
            m = models[p.party]
            rec = m.install_for_plan()
            again = m.install_for_plan()
            return rec, again, m.installed_objects()

        (r0, a0, inst0), (r1, a1, inst1) = h.run_same(fn)
        want = ["wte"] + [f"h0.{s}" for s in tf.LINEAR_SITES] + ["lm_head"]
        assert r0["objects"] == want and r1["objects"] == want
        assert r0["rounds"] == len(want) and r0["bytes"] > 0
        assert a0["objects"] == [] and sorted(a0["skipped"]) == sorted(want)
        assert set(inst0) == set(want) and inst0 == inst1
        assert all(v["model_version"] == "model-v1" for v in inst0.values())
        # generic plan needs only the LM head object
        for m in models:
            assert m.required_objects({"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}) == ["lm_head"]
    finally:
        h.close()


# ---------------------------------------------------------------------------
# synthetic GPT-2: both realizations, every k, prefill + decode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("plan", [
    {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2},
    {"embedding": "generic", "linear": "generic", "argmax_k": 4, "max_k": 4},
    {"embedding": "wmask", "linear": "generic", "argmax_k": 8, "max_k": 2},
])
def test_synthetic_gpt2_prefill_matches_float_reference(plan):
    n_layers = 2
    W, meta = synthetic_gpt2(3, n_layers=n_layers)
    n = 5
    ids, ref_logits = _pick_prompt(W, meta, n_layers, n, seed=plan["argmax_k"])
    secrets, shapes = dealer_side(W, meta, n_layers)
    h = TwoPartyHarness(seed=92 + plan["argmax_k"], secrets=secrets, mask_shapes=shapes, release=("token", 4))
    try:
        models = _make_gpt2(h, W, meta, n_layers, plan)

        def fn(p: Party):
            m = models[p.party]
            m.install_for_plan()
            tok_ids = ids if p.party == 1 else None
            logits = m.prefill_logits_shares(tok_ids, n)
            tok = m.prefill(tok_ids, n)
            return logits, tok

        (l0, t0), (l1, t1) = h.run_same(fn)
        logits = dec(ring.add(l0, l1))[0]
        assert np.max(np.abs(logits - ref_logits)) < 0.05
        assert t0 is None and int(t1) == int(np.argmax(ref_logits))
        # exactly one release (the token) and no other one-directional message to P1 besides installs
        st1 = h.channels[1].stats
        assert len(h.parties[1].release_log) == 1
        assert st1.messages_recv - st1.messages_sent == len(h.parties[1].release_log)
        assert h.parties[0].release.count == 1
        kinds = {s["kind"] for s in h.parties[0].all_stats}
        assert f"argmax_k{plan['argmax_k']}" in kinds and f"softmax_rows_k{plan['max_k']}" in kinds
    finally:
        h.close()


def test_synthetic_gpt2_greedy_decode_with_kv_cache():
    n_layers = 1
    W, meta = synthetic_gpt2(4, n_layers=n_layers, V=30, n_ctx=12)
    n = 3
    ids, _ = _pick_prompt(W, meta, n_layers, n, seed=11)
    # float greedy decode of 3 further tokens with a clear margin at every step
    cur = list(ids)
    ref_tokens = []
    for _ in range(3):
        lg = reference_gpt2_logits(W, meta, n_layers, cur)[-1]
        top = np.sort(lg)[::-1]
        if top[0] - top[1] < 0.2:
            pytest.skip("synthetic prompt has a near-tie in the float reference; choose another seed")
        t = int(np.argmax(lg))
        ref_tokens.append(t)
        cur.append(t)
    secrets, shapes = dealer_side(W, meta, n_layers)
    h = TwoPartyHarness(seed=99, secrets=secrets, mask_shapes=shapes, release=("token", 8))
    try:
        plan = {"embedding": "wmask", "linear": "wmask", "argmax_k": 4, "max_k": 2}
        models = _make_gpt2(h, W, meta, n_layers, plan)

        def fn(p: Party):
            m = models[p.party]
            m.install_for_plan()
            first = m.prefill(ids if p.party == 1 else None, n)
            out = [None if first is None else int(first)]
            last = out[0]
            for step in range(2):
                nxt = m.decode_step(last, n + step)
                out.append(nxt)
                last = nxt
            return out, [None if m.kv[0] is None else m.kv[0][0].shape]

        (o0, kv0), (o1, kv1) = h.run_same(fn)
        assert o0 == [None, None, None]
        assert o1 == ref_tokens
        assert kv0 == kv1 == [(meta["n_head"], n + 2, meta["d_model"] // meta["n_head"])]
        assert len(h.parties[1].release_log) == 3
    finally:
        h.close()


# ---------------------------------------------------------------------------
# real GPT-2 (local cache), 1 layer, 4-token prefill
# ---------------------------------------------------------------------------
def _gpt2_available() -> bool:
    try:
        tf.locate_snapshot(tf.GPT2_REPO, "model.safetensors")
        return True
    except MaterialError:
        return False


@pytest.mark.skipif(not _gpt2_available(), reason="openai-community/gpt2 not in the local HF cache")
def test_load_gpt2_weights_records_revision():
    W, meta = load_gpt2_weights(1)
    assert meta["revision"] == tf.GPT2_REVISION and len(meta["revision"]) == 40
    with pytest.raises(MaterialError, match="^MODEL_CACHE"):          # another commit of the repository is refused
        tf.locate_snapshot(tf.GPT2_REPO, "model.safetensors", revision="0" * 40)
    assert meta["model"] == tf.GPT2_REPO and meta["activation_model"] == "gelu_new"
    assert meta["activation_contract"] == "erf_gelu"
    assert W["wte"].shape == (50257, 768) and W["h0.qkv.w"].shape == (768, 2304)
    assert W["wte"].dtype == np.float32 and len(meta["weight_sha256"]) == 64
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    with pytest.raises(ContractViolation):
        load_gpt2_weights(13)


TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def _commit_only_cache(root: Path) -> Path:
    """A Hugging Face cache with ``snapshots/<GPT2_REVISION>/`` of openai-community/gpt2 and no ``refs/``."""
    snap = root / "models--openai-community--gpt2" / "snapshots" / tf.GPT2_REVISION
    snap.mkdir(parents=True)
    try:
        real = Path(tf.locate_snapshot(tf.GPT2_REPO, "tokenizer.json")[0])
    except MaterialError:
        real = None
    for name in TOKENIZER_FILES:
        if real is not None and (real / name).is_file():
            shutil.copyfile(real / name, snap / name)
    if not (snap / "config.json").is_file():
        (snap / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
    return snap


def test_pinned_snapshot_is_found_in_a_cache_filled_by_commit_hash(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants
    snap = _commit_only_cache(tmp_path / "hub")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    assert hub.try_to_load_from_cache(tf.GPT2_REPO, "config.json") is None          # 'main' does not resolve here
    found, revision = tf.locate_snapshot(tf.GPT2_REPO, "config.json")
    assert Path(found) == snap and revision == tf.GPT2_REVISION
    with pytest.raises(MaterialError, match="^MODEL_CACHE"):                           # another commit is refused
        tf.locate_snapshot(tf.GPT2_REPO, "config.json", revision="1" * 40)
    with pytest.raises(MaterialError, match="^MODEL_CACHE"):                           # no weights in this cache
        tf.locate_snapshot(tf.GPT2_REPO, "model.safetensors")
    # a snapshot without the tokenizer files is refused (transformers would build an empty tokenizer silently)
    bare = tmp_path / "bare" / "models--openai-community--gpt2" / "snapshots" / tf.GPT2_REVISION
    bare.mkdir(parents=True)
    (bare / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 50257}), encoding="utf-8")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "bare"))
    with pytest.raises(MaterialError, match="^MODEL_CACHE"):
        tf.load_gpt2_tokenizer()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    if not all((snap / n).is_file() for n in TOKENIZER_FILES):
        pytest.skip("the pinned GPT-2 tokenizer files are not in the local cache; the weight lookup was checked")
    pytest.importorskip("transformers")
    # the runners' tokenizer loads, in a fresh interpreter whose cache is the commit-only one
    pydir = Path(__file__).resolve().parents[1]
    code = ("import json, sys\n"
            f"sys.path.insert(0, {str(pydir)!r})\n"
            "from opencrab2.highlevel import transformer as tf\n"
            "from opencrab2.experiments import run_sessions as rs, workloads\n"
            "tok = tf.load_gpt2_tokenizer()\n"
            "print(json.dumps({'ids': tok.encode(workloads.PROMPT_TEXT), 'runner': rs.prompt_ids(),\n"
            "                  'pinned': list(workloads.PINNED_PROMPT_IDS)}))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME")}
    env.update(HF_HUB_CACHE=str(tmp_path / "hub"), HF_HOME=str(tmp_path / "home"), HF_HUB_OFFLINE="1")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ids"] == out["pinned"] == out["runner"]
    assert not (tmp_path / "hub" / "models--openai-community--gpt2" / "refs").exists()


@pytest.mark.skipif(not _gpt2_available(), reason="openai-community/gpt2 not in the local HF cache")
def test_real_gpt2_one_layer_prefill_reproduces_float_argmax():
    n_layers = 1
    W, meta = load_gpt2_weights(n_layers)
    ids = [464, 3139, 286, 4881]          # "The capital of France" (GPT-2 BPE ids)
    ref = reference_gpt2_logits(W, meta, n_layers, ids)[-1]           # declared functionality (erf-GELU)
    ref_new = reference_gpt2_logits(W, meta, n_layers, ids, activation="gelu_new")[-1]
    top = np.sort(ref)[::-1]
    assert top[0] - top[1] > 0.05, "prompt has a near-tie; the test needs a clear margin"
    secrets, shapes = dealer_side(W, meta, n_layers)
    secrets.pop("wte")                     # wmask embedding plan: no generic copy of wte needed
    for li in range(n_layers):
        for site in tf.LINEAR_SITES:
            secrets.pop(f"h{li}.{site}")   # wmask linear plan
    h = TwoPartyHarness(seed=123, secrets=secrets, mask_shapes=shapes, release=("token", 2))
    try:
        plan = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
        models = _make_gpt2(h, W, meta, n_layers, plan)

        def fn(p: Party):
            m = models[p.party]
            inst = m.install_for_plan()
            tok = m.prefill(ids if p.party == 1 else None, len(ids))
            return tok, inst, m.stats_by_kind()

        (t0, inst0, st0), (t1, inst1, st1) = h.run(fn, fn, timeout_s=1800.0)
        assert t0 is None
        assert int(t1) == int(np.argmax(ref))
        hf_argmax = int(np.argmax(ref_new))
        if hf_argmax == int(np.argmax(ref)):
            assert int(t1) == hf_argmax     # gelu_new (the HF model) and erf-GELU agree on this prompt
        assert len(h.parties[1].release_log) == 1 and h.parties[1].release_log[0]["kind"] == "token"
        stc = h.channels[1].stats
        assert stc.messages_recv - stc.messages_sent == 1          # the token release is the only extra message
        assert inst0["objects"] == ["wte"] + [f"h0.{s}" for s in tf.LINEAR_SITES] + ["lm_head"]
        assert "argmax_k2" in st0 and st0["argmax_k2"]["rounds"] == 2 * 16 + 1   # 2 ceil(log2 50257) + release
    finally:
        h.close()
