"""Secure GPT-2 decoding over the Party gadgets (paper, Section 5.4, operators composed from the blocks).

``load_gpt2_weights(n_layers, repo)`` reads the pinned local Hugging Face
snapshot of ``openai-community/gpt2`` at revision :data:`GPT2_REVISION`
(HF_HUB_OFFLINE=1; the snapshot hash is recorded in ``meta["revision"]``)
directly from ``model.safetensors`` as float32 numpy arrays in the Conv1D
layout (d_in, d_out). :func:`locate_snapshot` resolves the pinned commit in
the local cache, so a cache filled by commit hash alone (no ``refs/main``) is
found; :func:`load_gpt2_tokenizer` loads the tokenizer of the same commit.
Nothing in this package touches the network.

``SecureGPT2(party, weights_or_None, meta, n_layers, plan)`` runs prefill of n
tokens and greedy decode steps with an A64-shared KV cache. The client P1 owns
the token ids and feeds its one-hot rows as its own share of the embedding
input (the server share is zero, no traffic); the embedding is a
``linear_wmask`` or ``linear_generic`` call on that one-hot (plan
``embedding``), positions are public (P0 adds wpe[pos] locally), every layer
linear follows plan ``linear``, attention uses matrix triples and
``softmax_rows(k_max = plan["max_k"])``, the MLP uses erf-GELU (contract
``gelu_clip12``, disclosed deviation from GPT-2's gelu_new), LayerNorm uses
the faithful square with a public power-of-two pre-scaling (scale invariant),
the LM head is always ``linear_wmask`` on the tied embedding matrix, and the
token is selected by ``argmax_row(k = plan["argmax_k"])`` and released to P1
only through ``Party.reveal_final`` (kind ``token``). There is no other
release of any model-derived value (``tests/test_transformer.py`` checks the
release log and the channel transcript).

``install_for_plan(plan)`` runs the real one-round install protocol
(``Party.install_wmask``: P1 sends B1, P0 returns E = W - B0 - B1) for every
weight-masked object the plan needs and not yet installed under the current
(model_version, client_key_domain); ``installed_objects()`` lists them with
their versions.

``reference_gpt2_logits`` is the plaintext float64 reference of the declared
numeric functionality (erf-GELU) on PUBLIC weights and PUBLIC ids; it is a
test oracle and DIAGNOSTIC checker, never part of a secure run.

What is tested (tests/test_transformer.py): synthetic small GPT-2 (both plans,
every k) against the float reference (argmax equality and hidden tolerance on
the reconstructed logits in the test harness), install versioning (a second
install under a new model version produces a fresh E), a 1-layer real GPT-2
prefill of 4 tokens reproducing the float32 argmax token (skipped when the
pinned snapshot is not in the local cache), and the absence of any release
other than the final token. Design limits: the numeric contract bounds
(|activation| < 2^15 real for compared values, LN variance in the rsqrt range
after pre-scaling) are preconditions established by the model-specific
contract, not checked at run time on secret values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import erf

from ..errors import ContractViolation, MaterialError
from ..lowlevel import ring

GPT2_REPO = "openai-community/gpt2"
# the one commit of the repository every measured record and tokenizer call refers to
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GELU_CONTRACT = "gelu_clip12"
LINEAR_SITES = ("qkv", "out", "fc", "proj")
PLAN_KEYS = {"embedding": ("wmask", "generic"), "linear": ("wmask", "generic"),
             "argmax_k": (2, 4, 8), "max_k": (2, 4)}
DEFAULT_PLAN = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}

_LOAD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# weight loading (local HF cache only)
# ---------------------------------------------------------------------------
def _sha256_arrays(arrs: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(arrs[k], dtype=np.float32).tobytes())
    return h.hexdigest()


def locate_snapshot(repo: str, filename: str = "model.safetensors", revision: str = GPT2_REVISION) -> Tuple[str, str]:
    """(snapshot_dir, revision) of a cached HF repo at the pinned commit; never touches the network.

    The lookup is by commit hash, so a cache that holds only
    ``snapshots/<revision>/`` (no ``refs/main``) is found; another revision of
    the same repository is not accepted (``MODEL_CACHE``).
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as exc:  # pragma: no cover
        raise MaterialError(f"MODEL_CACHE: huggingface_hub unavailable ({exc})")
    path = try_to_load_from_cache(repo, filename, revision=revision)
    if not isinstance(path, str) or not os.path.isfile(path):
        raise MaterialError(f"MODEL_CACHE: {repo}/{filename} at revision {revision} is not in the local Hugging Face "
                            f"cache (download that revision with huggingface_hub.snapshot_download first)")
    snap = os.path.dirname(path)
    return snap, os.path.basename(snap)


TOKENIZER_FILES: Tuple[str, ...] = ("tokenizer.json", "vocab.json", "merges.txt")


def load_gpt2_tokenizer(repo: str = GPT2_REPO, revision: str = GPT2_REVISION) -> Any:
    """The GPT-2 tokenizer of the pinned snapshot, read from its local snapshot directory (never the network).

    Used by every tokenizer call of the package (the session runner's prompt
    ids, the sidecar's ``gpt2`` tokenizer). The files are located with
    :func:`locate_snapshot` at the commit hash, so a cache filled by commit
    hash alone is found. A missing file or a tokenizer whose vocabulary does
    not have the size of ``config.json`` raises ``MODEL_CACHE``
    (``transformers`` would otherwise build an empty tokenizer without an
    error); ImportError of ``transformers`` propagates.
    """
    snap = ""
    for name in TOKENIZER_FILES + ("config.json",):
        snap, _ = locate_snapshot(repo, name, revision)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(snap)
    with open(os.path.join(snap, "config.json"), "r", encoding="utf-8") as fh:
        vocab = int(json.load(fh).get("vocab_size", 0))
    if len(tok) != vocab:
        raise MaterialError(f"MODEL_CACHE: the tokenizer of {repo} at revision {revision} has {len(tok)} entries, "
                            f"config.json declares {vocab}")
    return tok


def validate_plan(plan: dict) -> dict:
    out = dict(DEFAULT_PLAN)
    out.update(plan or {})
    for key, allowed in PLAN_KEYS.items():
        if out.get(key) not in allowed:
            raise ContractViolation(f"PLAN_INVALID: {key} must be one of {allowed}, got {out.get(key)!r}")
    extra = set(out) - set(PLAN_KEYS)
    if extra:
        raise ContractViolation(f"PLAN_INVALID: unknown plan keys {sorted(extra)}")
    return out


def load_gpt2_weights(n_layers: int = 12, repo: str = GPT2_REPO) -> Tuple[Dict[str, np.ndarray], dict]:
    """GPT-2 weights (float32 numpy, Conv1D layout (d_in, d_out)) from the pinned local snapshot.

    Returns (weights, meta); meta records model, revision (snapshot hash),
    weight sha256, shapes, epsilon and the activation disclosure.
    """
    n_layers = int(n_layers)
    with _LOAD_LOCK:
        snap, revision = locate_snapshot(repo, "model.safetensors")
        with open(os.path.join(snap, "config.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        from safetensors.numpy import load_file
        sd = load_file(os.path.join(snap, "model.safetensors"))
    if n_layers < 1 or n_layers > int(cfg["n_layer"]):
        raise ContractViolation(f"MODEL_LAYERS: n_layers must be in [1, {cfg['n_layer']}], got {n_layers}")

    def get(name: str) -> np.ndarray:
        for pre in ("", "transformer."):
            if pre + name in sd:
                return np.ascontiguousarray(sd[pre + name], dtype=np.float32)
        raise MaterialError(f"MODEL_KEY: {name!r} missing from the GPT-2 checkpoint")

    W: Dict[str, np.ndarray] = {"wte": get("wte.weight"), "wpe": get("wpe.weight"),
                                "ln_f.g": get("ln_f.weight"), "ln_f.b": get("ln_f.bias")}
    for li in range(n_layers):
        p = f"h.{li}."
        W[f"h{li}.ln_1.g"] = get(p + "ln_1.weight")
        W[f"h{li}.ln_1.b"] = get(p + "ln_1.bias")
        W[f"h{li}.ln_2.g"] = get(p + "ln_2.weight")
        W[f"h{li}.ln_2.b"] = get(p + "ln_2.bias")
        W[f"h{li}.qkv.w"] = get(p + "attn.c_attn.weight")
        W[f"h{li}.qkv.b"] = get(p + "attn.c_attn.bias")
        W[f"h{li}.out.w"] = get(p + "attn.c_proj.weight")
        W[f"h{li}.out.b"] = get(p + "attn.c_proj.bias")
        W[f"h{li}.fc.w"] = get(p + "mlp.c_fc.weight")
        W[f"h{li}.fc.b"] = get(p + "mlp.c_fc.bias")
        W[f"h{li}.proj.w"] = get(p + "mlp.c_proj.weight")
        W[f"h{li}.proj.b"] = get(p + "mlp.c_proj.bias")
    meta = {"model": repo, "revision": revision, "weight_sha256": _sha256_arrays(W),
            "d_model": int(cfg["n_embd"]), "n_head": int(cfg["n_head"]), "d_ff": 4 * int(cfg["n_embd"]),
            "vocab": int(cfg["vocab_size"]), "ln_eps": float(cfg["layer_norm_epsilon"]),
            "n_layers_total": int(cfg["n_layer"]), "n_layers": n_layers, "n_ctx": int(cfg["n_positions"]),
            "activation_model": str(cfg.get("activation_function", "gelu_new")),
            "activation_contract": "erf_gelu", "source": "model.safetensors (local cache, HF_HUB_OFFLINE=1)"}
    return W, meta


# ---------------------------------------------------------------------------
# plaintext float references (public weights, public ids)
# ---------------------------------------------------------------------------
def gelu_erf(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _attention_ref(x: np.ndarray, Wqkv: np.ndarray, bqkv: np.ndarray, Wo: np.ndarray, bo: np.ndarray,
                   H: int, mask: Optional[np.ndarray]) -> np.ndarray:
    n, d = x.shape
    dh = d // H
    qkv = x @ Wqkv + bqkv
    q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(n, H, dh).transpose(1, 0, 2) for i in range(3))
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
    if mask is not None:
        scores = np.where(mask[None], scores, -np.inf)
    p = _softmax(scores)
    o = (p @ v).transpose(1, 0, 2).reshape(n, d)
    return o @ Wo + bo


def reference_gpt2_logits(W: Dict[str, np.ndarray], meta: dict, n_layers: int, ids: List[int],
                          activation: str = "erf_gelu") -> np.ndarray:
    """Float64 GPT-2 logits (n, V) of the declared functionality (erf-GELU) or of gelu_new."""
    act = gelu_erf if activation == "erf_gelu" else gelu_new
    ids = np.asarray(ids, dtype=np.int64)
    n = ids.size
    H = int(meta["n_head"])
    eps = float(meta["ln_eps"])
    x = W["wte"][ids].astype(np.float64) + W["wpe"][np.arange(n)].astype(np.float64)
    mask = np.tril(np.ones((n, n), bool))
    for li in range(int(n_layers)):
        g = lambda k: W[f"h{li}.{k}"].astype(np.float64)
        h = _ln(x, g("ln_1.g"), g("ln_1.b"), eps)
        x = x + _attention_ref(h, g("qkv.w"), g("qkv.b"), g("out.w"), g("out.b"), H, mask)
        h = _ln(x, g("ln_2.g"), g("ln_2.b"), eps)
        f = act(h @ g("fc.w") + g("fc.b"))
        x = x + f @ g("proj.w") + g("proj.b")
    h = _ln(x, W["ln_f.g"].astype(np.float64), W["ln_f.b"].astype(np.float64), eps)
    return h @ W["wte"].astype(np.float64).T


# ---------------------------------------------------------------------------
# secure transformer core
# ---------------------------------------------------------------------------
class _SecureTransformer:
    """Machinery shared by the decoder's phases (plaintext weights live on P0 only)."""

    def __init__(self, party: Any, weights_or_None: Optional[Dict[str, np.ndarray]], meta: dict,
                 n_layers: int, plan: Optional[dict], ln_prescale_bits: int) -> None:
        self.p = party
        if party.party == 1 and weights_or_None is not None:
            raise ContractViolation("MODEL_OWNER: the client P1 must not hold plaintext weights")
        if party.party == 0 and weights_or_None is None:
            raise ContractViolation("MODEL_OWNER: the model owner P0 must supply the weights")
        self.W = weights_or_None
        self.meta = dict(meta)
        self.n_layers = int(n_layers)
        self.d = int(meta["d_model"])
        self.H = int(meta["n_head"])
        self.dh = self.d // self.H
        self.d_ff = int(meta["d_ff"])
        self.V = int(meta["vocab"])
        self.eps = float(meta["ln_eps"])
        self.ln_k = int(ln_prescale_bits)
        self.plan = validate_plan(plan or {})
        self.kv: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * self.n_layers
        if getattr(party, "model_version", "unversioned") == "unversioned":
            party.model_version = f"{meta.get('model', 'model')}@{meta.get('revision', 'unpinned')}"

    # -- plan / install -------------------------------------------------------------
    def set_plan(self, plan: dict) -> None:
        """Switch realizations at a token boundary (the session runtime decides when)."""
        self.plan = validate_plan(plan)

    def required_objects(self, plan: Optional[dict] = None) -> List[str]:
        raise NotImplementedError

    def _W_enc(self, mat_id: str) -> Optional[np.ndarray]:
        if self.p.party != 0:
            return None
        if mat_id == "wte":
            return self.p.enc(self.W["wte"])
        if mat_id == "lm_head":
            return self.p.enc(np.ascontiguousarray(self.W["wte"].T))
        return self.p.enc(self.W[mat_id + ".w"])

    def install_for_plan(self, plan: Optional[dict] = None) -> dict:
        """Install (real protocol) every wmask object the plan needs; returns the cost record."""
        plan = validate_plan(plan) if plan is not None else self.plan
        rec = {"objects": [], "skipped": [], "bytes": 0, "seconds": 0.0, "rounds": 0}
        for mat_id in self.required_objects(plan):
            r = self.p.install_wmask(mat_id, self._W_enc(mat_id))
            if r.get("skipped"):
                rec["skipped"].append(mat_id)
            else:
                rec["objects"].append(mat_id)
                rec["bytes"] += int(r["bytes"])
                rec["seconds"] += float(r["seconds"])
                rec["rounds"] += int(r["rounds"])
        return rec

    def installed_objects(self) -> dict:
        return self.p.installed_objects()

    # -- operators --------------------------------------------------------------------
    def linear(self, x: np.ndarray, mat_id: str, realization: str, trunc: bool = True) -> np.ndarray:
        if realization == "wmask":
            return self.p.linear_wmask(x, mat_id, trunc=trunc)
        if realization == "generic":
            return self.p.linear_generic(x, mat_id, trunc=trunc)
        raise ContractViolation(f"PLAN_INVALID: unknown realization {realization!r}")

    def layernorm(self, x: np.ndarray, name: str) -> np.ndarray:
        """LayerNorm with public power-of-two pre-scaling (scale invariant; keeps var in range)."""
        g = self.p.vector_share(name + ".g")
        b = self.p.vector_share(name + ".b")
        xs = self.p.trunc_local(x, self.ln_k) if self.ln_k else x
        return self.p.layernorm_rows(xs, g, b, eps=self.eps / float(4 ** self.ln_k))

    def attention(self, li: int, x: np.ndarray, key_mask: Optional[np.ndarray], causal: bool,
                  cache: bool) -> np.ndarray:
        n = x.shape[0]
        qkv = self.linear(x, f"h{li}.qkv", self.plan["linear"])
        q, k, v = (qkv[:, i * self.d:(i + 1) * self.d] for i in range(3))
        q = np.ascontiguousarray(q.reshape(n, self.H, self.dh).transpose(1, 0, 2))
        k = np.ascontiguousarray(k.reshape(n, self.H, self.dh).transpose(1, 0, 2))
        v = np.ascontiguousarray(v.reshape(n, self.H, self.dh).transpose(1, 0, 2))
        if cache and self.kv[li] is not None:
            K = np.concatenate([self.kv[li][0], k], axis=1)
            Vv = np.concatenate([self.kv[li][1], v], axis=1)
        else:
            K, Vv = k, v
        if cache:
            self.kv[li] = (K, Vv)
        T = K.shape[1]
        scores = self.p.matmul_shared(q, np.ascontiguousarray(K.transpose(0, 2, 1)))
        scores = self.p.mul_public_real(scores, 1.0 / math.sqrt(self.dh))
        mask = None
        if causal and n > 1:
            mask = np.tril(np.ones((n, T), bool), k=T - n)[None]
        if key_mask is not None:
            km = np.asarray(key_mask, bool)[None, None, :]
            mask = km if mask is None else (mask & km)
        probs = self.p.softmax_rows(scores, mask, k_max=self.plan["max_k"])
        o = self.p.matmul_shared(probs, Vv)
        o = np.ascontiguousarray(o.transpose(1, 0, 2).reshape(n, self.d))
        return self.linear(o, f"h{li}.out", self.plan["linear"])

    def mlp(self, li: int, h: np.ndarray) -> np.ndarray:
        f = self.linear(h, f"h{li}.fc", self.plan["linear"])
        f = self.p.clip_poly_act(f, GELU_CONTRACT)
        return self.linear(f, f"h{li}.proj", self.plan["linear"])

    def onehot_share(self, token_ids: Optional[List[int]], n: int) -> np.ndarray:
        """Client one-hot rows as P1's share (P0 share zero): no communication."""
        e = np.zeros((n, self.V), np.uint64)
        if self.p.party == 1:
            if token_ids is None or len(token_ids) != n:
                raise ContractViolation(f"PROMPT_SHAPE: the client must supply {n} token ids")
            for i, t in enumerate(token_ids):
                t = int(t)
                if not (0 <= t < self.V):
                    raise ContractViolation(f"PROMPT_RANGE: token id {t} outside [0, {self.V})")
                e[i, t] = np.uint64(1)
        elif token_ids is not None:
            raise ContractViolation("PROMPT_OWNER: the server P0 must not receive token ids")
        return e

    def stats_by_kind(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for s in self.p.stats:
            d = out.setdefault(s["kind"], {"calls": 0, "seconds": 0.0, "bytes": 0, "rounds": 0, "material_bytes": 0})
            d["calls"] += 1
            d["seconds"] += float(s["seconds"])
            d["bytes"] += int(s["bytes"])
            d["rounds"] += int(s["rounds"])
            d["material_bytes"] += int(s["material_bytes"])
        return out


class SecureGPT2(_SecureTransformer):
    """Pre-LN decoder with A64-shared KV cache and plan-selected realizations (see module docstring)."""

    def __init__(self, party: Any, weights_or_None: Optional[Dict[str, np.ndarray]], meta: dict,
                 n_layers: int, plan: Optional[dict] = None, ln_prescale_bits: int = 3) -> None:
        super().__init__(party, weights_or_None, meta, n_layers, plan, ln_prescale_bits)
        self.generated: List[int] = []

    def required_objects(self, plan: Optional[dict] = None) -> List[str]:
        plan = self.plan if plan is None else validate_plan(plan)
        objs: List[str] = []
        if plan["embedding"] == "wmask":
            objs.append("wte")
        if plan["linear"] == "wmask":
            for li in range(self.n_layers):
                objs.extend(f"h{li}.{site}" for site in LINEAR_SITES)
        objs.append("lm_head")      # the LM head is weight-masked under every plan
        return objs

    def reset_cache(self) -> None:
        self.kv = [None] * self.n_layers
        self.generated = []

    def embed(self, token_ids: Optional[List[int]], positions: List[int]) -> np.ndarray:
        n = len(positions)
        e = self.onehot_share(token_ids, n)
        x = self.linear(e, "wte", self.plan["embedding"], trunc=False)
        if self.p.party == 0:
            x = ring.add(x, self.p.enc(self.W["wpe"][np.asarray(positions, dtype=np.int64)]))
        return x

    def layer(self, li: int, x: np.ndarray) -> np.ndarray:
        h = self.layernorm(x, f"h{li}.ln_1")
        x = ring.add(x, self.attention(li, h, None, causal=True, cache=True))
        h = self.layernorm(x, f"h{li}.ln_2")
        return ring.add(x, self.mlp(li, h))

    def logits_shares(self, x_last: np.ndarray) -> np.ndarray:
        """Shares of the LM-head logits (1, V) of the last row: final LN -> tied wmask head. No release."""
        h = self.layernorm(x_last, "ln_f")
        return self.p.linear_wmask(h, "lm_head")

    def head(self, x_last: np.ndarray) -> Optional[np.ndarray]:
        """Final LN -> tied LM head (wmask) -> k-ary argmax released to P1 (kind token)."""
        logits = self.logits_shares(x_last)
        return self.p.argmax_row(logits[0], k=self.plan["argmax_k"], kind="token")

    def prefill_logits_shares(self, token_ids_or_None: Optional[List[int]], n: int) -> np.ndarray:
        """DIAGNOSTIC helper for public-model tests: this party's SHARE of the last-position logits.

        Runs the same prefill circuit as :meth:`prefill` but stops before the
        argmax and releases nothing; a test reconstructs the two shares to
        compare against the float reference. Never called by a secure session.
        """
        self.reset_cache()
        x = self.embed(token_ids_or_None, list(range(int(n))))
        for li in range(self.n_layers):
            x = self.layer(li, x)
        return self.logits_shares(x[-1:])

    def prefill(self, token_ids_or_None: Optional[List[int]], n: int) -> Optional[np.ndarray]:
        """Prefill n tokens; returns the first released token on P1 (None on P0)."""
        self.reset_cache()
        x = self.embed(token_ids_or_None, list(range(int(n))))
        for li in range(self.n_layers):
            x = self.layer(li, x)
        tok = self.head(x[-1:])
        if tok is not None:
            self.generated.append(int(tok))
        return tok

    def decode_step(self, token_id_or_None: Optional[int], position: int) -> Optional[int]:
        """One greedy decode step at ``position``; returns the released token on P1 (None on P0)."""
        ids = None if token_id_or_None is None else [int(token_id_or_None)]
        x = self.embed(ids, [int(position)])
        for li in range(self.n_layers):
            x = self.layer(li, x)
        tok = self.head(x[-1:])
        if tok is None:
            return None
        t = int(tok)
        self.generated.append(t)
        return t


__all__ = ["load_gpt2_weights", "load_gpt2_tokenizer", "locate_snapshot", "validate_plan", "SecureGPT2",
           "reference_gpt2_logits", "gelu_erf", "gelu_new", "GPT2_REPO", "GPT2_REVISION", "DEFAULT_PLAN", "PLAN_KEYS",
           "LINEAR_SITES"]
