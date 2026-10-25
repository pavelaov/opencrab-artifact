"""Public workloads of the measured campaigns: prompts, the tiny byte-vocabulary model, float references.

The launcher side of a measured session (``run_sessions``, ``run_web``,
``scripts/smoke_session.py``) needs three public things that no party module
should own: the pinned prompt, a model small enough for smoke runs, and the
reference greedy continuation that ``output_match`` is checked against.

* :data:`PROMPT_TEXT` / :data:`PINNED_PROMPT_IDS`: the fixed 16-token GPT-2
  prompt from a public-domain text (paper, Section 7.1, models and traces).
* :func:`make_tiny_weights`: a random-weight GPT-2 layout (Conv1D ``(d_in,
  d_out)`` matrices) with a byte vocabulary; the same construction as
  ``tests/test_session.py`` so that tests and smoke runs agree.
* :func:`reference_greedy_tokens`: greedy decode with the float64 reference
  of the declared functionality (erf-GELU, exact softmax and LayerNorm,
  ``transformer.reference_gpt2_logits``, the reference of
  ``tests/test_transformer.py`` and ``tests/test_session.py``). The secure
  path computes the numeric-contract approximation of the same functionality,
  so ``output_match`` compares the released tokens with the tokens the
  declared functionality produces. :func:`reference_margins` reports the
  top-1 minus top-2 logit gap of every step, which says how much numeric
  error a step tolerates before the greedy token can differ.

The reference runs in the launcher process on public test inputs. It is never
sent to P0: the launcher hands the reference tokens to P1 only and P1 writes
the comparison result, not the tokens, to the shared record log.

What is tested (tests/test_run_sessions.py): the tiny model is deterministic
in its seed, the reference of the tiny model equals the one of
tests/test_session.py, and the margins have one entry per generated token.
Design limit: the reference is the float functionality, not a bit-exact
emulation of the fixed-point contract; a step whose margin is below the
contract's error bound may legitimately differ and is then reported as
``output_match = false`` with the margin in the detail, never hidden.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

MODEL_REPO = "openai-community/gpt2"
PROMPT_TEXT = "It was the best of times, it was the worst of times, it was"   # public domain, 16 GPT-2 tokens
PINNED_PROMPT_IDS: Tuple[int, ...] = (1026, 373, 262, 1266, 286, 1661, 11, 340, 373, 262, 5290, 286, 1661, 11, 340, 373)
TINY_PROMPT: Tuple[int, ...] = (72, 101, 108, 108)            # four byte-level tokens ("Hell")
TINY_LENGTH_BUCKET = 8
GPT2_LENGTH_BUCKET = 32


def make_tiny_weights(n_layers: int, d: int = 64, vocab: int = 256, n_head: int = 4, n_ctx: int = 64,
                      seed: int = 7) -> Tuple[Dict[str, np.ndarray], dict]:
    """Random-weight GPT-2 layout with a byte-level vocabulary (identical to tests/test_session.py)."""
    rng = np.random.default_rng(seed)
    d_ff = 4 * d
    W = {"wte": rng.normal(0, 0.4, size=(vocab, d)).astype(np.float32),
         "wpe": rng.normal(0, 0.1, size=(n_ctx, d)).astype(np.float32),
         "ln_f.g": rng.uniform(0.8, 1.2, size=d).astype(np.float32),
         "ln_f.b": rng.normal(0, 0.1, size=d).astype(np.float32)}
    for li in range(n_layers):
        W[f"h{li}.ln_1.g"] = rng.uniform(0.8, 1.2, size=d).astype(np.float32)
        W[f"h{li}.ln_1.b"] = rng.normal(0, 0.1, size=d).astype(np.float32)
        W[f"h{li}.ln_2.g"] = rng.uniform(0.8, 1.2, size=d).astype(np.float32)
        W[f"h{li}.ln_2.b"] = rng.normal(0, 0.1, size=d).astype(np.float32)
        W[f"h{li}.qkv.w"] = rng.normal(0, 0.12, size=(d, 3 * d)).astype(np.float32)
        W[f"h{li}.qkv.b"] = rng.normal(0, 0.05, size=3 * d).astype(np.float32)
        W[f"h{li}.out.w"] = rng.normal(0, 0.12, size=(d, d)).astype(np.float32)
        W[f"h{li}.out.b"] = rng.normal(0, 0.05, size=d).astype(np.float32)
        W[f"h{li}.fc.w"] = rng.normal(0, 0.15, size=(d, d_ff)).astype(np.float32)
        W[f"h{li}.fc.b"] = rng.normal(0, 0.05, size=d_ff).astype(np.float32)
        W[f"h{li}.proj.w"] = rng.normal(0, 0.12, size=(d_ff, d)).astype(np.float32)
        W[f"h{li}.proj.b"] = rng.normal(0, 0.05, size=d).astype(np.float32)
    h = hashlib.sha256()
    for k in sorted(W):
        h.update(k.encode())
        h.update(W[k].tobytes())
    meta = {"model": f"tiny-gpt2-d{d}-v{vocab}-L{n_layers}", "revision": h.hexdigest()[:16],
            "weight_sha256": h.hexdigest(), "d_model": d, "n_head": n_head, "d_ff": d_ff, "vocab": vocab,
            "ln_eps": 1e-5, "n_layers_total": n_layers, "n_layers": n_layers, "n_ctx": n_ctx}
    return W, meta


def reference_greedy_tokens(W: Dict[str, np.ndarray], meta: dict, n_layers: int, prompt_ids: Sequence[int],
                            n_generate: int) -> List[int]:
    """Greedy continuation under the float reference of the declared functionality (erf-GELU)."""
    return [t for t, _m in _reference_steps(W, meta, n_layers, prompt_ids, n_generate)]


def reference_margins(W: Dict[str, np.ndarray], meta: dict, n_layers: int, prompt_ids: Sequence[int],
                      n_generate: int) -> List[float]:
    """Top-1 minus top-2 reference logit of every generated step (real units)."""
    return [m for _t, m in _reference_steps(W, meta, n_layers, prompt_ids, n_generate)]


def _reference_steps(W: Dict[str, np.ndarray], meta: dict, n_layers: int, prompt_ids: Sequence[int],
                     n_generate: int) -> List[Tuple[int, float]]:
    from opencrab2.highlevel.transformer import reference_gpt2_logits
    if int(n_generate) < 1 or not len(prompt_ids):
        raise ValueError("WORKLOAD_PARAMS: the reference needs a non-empty prompt and n_generate >= 1")
    cur = [int(t) for t in prompt_ids]
    out: List[Tuple[int, float]] = []
    for _ in range(int(n_generate)):
        lg = np.asarray(reference_gpt2_logits(W, meta, int(n_layers), cur, "erf_gelu")[-1], dtype=np.float64)
        t = int(np.argmax(lg))
        top2 = np.partition(lg, -2)[-2:]
        out.append((t, float(top2[1] - top2[0])))
        cur.append(t)
    return out


TINY_MIN_MARGIN = 0.15          # three times the 0.05 logit tolerance of tests/test_transformer.py
_TINY_PROMPT_CACHE: Dict[Tuple[int, int, float], Tuple[Tuple[int, ...], float]] = {}


def tiny_prompt(n_layers: int, n_generate: int, min_margin: float = TINY_MIN_MARGIN) -> Tuple[Tuple[int, ...], float]:
    """A public four-byte prompt for the tiny model whose greedy reference has a clear margin at every step.

    Returns ``(prompt_ids, smallest_margin)``. :data:`TINY_PROMPT` is kept whenever all of its ``n_generate``
    reference margins reach ``min_margin`` (true for the one-layer model of the smoke matrix). Otherwise the
    first prompt of a fixed public sequence (numpy generator seeded with 20261010, vocabulary 256) that does is
    taken. Reason: with the four-layer tiny model the first reference margin of TINY_PROMPT is 0.012 logit,
    below the numeric contract's error, so every diag4 cell reported ``output_match = false`` although the
    protocol was correct; a cell is a correctness check only when its prompt leaves the contract room.
    The choice depends on public values only (model seed, layer count, token budget).
    """
    key = (int(n_layers), int(n_generate), float(min_margin))
    hit = _TINY_PROMPT_CACHE.get(key)
    if hit is not None:
        return hit
    W, meta = make_tiny_weights(int(n_layers))
    best: Tuple[Tuple[int, ...], float] = (tuple(TINY_PROMPT), min(reference_margins(W, meta, n_layers, TINY_PROMPT, n_generate)))
    if best[1] < float(min_margin):
        rng = np.random.default_rng(20261010)
        for _ in range(256):
            cand = tuple(int(v) for v in rng.integers(0, 256, size=len(TINY_PROMPT)))
            m = min(reference_margins(W, meta, n_layers, cand, n_generate))
            if m > best[1]:
                best = (cand, m)
            if m >= float(min_margin):
                break
    _TINY_PROMPT_CACHE[key] = best
    return best


def scope_for_layers(n_layers: int, n_layers_total: int = 12) -> str:
    """Evidence label of a GPT-2 run with ``n_layers`` of ``n_layers_total`` layers."""
    n = int(n_layers)
    if n >= int(n_layers_total):
        return "full-model"
    if n in (1, 4):
        return f"prefix-{n}L"
    raise ValueError(f"WORKLOAD_SCOPE_UNSUPPORTED: no evidence label for a {n}-layer prefix")


__all__ = ["MODEL_REPO", "PROMPT_TEXT", "PINNED_PROMPT_IDS", "TINY_PROMPT", "TINY_LENGTH_BUCKET", "GPT2_LENGTH_BUCKET",
           "make_tiny_weights", "reference_greedy_tokens", "reference_margins", "scope_for_layers", "tiny_prompt",
           "TINY_MIN_MARGIN"]
