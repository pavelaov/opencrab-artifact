"""Microbenchmarks of the low-level gates (paper, Section 5, building blocks).

Measures, in one process and without a network, the local work of every
low-level gate over a range of batch sizes and appends one
``MEASURED_NATIVE`` primitive record per (component, batch) to
``raw/primitives.jsonl`` under the data root:

* ``dcf_eval``: one-party evaluation of a batch of 32-bit DCF keys (payload 1).
* ``dcf_gen``: dealer-side generation of the same keys (both parties' keys).
* ``drelu``: one-round DReLU with product payload, both the dealer material
  and the party-side ``drelu_eval`` on the opened value.
* ``trunc_faithful``: faithful truncation gate at (f = 16, L = 40), material
  generation and ``trunc_faithful_eval`` with the signed offset.
* ``ring_to_field``: A64 -> Z_t resharing at t = 68720050177, material and eval.
* ``toy_bridge``: the power-of-two reference BFV scheme (N = 512, t = 2^32):
  keygen, encrypt, coefficient-packed matvec on d_in = 64 by d_out = 8, and
  decrypt; the two-party ``h2a_ref`` is exercised by tests/test_bridge.py and
  not timed here (it needs two parties).
* ``a2h_exact``: TenSEAL BFV (N = 8192, t = 68720050177): context setup,
  ``encrypt_share`` of a full ciphertext of Z_t shares, ``add_share`` on the
  server, and the serialised ciphertext size in bytes (this replaces the
  MODELED constant ``costmodel.BFV_CT_BYTES`` once measured). Skipped with a
  record of status ``DIAGNOSTIC`` carrying the reason when tenseal is missing.

Every timed quantity is the median over ``repeats`` runs of
``time.perf_counter`` deltas; the p95 and minimum are recorded too. Records
carry ``dealer_mode = NONE`` (material is generated in-process, there is no
isolated dealer) and ``security_eligible = False``; they are component
timings, not session evidence.

Usage::

    python -m opencrab2.experiments.run_primitives [--quick] [--out DIR] [--repeats N] [--batches 1024,4096,...]

What is tested (tests/test_schema.py imports the record builders; the full
runner is exercised manually and by ``--quick``): every emitted record
validates against the schema. Design limit: single-thread CPython timings on
this host; they are inputs to the cost model's calibration study, not the
calibration itself (that is run_calibration.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencrab2.experiments import schema
from opencrab2.highlevel.contract import DEFAULT_CONTRACT
from opencrab2.lowlevel import dcf, drelu, ring, ring_to_field, sharing, trunc
from opencrab2.lowlevel.toy_bfv import ToyBFV, ToyParams

DEFAULT_BATCHES: Tuple[int, ...] = (1024, 4096, 16384, 65536)
QUICK_BATCHES: Tuple[int, ...] = (1024, 4096)
DEFAULT_REPEATS = 5
QUICK_REPEATS = 3
SEED = 20261010
T_FIELD = 68720050177
TOY_N = 512
TOY_LOG_T = 32
TOY_D_IN = 64
TOY_D_OUT = 8
BFV_N = 8192
PRIMITIVES: Tuple[str, ...] = ("dcf_gen", "dcf_eval", "drelu", "trunc_faithful", "ring_to_field", "toy_bridge",
                               "a2h_exact")


def _timeit(fn: Callable[[], Any], repeats: int) -> Tuple[List[float], Any]:
    """Run ``fn`` ``repeats`` times; returns (seconds per run, last result)."""
    out: List[float] = []
    res = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        res = fn()
        out.append(time.perf_counter() - t0)
    return out, res


def _summary(secs: Sequence[float], batch: int) -> dict:
    arr = np.asarray(secs, dtype=np.float64)
    return {"repeats": int(arr.size), "seconds": [float(v) for v in arr], "median_s": float(np.median(arr)),
            "p95_s": float(np.percentile(arr, 95)), "min_s": float(arr.min()), "mean_s": float(arr.mean()),
            "per_element_s": float(np.median(arr) / max(1, int(batch)))}


def host_info() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "numpy": np.__version__, "processor": platform.processor()}


def primitive_record(component: str, batch: int, timing: Dict[str, Any], params: dict,
                     status: str = "MEASURED_NATIVE", note: str = "") -> dict:
    """One schema record for a primitive timing (status MEASURED_NATIVE unless the runner says otherwise)."""
    extra = {"component": component, "batch": int(batch), "params": dict(params), "host": host_info(), "note": note}
    extra.update(timing)
    return schema.new_record(
        status=status, record_kind="primitive", workload_scope="primitive", clock_basis="process-perf-counter",
        dealer_mode="NONE", policy_id="none", run_id=f"prim:{component}:{batch}:{schema.utc_now()}",
        T_online_s=float(timing.get("median_s", 0.0)) if "median_s" in timing else None,
        bytes=int(timing.get("bytes", 0)), rounds=int(timing.get("rounds", 0)),
        material_generated=int(timing.get("material_bytes", 0)), material_consumed=int(timing.get("material_bytes", 0)),
        peak_rss_client=schema.peak_rss_bytes(), tokens=0, label=component, extra=extra)


# ---------------------------------------------------------------------------
# individual benchmarks
# ---------------------------------------------------------------------------
def bench_dcf(batch: int, repeats: int, rng: np.random.Generator, n_bits: int = 32) -> List[dict]:
    alpha = rng.integers(0, 2 ** n_bits, size=batch, dtype=np.uint64)
    beta = np.ones((batch, 1), dtype=np.uint64)
    x = rng.integers(0, 2 ** n_bits, size=batch, dtype=np.uint64)
    gen_s, keys = _timeit(lambda: dcf.gen_dcf_batch(alpha, beta, n_bits, rng), repeats)
    k0, k1 = keys
    ev_s, out0 = _timeit(lambda: dcf.eval_dcf_batch(k0, x), repeats)
    out1 = dcf.eval_dcf_batch(k1, x)
    recon = ring.add(out0[:, 0], out1[:, 0])
    want = (x < alpha).astype(np.uint64)
    ok = bool(np.array_equal(recon, want))
    kb = dcf.key_bytes_per_element(n_bits, 1)
    recs = []
    g = _summary(gen_s, batch)
    g.update({"material_bytes": int(2 * kb * batch), "key_bytes_per_element": int(kb), "correct": ok})
    recs.append(primitive_record("dcf_gen", batch, g, {"n_bits": n_bits, "payloads": 1},
                                 note="dealer-side generation of both parties' keys"))
    e = _summary(ev_s, batch)
    e.update({"material_bytes": int(kb * batch), "key_bytes_per_element": int(kb), "correct": ok})
    recs.append(primitive_record("dcf_eval", batch, e, {"n_bits": n_bits, "payloads": 1},
                                 note="one-party evaluation; reconstruction checked against 1{x < alpha}"))
    return recs


def bench_drelu(batch: int, repeats: int, rng: np.random.Generator, n_bits: int = 32) -> List[dict]:
    xr = rng.uniform(-3000, 3000, size=batch)
    x = ring.encode(xr, DEFAULT_CONTRACT)
    gen_s, mats = _timeit(lambda: drelu.drelu_material(batch, n_bits, rng, with_product=True), repeats)
    m0, m1 = mats
    x0, x1 = sharing.share(x, rng)
    y = ring.add(ring.add(x0, m0.r_share), ring.add(x1, m1.r_share))     # the opened value y = x + r
    ev_s, res0 = _timeit(lambda: drelu.drelu_eval(m0, y), repeats)
    res1 = drelu.drelu_eval(m1, y)
    bits = ring.add(res0[0], res1[0])
    ok = bool(np.array_equal(bits, (ring.to_signed(x) >= 0).astype(np.uint64)))
    nb = drelu.drelu_bytes_per_element(n_bits, True)
    s = _summary(ev_s, batch)
    s.update({"gen_seconds": [float(v) for v in gen_s], "gen_median_s": float(np.median(gen_s)),
              "material_bytes": int(nb * batch), "material_bytes_per_element": int(nb), "correct": ok,
              "bytes": int(16 * batch), "rounds": 1})
    return [primitive_record("drelu", batch, s, {"n_bits": n_bits, "with_product": True},
                             note="party-side eval on the opened y; bytes = one 8-byte open per element each way")]


def bench_trunc(batch: int, repeats: int, rng: np.random.Generator, f: int = 16, L: int = 40) -> List[dict]:
    mag = rng.uniform(20, L - 2, size=batch)
    vals = np.sign(rng.uniform(-1, 1, size=batch)) * np.floor(2.0 ** mag)
    x = vals.astype(np.int64).astype(np.uint64)
    gen_s, mats = _timeit(lambda: trunc.trunc_material(batch, f, L, rng), repeats)
    m0, m1 = mats
    x0, x1 = sharing.share(x, rng)
    y = ring.add(ring.add(trunc.trunc_offset(x0, 0, L), m0.r_share), ring.add(trunc.trunc_offset(x1, 1, L), m1.r_share))
    ev_s, z0 = _timeit(lambda: trunc.trunc_faithful_eval(m0, y, 0), repeats)
    z1 = trunc.trunc_faithful_eval(m1, y, 1)
    z = ring.to_signed(ring.add(z0, z1))
    want = np.floor_divide(ring.to_signed(x), 2 ** f)
    ok = bool(np.array_equal(z, want))
    nb = trunc.trunc_bytes_per_element(f)
    s = _summary(ev_s, batch)
    s.update({"gen_seconds": [float(v) for v in gen_s], "gen_median_s": float(np.median(gen_s)),
              "material_bytes": int(nb * batch), "material_bytes_per_element": int(nb), "correct": ok,
              "bytes": int(16 * batch), "rounds": 1})
    return [primitive_record("trunc_faithful", batch, s, {"f": f, "L": L},
                             note="exact floor(x / 2^f) checked on signed magnitudes 2^20 .. 2^(L-2)")]


def bench_r2f(batch: int, repeats: int, rng: np.random.Generator, t: int = T_FIELD) -> List[dict]:
    x = rng.integers(-(2 ** 39), 2 ** 39, size=batch, dtype=np.int64).astype(np.uint64)
    gen_s, mats = _timeit(lambda: ring_to_field.r2f_material(batch, t, rng), repeats)
    m0, m1 = mats
    x0, x1 = sharing.share(x, rng)
    y = ring.add(ring.add(x0, m0.r_share), ring.add(x1, m1.r_share))
    ev_s, z0 = _timeit(lambda: ring_to_field.r2f_eval(m0, y, 0), repeats)
    z1 = ring_to_field.r2f_eval(m1, y, 1)
    z = (z0.astype(object) + z1.astype(object)) % t
    want = ring_to_field.signed_mod(x, t)
    ok = bool(np.array_equal(np.asarray(z, dtype=np.uint64), want))
    nb = ring_to_field.r2f_bytes_per_element(t)
    s = _summary(ev_s, batch)
    s.update({"gen_seconds": [float(v) for v in gen_s], "gen_median_s": float(np.median(gen_s)),
              "material_bytes": int(nb * batch), "material_bytes_per_element": int(nb), "correct": ok,
              "bytes": int(16 * batch), "rounds": 1})
    return [primitive_record("ring_to_field", batch, s, {"t": t},
                             note="signed representative mod t checked against the plaintext reference")]


def bench_toy_bridge(repeats: int, rng: np.random.Generator) -> List[dict]:
    params = ToyParams(N=TOY_N, log_t=TOY_LOG_T)
    bfv = ToyBFV(params)
    w_inf = 2 ** 8
    if not bfv.noise_budget_ok(TOY_D_IN, w_inf):
        raise ValueError("PRIMITIVES_TOY_PARAMS: declared toy parameters violate the noise bound")
    W = rng.integers(-w_inf, w_inf + 1, size=(TOY_D_OUT, TOY_D_IN)).astype(np.int64)
    x = rng.integers(0, 2 ** 16, size=TOY_D_IN, dtype=np.uint64)
    kg_s, kp = _timeit(lambda: bfv.keygen(rng), repeats)
    sk, pk = kp
    m = bfv.encode_vector(x)
    enc_s, ct = _timeit(lambda: bfv.encrypt(pk, m, rng), repeats)
    mv_s, res = _timeit(lambda: bfv.matvec_pt(ct, W % params.t), repeats)
    ct_y, positions = res
    dec_s, pt = _timeit(lambda: bfv.decrypt(sk, ct_y), repeats)
    got = pt[np.asarray(positions, dtype=np.int64)]
    want = (W.astype(object) @ x.astype(object)) % params.t
    ok = bool(np.array_equal(got.astype(object), want))
    recs = []
    for name, secs in (("toy_keygen", kg_s), ("toy_encrypt", enc_s), ("toy_matvec_pt", mv_s), ("toy_decrypt", dec_s)):
        s = _summary(secs, TOY_N)
        s.update({"correct": ok, "bytes": int(2 * 8 * TOY_N), "ct_products": 1})
        recs.append(primitive_record("toy_bridge", TOY_N, s, {"N": TOY_N, "log_t": TOY_LOG_T, "d_in": TOY_D_IN,
                                                               "d_out": TOY_D_OUT, "step": name},
                                     note="power-of-two reference scheme, no computational security; "
                                          "h2a_ref (two parties) is covered by tests/test_bridge.py"))
    return recs


def bench_a2h_exact(repeats: int, rng: np.random.Generator) -> List[dict]:
    if importlib.util.find_spec("tenseal") is None:
        s = {"skipped": True, "reason": "ModuleNotFoundError: tenseal"}
        return [primitive_record("a2h_exact", BFV_N, s, {"N": BFV_N, "t": T_FIELD}, status="DIAGNOSTIC",
                                 note="tenseal unavailable; no timing produced")]
    try:
        from opencrab2.highlevel.he_lane import BFVExactA2H
    except Exception as exc:  # the HE lane refused to load: report, do not invent
        s = {"skipped": True, "reason": f"{exc.__class__.__name__}: {exc}"}
        return [primitive_record("a2h_exact", BFV_N, s, {"N": BFV_N, "t": T_FIELD}, status="DIAGNOSTIC",
                                 note="tenseal unavailable; no timing produced")]
    setup_s, he1 = _timeit(lambda: BFVExactA2H(1, poly_modulus_degree=BFV_N, plain_modulus=T_FIELD), repeats)
    pub = he1.public_bytes()
    he0 = BFVExactA2H(0, poly_modulus_degree=BFV_N, plain_modulus=T_FIELD)
    he0.load_public(pub)
    z1 = rng.integers(0, T_FIELD, size=BFV_N, dtype=np.uint64)
    z0 = rng.integers(0, T_FIELD, size=BFV_N, dtype=np.uint64)
    enc_s, ct = _timeit(lambda: he1.encrypt_share(z1), repeats)
    add_s, ct2 = _timeit(lambda: he0.add_share(None, ct, z0), repeats)
    dec = he1.decrypt(ct2)
    want = (z0.astype(object) + z1.astype(object)) % T_FIELD
    ok = bool(np.array_equal(dec.astype(object), want))
    recs = []
    for name, secs in (("bfv_context", setup_s), ("bfv_encrypt_share", enc_s), ("bfv_add_share", add_s)):
        s = _summary(secs, BFV_N)
        s.update({"correct": ok, "ct_bytes": int(len(ct)), "ct_bytes_after_add": int(len(ct2)),
                  "public_context_bytes": int(len(pub)), "bytes": int(len(ct))})
        recs.append(primitive_record("a2h_exact", BFV_N, s, {"N": BFV_N, "t": T_FIELD, "step": name},
                                     note="TenSEAL BFV; ct_bytes is the measured serialised ciphertext size"))
    return recs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(batches: Sequence[int], repeats: int, out_path: Path, only: Optional[Sequence[str]] = None,
        seed: int = SEED) -> List[dict]:
    rng = np.random.default_rng(seed)
    only_set = set(only) if only else set(PRIMITIVES)
    records: List[dict] = []
    for b in batches:
        if "dcf_eval" in only_set or "dcf_gen" in only_set:
            records += bench_dcf(int(b), repeats, rng)
        if "drelu" in only_set:
            records += bench_drelu(int(b), repeats, rng)
        if "trunc_faithful" in only_set:
            records += bench_trunc(int(b), repeats, rng)
        if "ring_to_field" in only_set:
            records += bench_r2f(int(b), repeats, rng)
    if "toy_bridge" in only_set:
        records += bench_toy_bridge(repeats, rng)
    if "a2h_exact" in only_set:
        records += bench_a2h_exact(repeats, rng)
    # the raw directory is <root>/raw: the record names its file relative to that root
    rel = schema.relative_raw_path(Path(out_path), Path(out_path).resolve().parent.parent)
    for r in records:
        r["raw_measurement_path"] = rel
        schema.append_jsonl(out_path, r)
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_primitives", description="low-level gate microbenchmarks")
    p.add_argument("--quick", action="store_true", help="small batches, fewer repeats")
    p.add_argument("--out", default=None, help="raw directory (default: raw/ under the data root)")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--batches", default=None, help="comma separated batch sizes")
    p.add_argument("--only", default=None, help=f"comma separated subset of {','.join(PRIMITIVES)}")
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    batches = tuple(int(v) for v in args.batches.split(",")) if args.batches else (QUICK_BATCHES if args.quick else DEFAULT_BATCHES)
    repeats = args.repeats if args.repeats is not None else (QUICK_REPEATS if args.quick else DEFAULT_REPEATS)
    out_dir = Path(args.out) if args.out else schema.experiments_root() / "raw"
    only = args.only.split(",") if args.only else None
    t0 = time.perf_counter()
    recs = run(batches, repeats, out_dir / "primitives.jsonl", only, args.seed)
    for r in recs:
        e = r["extra"]
        med = e.get("median_s")
        line = f"{e['component']:>14s} batch={e['batch']:>6d}"
        if med is not None:
            line += f" median={med * 1e3:9.3f} ms per_elem={e['per_element_s'] * 1e9:9.1f} ns"
        if "ct_bytes" in e:
            line += f" ct_bytes={e['ct_bytes']}"
        if e.get("skipped"):
            line += f" skipped ({e.get('reason')})"
        sys.stdout.write(line + "\n")
    sys.stdout.write(f"wrote {len(recs)} records to {out_dir / 'primitives.jsonl'} in {time.perf_counter() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["PRIMITIVES", "DEFAULT_BATCHES", "QUICK_BATCHES", "primitive_record", "bench_dcf", "bench_drelu",
           "bench_trunc", "bench_r2f", "bench_toy_bridge", "bench_a2h_exact", "run", "main"]
