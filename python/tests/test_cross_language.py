"""Cross-language parity, both directions.

Two groups of tests.

1. Committed vector files under ``cpp/tests/vectors`` (always run, no binary needed).
   The ``cpp_*.json`` files were written by the C++ implementation
   (``test_lowlevel <dir>``, ``test_vectors --emit <dir>``) and are replayed here with the
   Python package: PRG blocks and Convert words, DCF keys for P in {1, 2, 3} in both payload
   groups, DReLU, truncation and ring_to_field material, wire codec frames (none and bool
   included), contract hashes, cost-model predictions and formulas. The Python-written files of
   the same formats are replayed too, so a stale committed file is caught on this side as well
   (the C++ ``test_vectors`` executable checks them on its side). Controller decisions are
   covered by ``test_cross_language_controller.py``.

2. Live tests (skipped with a reason when ``cpp/build/bin`` holds no binaries):
   * Python to C++: ``gen_vectors.py`` writes fresh vectors (another seed, 2000 P = 2 DReLU
     keys) into a temporary directory and ``test_vectors`` must pass on them;
   * C++ to Python: ``test_vectors --emit`` writes fresh files (2000 P = 2 DReLU keys) into a
     temporary directory and the checks of group 1 run on them; the committed codec and contract
     files must equal the fresh ones exactly, the committed cost-model file up to the float
     tolerance of the checks (compilers differ in fused multiply-add);
   * round counters: the rounds and bytes the C++ gates measure on a loopback channel
     (``opencrab2_bench``) equal the Python formulas, which ``test_costmodel.py`` pins to the
     Python channel counters (equal counters per gadget).
   * live sessions (``cpp/tests/interop_cross_language.py``): the Python dealer service serves a
     C++ ``opencrab2_party`` pair, and the C++ ``opencrab2_dealer`` serves the Python tiny-model
     session; both must release the reference tokens with security-eligible records.

KEY_COMPONENT_INDEPENDENCE (paper, Section 5.3, a finding on comparison keys) in both languages:
the Python attack code of ``test_dcf_component_independence.py`` (whose positive control breaks
the old rule in 2000 of 2000 trials in that file) runs on C++ generated keys and must recover
nothing; the C++ attack code of ``test_vectors`` runs on Python generated keys and must recover
nothing (it carries its own positive control).

What this does not show: a session whose two PARTIES are of different languages (the session
layers differ from the first message on; the dealer dialogue and every material payload do
cross the boundary in the live sessions) and pseudo-randomness of the keys.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencrab2.highlevel import channel as hl_channel
from opencrab2.highlevel import contract as hl_contract
from opencrab2.highlevel import costmodel as cm
from opencrab2.lowlevel import dcf, drelu, prg, ring, ring_to_field as r2f, trunc

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "cpp" / "tests" / "vectors"
BIN = ROOT / "cpp" / "build" / "bin"
GEN = ROOT / "cpp" / "tests" / "gen_vectors.py"
EXE = ".exe" if os.name == "nt" else ""
MASK64 = (1 << 64) - 1
REL = 1e-9


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gv = _load_module("opencrab2_gen_vectors", GEN)
interop = _load_module("opencrab2_interop", ROOT / "cpp" / "tests" / "interop_cross_language.py")
kci = _load_module("opencrab2_kci_attack", Path(__file__).resolve().parent / "test_dcf_component_independence.py")


def _doc(path: Path) -> dict:
    assert path.is_file(), f"VECTORS_MISSING: {path}"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _u64(items) -> np.ndarray:
    return np.array([int(v) for v in items], dtype=np.uint64)


def _ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).reshape(-1)]


def _binary(name: str) -> Path:
    return BIN / (name + EXE)


def _need(*names: str) -> None:
    missing = [n for n in names if not _binary(n).is_file()]
    if missing:
        pytest.skip("C++ binaries missing (%s); build with: bash scripts/ubuntu.sh build on Ubuntu, "
                    "python cpp/build.py on Windows" % ", ".join(missing))


# ---------------------------------------------------------------------------
# checkers (one per file format; used on committed and on freshly generated files)
# ---------------------------------------------------------------------------
def check_prg_cpp(doc: dict) -> int:
    """cpp_prg_vectors.json: base expansion, raw blocks B_0..B_9 and the component-specific Convert words."""
    assert bytes.fromhex(doc["fixed_key_hex"]) == prg.FIXED_KEY
    n = 0
    for v in doc["vectors"]:
        seed = bytes.fromhex(v["seed_hex"])
        assert prg.expand_bytes(seed).hex() == v["out_hex"]
        seeds = prg.seeds_from_bytes(seed)
        sL, sR, vL, vR, tL, tR = prg.expand(seeds)
        assert np.ascontiguousarray(sL, dtype="<u8").tobytes().hex() == v["sL_hex"]
        assert np.ascontiguousarray(sR, dtype="<u8").tobytes().hex() == v["sR_hex"]
        assert (int(vL[0]), int(vR[0]), int(tL[0]), int(tR[0])) == (int(v["vL"]), int(v["vR"]), int(v["tL"]), int(v["tR"]))
        blocks = [prg.mmo_bytes(seed, j) for j in range(10)]
        assert [b.hex() for b in blocks] == v["blocks_hex"]
        # the Convert rule of lowlevel/prg.py, from the raw blocks (pure integers) and through the library Convert
        _sL, _sR, cL, cR, _tL, _tR = prg.expand_convert(seeds, 3, None)
        for c_str, (wl, wr) in v["ring_words"].items():
            c = int(c_str)
            blk = blocks[3 + c]
            assert int.from_bytes(blk[0:8], "little") == int(wl) and int.from_bytes(blk[8:16], "little") == int(wr)
            assert int(cL[0, c]) == int(wl) and int(cR[0, c]) == int(wr)
        assert int(cL[0, 0]) == int(v["vL"]) and int(cR[0, 0]) == int(v["vR"])     # component 0 is unchanged
        for t_str, words in v["zt_words"].items():
            t = int(t_str)
            _sL, _sR, zL, zR, _tL, _tR = prg.expand_convert(seeds, len(words), t)
            for c, (wl, wr) in enumerate(words):
                assert int.from_bytes(blocks[4 + 2 * c], "little") % t == int(wl)
                assert int.from_bytes(blocks[5 + 2 * c], "little") % t == int(wr)
                assert int(zL[0, c]) == int(wl) and int(zR[0, c]) == int(wr)
        n += 1
    assert n >= 17
    return n


def _check_dcf_case(name, n_bits, P, modulus, alpha, beta, x, key_hex, outs) -> None:
    mod = None if modulus == 0 else modulus
    got = []
    for p in range(2):
        raw = bytes.fromhex(key_hex[p])
        key = dcf.DCFKeyBatch.from_bytes(raw)
        assert (key.party, key.n_bits, key.P, key.M, key.modulus) == (p, n_bits, P, len(alpha), mod), name
        assert raw[0] == dcf.WIRE_VERSION and key.to_bytes() == raw, f"{name}: binding layout, byte-identical re-serialization"
        o = dcf.eval_dcf_batch(key, x)
        assert _ints(o) == [int(v) for v in outs[p]], f"{name}: party {p} output shares"
        got.append(o)
    rec = dcf.reconstruct_dcf(got[0], got[1], mod)
    b = beta.reshape(len(alpha), P)
    for m in range(len(alpha)):
        for c in range(P):
            want = (int(b[m, c]) % modulus if modulus else int(b[m, c])) if int(x[m]) < int(alpha[m]) else 0
            assert int(rec[m, c]) == want, f"{name}: reconstruction equals beta 1{{x < alpha}}"


def check_dcf(doc: dict) -> set:
    """dcf_vectors.json (Python origin) or cpp_dcf_vectors.json (C++ origin)."""
    seen = set()
    for c in doc["cases"]:
        keys = (c["key0"], c["key1"]) if "key0" in c else (c["key0_binding_hex"], c["key1_binding_hex"])
        modulus = int(c["modulus"])
        _check_dcf_case(c["name"], int(c["n_bits"]), int(c["P"]), modulus, _u64(c["alpha"]), _u64(c["beta"]), _u64(c["x"]),
                        keys, (c["out0"], c["out1"]))
        seen.add((int(c["P"]), "zt" if modulus else "ring"))
        if "key0_pythonv2_hex" in c:
            # the legacy layout (OC2DCF01) is read only for keys the repair did not change
            legacy = bytes.fromhex(c["key0_pythonv2_hex"])
            if int(c["P"]) == 1 and modulus == 0:
                k = dcf.DCFKeyBatch.from_bytes(legacy)
                assert k.to_bytes() == bytes.fromhex(c["key0_binding_hex"])
            else:
                with pytest.raises(Exception) as ei:
                    dcf.DCFKeyBatch.from_bytes(legacy)
                assert "DCF_WIRE_LEGACY_CONVERT" in str(ei.value)
    assert {(1, "ring"), (2, "ring"), (3, "ring"), (1, "zt"), (3, "zt")} <= seen and (2, "zt") in seen, seen
    return seen


def run_kci(mat, r, n_bits: int) -> tuple:
    """The attack of test_dcf_component_independence on ONE party's material: (mask, alpha, relation) hits."""
    return kci.run_attack((mat,), r, n_bits)


def check_drelu(doc: dict, min_kci_keys: int = 1) -> int:
    """drelu_vectors.json / cpp_drelu_vectors.json: material of either language evaluated by Python, and the KCI batch."""
    assert doc["format"] == "opencrab2-drelu-vectors-v1"
    seen = set()
    for c in doc["cases"]:
        name, n_bits, wp = c["name"], int(c["n_bits"]), bool(c["with_product"])
        r, y = _u64(c["r"]), _u64(c["y"])
        outs = []
        r_sum = np.zeros(len(r), np.uint64)
        for p in range(2):
            raw = bytes.fromhex(c["material%d" % p])
            mat = drelu.DReLUMaterial.from_bytes(raw)
            assert (mat.party, mat.n_bits, mat.M, mat.with_product) == (p, n_bits, len(y), wp), name
            assert mat.to_bytes() == raw, f"{name}: byte-identical re-serialization"
            r_sum = ring.add(r_sum, drelu.drelu_mask_share(mat))
            b, rb = drelu.drelu_eval(mat, y)
            assert _ints(b) == [int(v) for v in c["b%d" % p]], f"{name}: party {p} bit share"
            if wp:
                assert _ints(rb) == [int(v) for v in c["rb%d" % p]], f"{name}: party {p} product share"
            outs.append((b, rb))
        assert np.array_equal(r_sum, r), f"{name}: mask shares"
        bit = _u64(c["expected_bit"])
        xs = [int(v) for v in c["x_signed"]]
        assert [1 if v >= 0 else 0 for v in xs] == _ints(bit)
        assert [(v + int(r[i])) & MASK64 for i, v in enumerate(xs)] == _ints(y)
        assert np.array_equal(ring.add(outs[0][0], outs[1][0]), bit), f"{name}: reconstruction equals 1{{x >= 0}}"
        if wp:
            assert np.array_equal(ring.add(outs[0][1], outs[1][1]), ring.mul(r, bit)), f"{name}: reconstruction equals r 1{{x >= 0}}"
        seen.add((n_bits, wp))
    assert {(32, False), (32, True), (64, True)} <= seen
    k = doc["kci"]
    mat = drelu.DReLUMaterial.from_bytes(bytes.fromhex(k["material"]))
    r = _u64(k["r"])
    assert mat.key_p.P == 2 and mat.key_q.P == 2 and mat.M == int(k["M"]) == len(r) and mat.M >= min_kci_keys
    mask_hits, alpha_hits, relation_hits = run_kci(mat, r, int(k["n_bits"]))
    assert mask_hits == 0, "KEY_COMPONENT_INDEPENDENCE: the blind attack recovered a mask from a key of the other language"
    assert relation_hits == 0, "KEY_COMPONENT_INDEPENDENCE: a level difference equals +-(beta_1 - beta_0)"
    assert alpha_hits == 0, "KEY_COMPONENT_INDEPENDENCE: the non-zero pattern spells alpha"
    return 2 * mat.M


def check_material_cpp(doc: dict) -> None:
    """cpp_material_vectors.json: C++ truncation, DReLU and ring_to_field material evaluated by Python."""
    t = doc["trunc"]
    k, f = int(t["k"]), int(t["f"])
    y = _u64(t["y"])
    mats = [trunc.TruncMaterial.from_bytes(bytes.fromhex(t["material%d_binding_hex" % p])) for p in range(2)]
    assert all((m.party, m.f, m.L, m.k, m.M) == (p, f, int(t["L"]), k, len(y)) for p, m in enumerate(mats))
    shares = [_u64(t["x_share0"]), _u64(t["x_share1"])]
    opened = ring.reduce_k(ring.add(trunc.trunc_open_share(shares[0], mats[0], 0), trunc.trunc_open_share(shares[1], mats[1], 1)), k)
    assert np.array_equal(opened, y), "trunc: the opened y"
    z = [trunc.trunc_faithful_eval(mats[p], y, p) for p in range(2)]
    assert _ints(z[0]) == [int(v) for v in t["z0"]] and _ints(z[1]) == [int(v) for v in t["z1"]], "trunc: output shares"
    rec = ring.reduce_k(ring.add(z[0], z[1]), k)
    signed = [int(v) - (1 << k) if int(v) >= (1 << (k - 1)) else int(v) for v in rec]
    assert signed == [int(v) for v in t["floor_x_div_2f"]] == [int(v) >> f for v in t["x"]], "trunc: floor(x / 2^f)"
    for p in range(2):
        assert mats[p].to_bytes() == bytes.fromhex(t["material%d_binding_hex" % p])

    d = doc["drelu"]
    y = _u64(d["y"])
    r = _u64(d["r"])
    outs = []
    for p in range(2):
        mat = drelu.DReLUMaterial.from_bytes(bytes.fromhex(d["material%d_binding_hex" % p]))
        assert (mat.party, mat.n_bits, mat.with_product) == (p, int(d["n_bits"]), bool(d["with_product"]))
        b, rb = drelu.drelu_eval(mat, y)
        assert _ints(b) == [int(v) for v in d["b%d" % p]] and _ints(rb) == [int(v) for v in d["rb%d" % p]], "drelu: output shares"
        outs.append((b, rb))
    bit = _u64(d["drelu_x"])
    assert np.array_equal(ring.add(outs[0][0], outs[1][0]), bit) and [1 if int(v) >= 0 else 0 for v in d["x"]] == _ints(bit)
    assert np.array_equal(ring.add(outs[0][1], outs[1][1]), ring.mul(r, bit))

    q = doc["ring_to_field"]
    k, t_mod = int(q["k"]), int(q["t"])
    y = _u64(q["y"])
    mats = [r2f.R2FMaterial.from_bytes(bytes.fromhex(q["material%d_binding_hex" % p])) for p in range(2)]
    a = [_u64(q["a0"]), _u64(q["a1"])]
    opened = ring.reduce_k(ring.add(r2f.r2f_open_share(a[0], mats[0], 0), r2f.r2f_open_share(a[1], mats[1], 1)), k)
    assert np.array_equal(opened, y), "r2f: the opened y"
    z = [r2f.r2f_eval(mats[p], y, p) for p in range(2)]
    assert _ints(z[0]) == [int(v) for v in q["z0"]] and _ints(z[1]) == [int(v) for v in q["z1"]], "r2f: output shares"
    u = ring.reduce_k(ring.add(a[0], a[1]), k)
    want = _ints(r2f.signed_mod(u, t_mod, k))
    assert [(int(z[0][m]) + int(z[1][m])) % t_mod for m in range(len(y))] == want == [int(v) for v in q["z"]]


def _same_value(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.copysign(1.0, a) == math.copysign(1.0, b) and (a == b or (math.isnan(a) and math.isnan(b)))
    return a == b


def check_codec(doc: dict) -> int:
    """codec_vectors.json / cpp_codec_vectors.json: every value encodes to the other language's bytes and decodes from them."""
    assert doc["format"] == "opencrab2-codec-vectors-v1"
    tags = set()
    for c in doc["cases"]:
        name = c["name"]
        payload = bytes.fromhex(c["payload_hex"])
        frame = bytes.fromhex(c["frame_hex"])
        obj = gv.untagged(c["value"])
        assert hl_channel.encode_payload(obj) == payload, f"{name}: Python encodes the described value to the reference bytes"
        assert frame == len(payload).to_bytes(8, "big") + payload and hl_channel.frame_size(obj) == len(frame), f"{name}: frame"
        dec = hl_channel.decode_payload(payload)
        assert _same_value(dec, obj), f"{name}: Python decodes the reference bytes to the described value (types included)"
        assert gv.tagged(dec) == c["value"], f"{name}: description of the decoded value"
        assert hl_channel.encode_payload(dec) == payload
        tags.add(payload[0])
        if dec is None:
            assert payload == b"\x08"
        if isinstance(dec, bool):
            assert payload == (b"\x09\x01" if dec else b"\x09\x00")
    assert tags == {1, 2, 3, 4, 5, 6, 7, 8, 9}, f"every wire tag is covered: {sorted(tags)}"
    return len(doc["cases"])


def check_contract(doc: dict) -> int:
    """contract_vectors.json / cpp_contract_vectors.json."""
    assert doc["format"] == "opencrab2-contract-vectors-v1"
    names = set()
    for c in doc["cases"]:
        nc = hl_contract.NumericContract.from_json(c["json"])
        assert nc.to_json() == c["json"], c["name"]
        assert hl_contract.canonical_json_bytes(nc.to_json()).hex() == c["canonical_hex"], f"{c['name']}: canonical JSON bytes"
        assert nc.contract_hash() == c["hash"], f"{c['name']}: contract hash"
        if c["contract_name"] in hl_contract.CONTRACTS:
            assert hl_contract.contract_by_name(c["contract_name"]).contract_hash() == c["hash"], f"{c['name']}: named contract"
        names.add(c["name"])
    assert {"default", "smollm"} <= names
    default = next(c for c in doc["cases"] if c["name"] == "default")
    assert default["hash"] == hl_contract.DEFAULT_CONTRACT.contract_hash()
    return len(doc["cases"])


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= REL * max(1.0, abs(float(a)), abs(float(b)))


def check_costmodel(doc: dict) -> int:
    """costmodel_vectors.json / cpp_costmodel_vectors.json: formulas exactly, predictions to relative 1e-9."""
    assert doc["format"] == "opencrab2-costmodel-vectors-v1"
    model = cm.FittedCostModel.from_json(doc["model"])
    assert set(model.fits) == set(cm.COMPONENTS), "the coefficient table names the same components in both languages"
    envs = [cm.Env(e["bandwidth_bps"], e["one_way_latency_s"], e.get("cpu_share", 1.0)) for e in doc["envs"]]
    f = doc["formulas"]
    assert int(f["bfv_ct_bytes"]) == cm.bfv_ct_bytes() == cm.load_formulas()["bfv"]["ct_bytes"], "a2h_exact ciphertext bytes"
    assert int(f["frame_bytes"]) == cm.FRAME_BYTES
    assert int(f["drelu_p1_bytes"]) == cm.drelu_material_bytes(cm.DCF_DOMAIN_BITS, 1)
    assert int(f["drelu_p2_bytes"]) == cm.drelu_material_bytes(cm.DCF_DOMAIN_BITS, 2)
    assert int(f["trunc_faithful_bytes"]) == cm.trunc_material_bytes(cm.FRAC_BITS)
    assert int(f["ring_to_field_bytes"]) == cm.r2f_material_bytes()
    comps = set()
    for c in doc["cases"]:
        comp, n, d_in, d_out = c["component"], int(c["n"]), int(c["d_in"]), int(c["d_out"])
        ident = (comp, n, d_in, d_out)
        assert cm.bytes_formula(comp, n, d_in, d_out) == int(c["bytes"]), (ident, "bytes")
        assert cm.payload_bytes(comp, n, d_in, d_out) == int(c["payload_bytes"]), (ident, "payload_bytes")
        assert cm.rounds_formula(comp, n, d_in, d_out) == int(c["rounds"]), (ident, "rounds")
        assert cm.messages_formula(comp, n, d_in, d_out) == int(c["messages"]), (ident, "messages")
        assert cm.macs(comp, n, d_in, d_out) == int(c["macs"]), (ident, "macs")
        assert cm.material_formula(comp, n, d_in, d_out) == int(c["material_bytes"]), (ident, "material_bytes")
        assert len(c["predictions"]) == len(envs)
        for e, want in zip(envs, c["predictions"]):
            mean, sigma = model.predict(comp, n, d_in, d_out, e)
            assert _close(mean, want["mean_s"]), (ident, e, "mean", mean, want["mean_s"])
            assert _close(sigma, want["sigma_s"]), (ident, e, "sigma", sigma, want["sigma_s"])
        comps.add(comp)
    assert comps == set(cm.COMPONENTS), f"every component has a shape: missing {sorted(set(cm.COMPONENTS) - comps)}"
    for t in doc["tournament"]:
        levels, p1, p2, bv = cm.tournament_gates(int(t["n"]), int(t["k"]), bool(t["with_index"]))
        ident = (t["n"], t["k"], t["with_index"])
        assert (levels, 2 * levels) == (int(t["levels"]), int(t["rounds"])), (ident, "levels, rounds = 2 per level")
        assert (p1, p2, bv) == (int(t["drelu_p1"]), int(t["drelu_p2"]), int(t["beaver_select"])), (ident, "fused gate counts")
    assert len(doc["tournament"]) >= 30
    return len(doc["cases"])


# ---------------------------------------------------------------------------
# 1. committed vector files
# ---------------------------------------------------------------------------
def test_cpp_prg_blocks_and_convert_words_replayed_by_python():
    assert check_prg_cpp(_doc(VECTORS / "cpp_prg_vectors.json")) >= 17


@pytest.mark.parametrize("name", ["cpp_dcf_vectors.json", "dcf_vectors.json"])
def test_dcf_keys_of_both_languages_evaluate_identically_in_python(name):
    check_dcf(_doc(VECTORS / name))


@pytest.mark.parametrize("name", ["cpp_drelu_vectors.json", "drelu_vectors.json"])
def test_drelu_material_and_key_component_independence(name):
    assert check_drelu(_doc(VECTORS / name)) >= 2


def test_cpp_trunc_drelu_r2f_material_replayed_by_python():
    check_material_cpp(_doc(VECTORS / "cpp_material_vectors.json"))


@pytest.mark.parametrize("name", ["cpp_codec_vectors.json", "codec_vectors.json"])
def test_codec_frames_including_bool_and_none(name):
    assert check_codec(_doc(VECTORS / name)) >= 20


@pytest.mark.parametrize("name", ["cpp_contract_vectors.json", "contract_vectors.json"])
def test_contract_hashes(name):
    assert check_contract(_doc(VECTORS / name)) >= 3


@pytest.mark.parametrize("name", ["cpp_costmodel_vectors.json", "costmodel_vectors.json"])
def test_cost_model_predictions_and_formulas(name):
    assert check_costmodel(_doc(VECTORS / name)) >= 30


def test_committed_python_vectors_are_what_the_generator_writes_now():
    # the deterministic files (no randomness or a fixed seed) are regenerated in memory and compared
    assert gv.gen_contract() == _doc(VECTORS / "contract_vectors.json")
    assert json.loads(json.dumps(gv.gen_costmodel())) == _doc(VECTORS / "costmodel_vectors.json")
    assert gv.gen_codec(np.random.default_rng(2026 + 2)) == _doc(VECTORS / "codec_vectors.json")


def test_python_attack_code_is_the_one_with_a_positive_control():
    # run_attack is imported from the component-independence regression file, where it breaks the old rule in
    # 2000 of 2000 trials;
    # here a tiny emulation of the old shared word shows the import is live and the counts are per key
    rng = np.random.default_rng(5)
    r = ring.rand_u64((16,), rng)
    saved = (prg.expand_convert, prg.leaf_words)
    try:
        prg.expand_convert, prg.leaf_words = kci._legacy_expand_convert, kci._legacy_leaf_words
        m0, _m1 = drelu.drelu_material_from_mask(r, 32, rng, with_product=True)
    finally:
        prg.expand_convert, prg.leaf_words = saved
    assert run_kci(m0, r, 32) == (32, 32, 32)          # both keys of one party, 16 masks each
    n0, _n1 = drelu.drelu_material_from_mask(r, 32, rng, with_product=True)
    assert run_kci(n0, r, 32) == (0, 0, 0)


# ---------------------------------------------------------------------------
# 2. live tests against the C++ binaries
# ---------------------------------------------------------------------------
def _run(cmd, timeout=600):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def test_live_python_vectors_pass_the_cpp_checks(tmp_path):
    _need("test_vectors")
    code, out, err = _run([sys.executable, GEN, "--out", tmp_path, "--seed", 77, "--kci-keys", 2000])
    assert code == 0, out + err
    code, out, err = _run([_binary("test_vectors"), tmp_path])
    assert code == 0, out + err
    assert "PASS" in out and "SKIP" not in out, out
    for line in ("prg:", "dcf: 27 key batches ok", "trunc:", "r2f:", "controller:", "codec:", "contract:", "costmodel:"):
        assert line in out, (line, out)
    # the C++ attack ran on 2000 Python P = 2 DReLU materials (keys p and q) and recovered nothing
    assert "KEY_COMPONENT_INDEPENDENCE attack fails on 4000 python keys" in out, out


def test_live_cpp_vectors_pass_the_python_checks(tmp_path):
    _need("test_vectors")
    code, out, err = _run([_binary("test_vectors"), "--emit", tmp_path, "--kci-keys", 2000])
    assert code == 0, out + err
    assert check_drelu(_doc(tmp_path / "cpp_drelu_vectors.json"), min_kci_keys=2000) == 4000
    assert check_codec(_doc(tmp_path / "cpp_codec_vectors.json")) >= 20
    assert check_contract(_doc(tmp_path / "cpp_contract_vectors.json")) >= 3
    assert check_costmodel(_doc(tmp_path / "cpp_costmodel_vectors.json")) >= 30
    # the emitter is deterministic: the committed files are what this build writes. The codec and contract
    # files hold no float and must match exactly; the cost-model predictions may differ in the last digit
    # between compilers (a build that fuses multiply-add and one that does not), so their floats are compared
    # to the tolerance of _close and everything else exactly
    for name in ("cpp_codec_vectors.json", "cpp_contract_vectors.json"):
        assert _doc(tmp_path / name) == _doc(VECTORS / name), f"{name} is stale: run test_vectors --emit cpp/tests/vectors"
    _same_up_to_float_tolerance(_doc(tmp_path / "cpp_costmodel_vectors.json"), _doc(VECTORS / "cpp_costmodel_vectors.json"),
                                "cpp_costmodel_vectors.json")


def _same_up_to_float_tolerance(a, b, where: str) -> None:
    """Equal structure and equal non-float values; floats equal to the relative tolerance of :func:`_close`."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{where}: keys differ {sorted(set(a) ^ set(b))} (stale: run test_vectors --emit)"
        for k in a:
            _same_up_to_float_tolerance(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{where}: {len(a)} entries against {len(b)} (stale: run test_vectors --emit)"
        for i, (x, y) in enumerate(zip(a, b)):
            _same_up_to_float_tolerance(x, y, f"{where}[{i}]")
    elif (isinstance(a, float) or isinstance(b, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)) and _close(a, b), f"{where}: {a!r} != {b!r}"
    else:
        assert a == b, f"{where}: {a!r} != {b!r} (stale: run test_vectors --emit cpp/tests/vectors)"


def test_live_cpp_rejects_a_tampered_python_key(tmp_path):
    # the C++ check is not vacuous: one flipped bit in a correction word of a Python key must fail it
    _need("test_vectors")
    for name in ("prg_vectors.json", "dcf_vectors.json"):
        doc = _doc(VECTORS / name)
        if name == "dcf_vectors.json":
            case = next(c for c in doc["cases"] if c["P"] == 2 and c["modulus"] == "0" and c["n_bits"] == 32)
            raw = bytearray(bytes.fromhex(case["key1"]))
            raw[dcf.HEADER_BYTES + 16 * case["M"] + 3] ^= 0x10          # level 0 seed correction word (party 1 applies it: t = 1 at the root)
            case["key1"] = bytes(raw).hex()
        with open(tmp_path / name, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
    code, out, err = _run([_binary("test_vectors"), tmp_path])
    assert code == 1 and "FAIL" in err and "ring_n32_p2" in err, out + err


def test_live_measured_cpp_round_and_byte_counters_equal_the_python_formulas():
    """Equal round and byte counters per gadget in both languages.

    ``opencrab2_bench`` runs each gate between two C++ parties and records the channel counters.
    The Python formulas are pinned to the Python channel counters by ``test_costmodel.py``; the
    argmax gate of the bench releases the index (one message of 8 payload bytes, one round), which
    the formula excludes by convention (formulas.json "conventions").
    """
    _need("opencrab2_bench")
    code, out, err = _run([_binary("opencrab2_bench"), "--only", "drelu,trunc,r2f,argmax", "--batches", "64", "--repeats", "2",
                           "--argmax-n", "17,70", "--argmax-k", "2,4,8", "--argmax-repeats", "1", "--seed", "11"])
    assert code == 0, out[-2000:] + err[-2000:]
    seen = {}
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        ex = rec.get("extra", {})
        seen[(ex.get("component"), int(ex.get("batch", 0)))] = rec
    M = 64
    for comp in ("drelu_p1", "drelu_p2", "trunc_faithful", "ring_to_field"):
        rec = seen[(comp, M)]
        assert rec["output_match"] is True
        assert rec["rounds"] == 1 and rec["bytes"] == cm.OPEN_BYTES * M, (comp, rec["rounds"], rec["bytes"])
    assert cm.rounds_formula("trunc_faithful", 1, M, M) == 1 and cm.payload_bytes("trunc_faithful", 1, M, M) == cm.OPEN_BYTES * M
    assert cm.rounds_formula("ring_to_field", 1, M, M) == 1 and cm.payload_bytes("ring_to_field", 1, M, M) == cm.OPEN_BYTES * M
    assert seen[("drelu_p1", M)]["material_consumed"] == M * cm.drelu_material_bytes(cm.DCF_DOMAIN_BITS, 1)
    assert seen[("drelu_p2", M)]["material_consumed"] == M * cm.drelu_material_bytes(cm.DCF_DOMAIN_BITS, 2)
    assert seen[("trunc_faithful", M)]["material_consumed"] == M * cm.trunc_material_bytes(cm.FRAC_BITS)
    assert seen[("ring_to_field", M)]["material_consumed"] == M * cm.r2f_material_bytes()
    release_bytes = 8 + cm.FRAME_BYTES
    for n in (17, 70):
        for k in (2, 4, 8):
            comp = "argmax_k%d" % k
            rec = seen[(comp, n)]
            assert rec["output_match"] is True
            assert rec["rounds"] == cm.rounds_formula(comp, 1, n, 1) + 1, (comp, n, rec["rounds"])
            assert rec["bytes"] == cm.bytes_formula(comp, 1, n, 1) + release_bytes, (comp, n, rec["bytes"])
            assert rec["material_consumed"] == cm.material_formula(comp, 1, n, 1), (comp, n, rec["material_consumed"])
            levels, p1, p2, bv = cm.tournament_gates(n, k, True)
            params = seen[(comp + "_material", n)]["extra"]["params"]
            assert (params["drelu_p1"], params["drelu_p2"], params["kary_select"]) == (p1, p2, bv), (comp, n, params)


def test_live_python_dealer_serves_a_cpp_party_pair(tmp_path):
    """HELLO, CHALLENGE, AUTH, REQ, BUNDLE and every material payload cross the boundary Python to C++."""
    _need("opencrab2_party", "opencrab2_dealer")
    res = interop.run_py_dealer_cpp_parties(str(BIN), str(tmp_path), n_tokens=2)
    assert res["ok"], res["notes"]
    assert res["tokens"] == res["reference"] and len(res["tokens"]) == 2
    assert res["dealer_delivered"] > 0
    for rec in res["records"]:
        assert rec["security_eligible"] is True and rec["dealer_mode"] == "ISOLATED_RECIPIENT_BUNDLES"


def test_live_cpp_dealer_serves_the_python_session(tmp_path):
    """The same dialogue in the other direction: named requests (request_id, remaining) answered by the C++ dealer."""
    _need("opencrab2_dealer")
    res = interop.run_cpp_dealer_python_session(str(BIN), str(tmp_path), n_tokens=2)
    assert res["ok"], res["notes"]
    assert res["tokens"] == res["reference"] and len(res["tokens"]) == 2
    for rec in res["records"]:
        assert rec["security_eligible"] is True
        assert rec["output_match"] is (True if rec["extra"]["party"] == 1 else None)
