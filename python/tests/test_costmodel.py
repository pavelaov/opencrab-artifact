"""Tests for opencrab2.highlevel.costmodel (formulas, NNLS fit, prediction, default model, metrics).

Measured-versus-formula test.  Every
component of ``COMPONENTS`` is run once per declared small shape
(``costmodel.FORMULA_TEST_SHAPES``, at least three per component) in the
in-process two-party harness of tests/test_party.py (the one tests/test_gadgets.py
uses) and the channel counters are compared with the formulas.

Framing rule.  Every message is one frame.  The framing of a frame is its
size on the wire minus the raw data it carries (the element bytes of an
array, the bytes of a ciphertext): the 8-byte length prefix plus the typed
header of the wire codec.  A frame that carries one array of ndim dimensions
has ``FRAME_BYTES + 8 * (ndim - 1)`` bytes of framing (19 for a 1-D uint64
array).  The test records both figures for every frame, asserts that rule,
and compares ``measured bytes - measured framing`` with ``payload_bytes``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel.channel import frame_size
from tests.test_party import CONTRACT, TwoPartyHarness, enc, shares_of

ENV_LAN = cm.Env(1e9, 0.00025)
ENV_WAN = cm.Env(50e6, 0.02)


def _shape(comp):
    n, d_in, d_out = 2, 768, 768
    if comp.startswith("argmax") or comp.startswith("embedding"):
        d_in = 50257
    if comp.startswith("softmax"):
        d_in = 32
    return n, d_in, d_out


@pytest.mark.parametrize("comp", cm.COMPONENTS)
def test_formulas_positive_and_deterministic(comp):
    n, d_in, d_out = _shape(comp)
    b1 = cm.bytes_formula(comp, n, d_in, d_out)
    b2 = cm.bytes_formula(comp, n, d_in, d_out)
    assert b1 == b2 and b1 > 0
    assert cm.rounds_formula(comp, n, d_in, d_out) >= 1
    assert cm.material_formula(comp, n, d_in, d_out) >= 0
    assert cm.macs(comp, n, d_in, d_out) >= 1
    types = set(cm.material_breakdown(comp, n, d_in, d_out))
    assert types <= {"beaver", "matrix_triple", "wmask_mask", "drelu", "trunc_faithful", "ring_to_field", "kary_select"}


@pytest.mark.parametrize("comp", cm.COMPONENTS)
def test_bytes_monotone_in_elements(comp):
    n, d_in, d_out = _shape(comp)
    small = cm.bytes_formula(comp, n, d_in, d_out)
    large = cm.bytes_formula(comp, 2 * n, d_in, d_out) if comp != "install_wmask" else cm.bytes_formula(comp, n, 2 * d_in, d_out)
    assert large > small


def test_kary_tournament_geometry():
    for k in (2, 4, 8):
        levels, cmp_count = cm.kary_tournament(50257, k)
        assert cmp_count == 50256                       # every element but the winner loses exactly once
        assert levels == math.ceil(math.log(50257, k))
    assert cm.kary_tournament(1, 2) == (0, 0)
    assert cm.rounds_formula("argmax_k2", 1, 50257, 1) == 2 * 16
    assert cm.rounds_formula("argmax_k4", 1, 50257, 1) == 2 * 8
    assert cm.rounds_formula("argmax_k8", 1, 50257, 1) == 2 * 6
    assert cm.rounds_formula("argmax_k8", 1, 50257, 1) < cm.rounds_formula("argmax_k2", 1, 50257, 1)
    with pytest.raises(ValueError, match="^SHAPE_PARAM"):
        cm.kary_tournament(10, 1)


def test_install_and_primitive_byte_figures():
    framing = 2 * cm.FRAME_BYTES                                  # one round, two frames
    assert cm.FRAME_BYTES == 19 == frame_size(np.zeros(0, dtype=np.uint64))
    assert cm.bytes_formula("install_wmask", 1, 768, 768) == 16 * 768 * 768 + framing
    assert cm.bytes_formula("install_wmask", 1, 768, 3072) == 16 * 768 * 3072 + framing
    # both directions for every opened element
    assert cm.bytes_formula("embedding_generic", 1, 50257, 768) == 16 * (50257 + 50257 * 768) + framing
    assert cm.payload_bytes("embedding_generic", 3, 50257, 768) == cm.payload_bytes("linear_generic", 3, 50257, 768)
    assert cm.payload_bytes("embedding_wmask", 3, 50257, 768) == 16 * 3 * 50257
    assert cm.payload_bytes("trunc_faithful", 1, 1000, 1) == 16 * 1000
    assert cm.payload_bytes("ring_to_field", 1, 1000, 1) == 16 * 1000
    assert cm.rounds_formula("trunc_faithful", 1, 1000, 1) == 1
    assert cm.rounds_formula("ring_to_field", 1, 1000, 1) == 1
    n_ct = math.ceil(3 * 8192 / 8192)
    assert cm.bfv_ct_bytes() == cm.BFV_CT_BYTES == 432404        # the measured TenSEAL ciphertext size
    assert cm.payload_bytes("a2h_exact", 3, 8192, 1) == 16 * 3 * 8192 + n_ct * cm.bfv_ct_bytes()
    assert cm.payload_bytes("a2h_exact", 1, 8193, 1) == 16 * 8193 + 2 * cm.bfv_ct_bytes()
    assert cm.rounds_formula("a2h_exact", 1, 8192, 1) == 2 and cm.messages_formula("a2h_exact", 1, 8192, 1) == 3
    assert cm.rounds_formula("gelu_clip12", 1, 3072, 3072) == 14
    assert cm.rounds_formula("layernorm_rows", 1, 768, 768) == 18
    assert cm.rounds_formula("softmax_rows_k2", 12, 1, 1) == 23
    assert cm.rounds_formula("softmax_rows_k2", 12, 8, 8) == 29


def test_material_sizes_match_spec():
    assert cm.trunc_material_bytes(16) == 2064
    assert cm.r2f_material_bytes() == 3272
    assert cm.dcf_key_bytes(32, 2) == 1088
    assert cm.drelu_material_bytes(32, 1) == 2 * 824 + 24
    assert cm.material_formula("trunc_faithful", 1, 10, 1) == 10 * 2064
    assert cm.material_breakdown("ring_to_field", 2, 5, 1) == {"ring_to_field": 10 * 3272}
    assert cm.material_breakdown("install_wmask", 1, 768, 768) == {"wmask_mask": 8 * 768 * 768}
    assert cm.material_breakdown("linear_generic", 2, 3, 4) == {"matrix_triple": 8 * (2 * 3 + 3 * 4 + 2 * 4)}
    assert cm.drelu_material_bytes(32, 2) == 2 * 1088 + 24 == 2200
    # the fused k-ary realization spends fewer rounds and MORE material than the binary one
    m2, m4, m8 = (cm.material_formula(f"argmax_k{k}", 1, 50257, 1) for k in (2, 4, 8))
    assert m2 < m4 < m8
    assert cm.rounds_formula("argmax_k8", 1, 50257, 1) < cm.rounds_formula("argmax_k4", 1, 50257, 1) \
        < cm.rounds_formula("argmax_k2", 1, 50257, 1)


# ---------------------------------------------------------------------------
# fused k-ary tournament: gate counts
# ---------------------------------------------------------------------------
def test_tournament_gates_equal_dealer_and_session_counts():
    from opencrab2.highlevel.dealer import tournament_material
    from opencrab2.highlevel.session import tournament_counts
    for n in list(range(1, 260)) + [1000, 4097, 50257]:
        for k in (2, 4, 8):
            for with_index in (False, True):
                levels, p1, p2, bv = cm.tournament_gates(n, k, with_index)
                assert (p1, p2, bv) == tournament_material(n, k, with_index) == tournament_counts(n, k, with_index)
                assert levels == cm.kary_tournament(n, k)[0]
    with pytest.raises(ValueError, match="^SHAPE_PARAM"):
        cm.tournament_gates(8, 1, True)
    with pytest.raises(ValueError, match="^SHAPE_PARAM"):
        cm.tournament_gates(0, 2, True)


def test_tournament_gates_closed_cases():
    # one node of 8 members: 28 pairwise comparisons, 7 payload DReLUs (the index is public on the first level)
    assert cm.tournament_gates(8, 8, True) == (1, 28, 7, 0)
    assert cm.tournament_gates(8, 8, False) == (1, 28, 7, 0)
    # binary: n - 1 comparisons; one select per pair for the value, one more for the index after level one
    assert cm.tournament_gates(8, 2, False) == (3, 7, 0, 7)
    assert cm.tournament_gates(8, 2, True) == (3, 7, 0, 4 + 2 * 2 + 2 * 1)
    # 4-ary over 16: four nodes of 4 (6 pairs, 3 payload DReLUs each), then one node of 4 with a shared index
    assert cm.tournament_gates(16, 4, False) == (2, 5 * 6, 5 * 3, 0)
    assert cm.tournament_gates(16, 4, True) == (2, 5 * 6, 4 * 3 + 2 * 3, 0)
    # a remainder node of 2 members inside a 4-ary level is a binary node: Beaver select, no payload DReLU
    assert cm.tournament_gates(6, 4, False) == (2, 6 + 1 + 1, 3, 1 + 1)
    assert cm.tournament_gates(1, 4, True) == (0, 0, 0, 0)
    assert cm.tournament_node_sizes(6, 4) == (2, 4) and cm.tournament_node_sizes(50257, 8) == (2, 3, 5, 8)
    # the argmax and softmax formulas are exactly these counts
    g = cm.gate_counts("argmax_k4", 3, 16, 1)
    assert (g.drelu_p1, g.drelu_p2, g.kary_select, g.rounds, g.messages) == (3 * 30, 3 * 18, 0, 4, 8)
    assert cm.payload_bytes("argmax_k4", 3, 16, 1) == 16 * 3 * (30 + 18)
    assert cm.material_breakdown("argmax_k4", 3, 16, 1) == {"drelu": 3 * (30 * 1672 + 18 * 2200)}
    s = cm.gate_counts("softmax_rows_k4", 2, 16, 16)
    assert s.drelu_p1 == 2 * 30 + 32 + 11 * 2 and s.drelu_p2 == 2 * 15 and s.kary_select == 0
    assert s.beaver == 9 * 32 + 11 * 2 + 32 and s.rounds == 2 * 2 + 10 + 12 + 1


def test_gate_counts_follow_the_contract():
    from opencrab2.highlevel.contract import DEFAULT_CONTRACT, SMOLLM_CONTRACT
    P = cm.protocol_params(DEFAULT_CONTRACT)
    assert P == cm.ProtocolParams() and cm.protocol_params(None) == P
    assert (P.recip_live, P.rsqrt_live) == (cm.RECIP_LIVE, cm.RSQRT_LIVE) == (11, 14)
    assert (P.exp_products, P.exp_rounds, P.recip_products, P.recip_rounds) == (9, 10, 11, 12)
    assert (P.rsqrt_products, P.rsqrt_rounds, P.gelu_products, P.gelu_rounds) == (13, 14, 14, 14)
    assert cm.LAYERNORM_ROUNDS == cm.rounds_formula("layernorm_rows", 1, 768, 768) == 18
    # the model-specific contract has m = 16 thresholds, of which the DCF bound still admits 14
    S = cm.protocol_params(SMOLLM_CONTRACT)
    assert S.rsqrt_m == 16 and S.rsqrt_live == 14
    assert cm.material_formula("layernorm_rows", 2, 8, 8, SMOLLM_CONTRACT) == cm.material_formula("layernorm_rows", 2, 8, 8)
    # per element: three Beaver products and one faithful truncation; per row: rsqrt
    g = cm.gate_counts("layernorm_rows", 2, 8, 8)
    assert (g.beaver, g.trunc, g.drelu_p1) == (3 * 16 + 13 * 2, 16, 14 * 2)
    assert cm.payload_bytes("layernorm_rows", 2, 8, 8) == 112 * 16 + 640 * 2
    assert cm.payload_bytes("gelu_clip12", 2, 8, 8) == 480 * 16
    assert cm.material_breakdown("gelu_clip12", 1, 6, 6) == {"drelu": 12 * 1672, "beaver": 14 * 6 * 24}


def test_unknown_component_rejected_with_code():
    with pytest.raises(ValueError, match="^UNKNOWN_COMPONENT"):
        cm.bytes_formula("he_linear", 1, 1, 1)
    with pytest.raises(ValueError, match="^SHAPE_PARAM"):
        cm.bytes_formula("linear_wmask", 0, 1, 1)
    m = cm.FittedCostModel()
    with pytest.raises(ValueError, match="^NO_FIT"):
        m.predict("linear_wmask", 1, 1, 1, ENV_LAN)


def _synthetic_rows(comp, a, k, b, rng, repeats=2):
    rows = []
    shapes = [(1, 256, 256), (4, 512, 512), (8, 768, 3072), (2, 768, 768), (16, 256, 768)]
    if comp.startswith("argmax") or comp.startswith("embedding"):
        shapes = [(1, 1000, 64), (1, 5000, 64), (2, 20000, 64), (1, 50257, 768), (3, 3000, 64)]
    if comp.startswith("softmax"):
        shapes = [(12, 4, 4), (24, 16, 16), (96, 32, 32), (12, 64, 64), (48, 8, 8)]
    for (n, d_in, d_out) in shapes:
        for env in (ENV_LAN, ENV_WAN):
            nbytes = cm.bytes_formula(comp, n, d_in, d_out)
            rounds = cm.rounds_formula(comp, n, d_in, d_out)
            compute = a * cm.macs(comp, n, d_in, d_out) + k * nbytes + b
            for _ in range(repeats):
                rows.append({"component": comp, "n": n, "d_in": d_in, "d_out": d_out,
                             "bandwidth_bps": env.bandwidth_bps, "one_way_latency_s": env.one_way_latency_s,
                             "measured_s": compute + cm.transport_s(nbytes, rounds, env),
                             "bytes": nbytes, "rounds": rounds})
    return rows


@pytest.mark.parametrize("comp", ["linear_wmask", "gelu_clip12", "argmax_k4", "softmax_rows_k2", "install_wmask"])
def test_fit_recovers_planted_coefficients(comp):
    rng = np.random.default_rng(1)
    a, k, b = 3e-9, 2e-9, 0.004
    model = cm.FittedCostModel.fit(_synthetic_rows(comp, a, k, b, rng))
    f = model.fits[comp]
    assert f.n_train == 20
    if comp in cm.LINEAR_COMPONENTS:
        # MACs (n d_in d_out) and bytes (16 n d_in) are not collinear: every coefficient is identifiable
        assert math.isclose(f.compute_per_mac_s, a, rel_tol=1e-5, abs_tol=1e-15)
        assert math.isclose(f.compute_per_byte_s, k, rel_tol=1e-5, abs_tol=1e-15)
        assert math.isclose(f.compute_fixed_s, b, rel_tol=1e-5, abs_tol=1e-12)
    else:
        # elementwise and row components: bytes is affine in the element count, so only the
        # combination a + k * bytes_per_element is identifiable; the prediction below is exact anyway
        assert f.compute_per_mac_s >= 0 and f.compute_per_byte_s >= 0 and f.compute_fixed_s >= 0
    assert f.sigma_compute_s < 1e-9
    shapes_out = [(2, 300, 300)] if comp not in ("argmax_k4",) else [(1, 777, 1)]
    for (n, d_in, d_out) in shapes_out:
        mean, sigma = model.predict(comp, n, d_in, d_out, ENV_WAN)
        nbytes = cm.bytes_formula(comp, n, d_in, d_out)
        want = a * cm.macs(comp, n, d_in, d_out) + k * nbytes + b + cm.transport_s(nbytes, cm.rounds_formula(comp, n, d_in, d_out), ENV_WAN)
        assert math.isclose(mean, want, rel_tol=1e-6)
        assert sigma >= 0.0


def test_fit_nnls_never_negative_and_sigma_from_repeats():
    rng = np.random.default_rng(7)
    rows = _synthetic_rows("linear_generic", 1e-9, 0.0, 0.001, rng, repeats=3)
    for r in rows:
        r["measured_s"] += rng.normal(0.0, 0.002)
    m = cm.FittedCostModel.fit(rows)
    f = m.fits["linear_generic"]
    assert f.compute_per_mac_s >= 0 and f.compute_per_byte_s >= 0 and f.compute_fixed_s >= 0
    assert 0.0005 < f.sigma_compute_s < 0.01


def test_env_outside_its_domain_is_refused_not_repaired():
    # cpu_share <= 0 used to be priced as 1.0 and a NaN bandwidth gave (nan, nan); the C++ model returns
    # ENV_PARAMS for the same inputs (cpp test_decision)
    from opencrab2.errors import ContractViolation, OpenCrabError
    m = cm.handwritten_model()
    bad = [cm.Env(1e9, 0.001, 0.0), cm.Env(1e9, 0.001, -0.5), cm.Env(1e9, 0.001, float("nan")), cm.Env(1e9, 0.001, float("inf")),
           cm.Env(float("nan"), 0.001), cm.Env(float("inf"), 0.001), cm.Env(-1.0, 0.001), cm.Env(1e9, float("nan")),
           cm.Env(1e9, -0.001), cm.Env(1e9, float("inf")), {"bandwidth_bps": 1e9, "rtt_s": 0.002, "cpu_share": 0.0}]
    for env in bad:
        with pytest.raises(cm.EnvError) as ei:
            m.predict("linear_wmask", 4, 768, 768, env)
        assert ei.value.code == "ENV_PARAMS" and isinstance(ei.value, (ContractViolation, OpenCrabError, ValueError))
        with pytest.raises(cm.EnvError):
            cm.handwritten_predict("linear_wmask", 4, 768, 768, env)
    for env in (cm.Env(None, 0.0), cm.Env(0.0, 0.0), cm.Env(1e9, 0.0, 0.25)):
        mean, sigma = m.predict("linear_wmask", 4, 768, 768, env)
        assert math.isfinite(mean) and math.isfinite(sigma) and mean > 0
    # bandwidth None and 0 both mean unlimited
    assert m.predict("linear_wmask", 4, 768, 768, cm.Env(None, 0.001)) == m.predict("linear_wmask", 4, 768, 768, cm.Env(0.0, 0.001))


def test_predict_moves_with_environment_and_cpu_share():
    m = cm.FittedCostModel.fit(_synthetic_rows("linear_wmask", 2e-9, 0.0, 0.002, np.random.default_rng(0)))
    fast, s_fast = m.predict("linear_wmask", 4, 768, 768, ENV_LAN)
    slow, s_slow = m.predict("linear_wmask", 4, 768, 768, ENV_WAN)
    assert slow > fast and s_slow > s_fast
    unlimited, _ = m.predict("linear_wmask", 4, 768, 768, cm.Env(None, 0.0))
    half, _ = m.predict("linear_wmask", 4, 768, 768, cm.Env(None, 0.0, cpu_share=0.5))
    assert math.isclose(half, 2 * unlimited, rel_tol=1e-9)
    assert math.isclose(cm.Env(1.0, 0.01).rtt_s, 0.02)
    assert cm.as_env({"bandwidth_bps": 1e6, "rtt_s": 0.02}).one_way_latency_s == 0.01


def test_json_round_trip_and_perturbed(tmp_path):
    m = cm.FittedCostModel.fit(_synthetic_rows("layernorm_rows", 1e-7, 0.0, 0.01, np.random.default_rng(0)))
    p = tmp_path / "fit.json"
    m.save(p)
    m2 = cm.FittedCostModel.load(p)
    assert m2.to_json() == m.to_json()
    assert json.dumps(m.to_json())
    m3 = m.perturbed({"layernorm_rows": {"compute_per_mac_s": 2.0}})
    assert math.isclose(m3.fits["layernorm_rows"].compute_per_mac_s, 2 * m.fits["layernorm_rows"].compute_per_mac_s)
    assert m3.fits["layernorm_rows"].status == m.fits["layernorm_rows"].status


@pytest.mark.parametrize("comp", cm.COMPONENTS)
def test_handwritten_predict_finite(comp):
    n, d_in, d_out = _shape(comp)
    v = cm.handwritten_predict(comp, n, d_in, d_out, ENV_WAN)
    assert math.isfinite(v) and v > 0
    hm = cm.handwritten_model()
    mean, sigma = hm.predict(comp, n, d_in, d_out, ENV_WAN)
    assert math.isfinite(mean) and sigma >= 0 and hm.fits[comp].status == "MODELED"


def test_error_metrics_known_vectors():
    m = [1.0, 2.0, 3.0, 4.0]
    perfect = cm.error_metrics(m, m)
    assert perfect["mape"] == 0.0 and perfect["spearman"] == 1.0 and perfect["kendall"] == 1.0 and perfect["n"] == 4
    reversed_ = cm.error_metrics(m, [4.0, 3.0, 2.0, 1.0])
    assert reversed_["spearman"] == -1.0 and reversed_["kendall"] == -1.0
    scaled = cm.error_metrics(m, [1.1, 2.2, 3.3, 4.4])
    assert math.isclose(scaled["mape"], 0.1, rel_tol=1e-9)
    with pytest.raises(ValueError, match="^METRIC_SHAPE"):
        cm.error_metrics([1.0], [1.0, 2.0])


def test_default_model_follows_the_data_root(tmp_path, monkeypatch):
    """The shipped calibration is the default; a rerun root named by OPENCRAB2_RESULTS_DIR replaces it only when
    it holds fitted_model.json; formulas.json is always the shipped file."""
    shipped = cm.shipped_calibration_dir() / "fitted_model.json"
    assert shipped.is_file() and cm.calibration_path() == shipped
    monkeypatch.setattr(cm, "_DEFAULT", None)
    assert cm.default_model() is cm.default_model()
    assert cm.default_model().to_json() == cm.FittedCostModel.load(shipped).to_json()
    monkeypatch.setenv("OPENCRAB2_RESULTS_DIR", str(tmp_path))
    assert cm.calibration_path() == shipped                     # the rerun root has no calibration yet
    assert cm.formulas_path() == cm.shipped_calibration_dir() / "formulas.json"
    (tmp_path / "calibration").mkdir()
    other = cm.handwritten_model()
    other.label = "rerun"
    other.save(tmp_path / "calibration" / "fitted_model.json")
    assert cm.calibration_path() == tmp_path / "calibration" / "fitted_model.json"
    monkeypatch.setattr(cm, "_DEFAULT", None)
    assert cm.default_model().label == "rerun"
    assert cm.formulas_path() == cm.shipped_calibration_dir() / "formulas.json"
    monkeypatch.setattr(cm, "_DEFAULT", None)                   # the module cache is left as the test found it


def test_no_em_dashes_in_source():
    src = Path(cm.__file__).read_text(encoding="utf-8")
    assert chr(0x2014) not in src and chr(0x2013) not in src



# ---------------------------------------------------------------------------
# measured channel counters versus the formulas (two-party harness)
# ---------------------------------------------------------------------------
T_FIELD = cm.BFV_T


def _raw_bytes(obj) -> int:
    """Raw data bytes a frame carries: array elements and byte strings; everything else is framing."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, dict):
        return sum(_raw_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_raw_bytes(v) for v in obj)
    return 0


class _Meter:
    """Records (frame bytes, raw data bytes, ndim) of every frame both parties send through the harness channels."""

    def __init__(self, h: TwoPartyHarness):
        self.h = h
        self.frames = []
        for ch in h.channels:
            original = ch.send

            def send(obj, _original=original):
                ndim = int(obj.ndim) if isinstance(obj, np.ndarray) else None
                self.frames.append((frame_size(obj), _raw_bytes(obj), ndim))
                _original(obj)

            ch.send = send
        # exact provisioning: every bundle holds what one request asks for, so a comparison request is never
        # served from leftover payload stock and the consumed material equals the gates issued
        for a in h.auto:
            a.DRELU_CHUNK = 1
            a.BEAVER_CHUNK = 1
            a.TRUNC_CHUNK = 1

    def run(self, fn0, fn1=None) -> dict:
        self.frames.clear()
        before = self.h.channels[0].stats.copy()
        mat0 = [c.consumed_bytes for c in self.h.clients]
        self.h.run(fn0, fn1 if fn1 is not None else fn0)
        d = self.h.channels[0].stats.diff(before)
        frames = list(self.frames)
        total = d["bytes_sent"] + d["bytes_recv"]
        assert total == sum(f for f, _, _ in frames)             # the meter saw every frame of the gadget
        assert d["messages_sent"] + d["messages_recv"] == len(frames)
        return {"rounds": d["rounds"], "messages": len(frames), "total": total, "frames": frames,
                "framing": sum(f - r for f, r, _ in frames), "payload": sum(r for _, r, _ in frames),
                "material": [c.consumed_bytes - m for c, m in zip(self.h.clients, mat0)]}


def _run_component(comp: str, n: int, d_in: int, d_out: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k = int(comp.rsplit("_k", 1)[1]) if "_k" in comp else 0
    secrets, masks = {}, {}
    W = None
    if comp in ("linear_wmask", "linear_generic", "embedding_wmask", "embedding_generic", "matmul_shared", "install_wmask"):
        W = enc(rng.uniform(-0.1, 0.1, size=(d_in, d_out)))
        secrets, masks = {"M": W}, {"M": (d_in, d_out)}
    h = TwoPartyHarness(seed=seed, secrets=secrets, mask_shapes=masks)
    try:
        meter = _Meter(h)
        if comp in ("embedding_wmask", "embedding_generic"):
            onehot = np.zeros((n, d_in), np.uint64)
            onehot[np.arange(n), rng.integers(0, d_in, size=n)] = np.uint64(1)
            xs = (np.zeros_like(onehot), onehot)                  # the client holds its one-hot as its own share
        elif comp.startswith("argmax_k"):
            xs = shares_of(enc(rng.integers(-500, 500, size=(n, d_in)).astype(np.float64)), 3)
        elif comp == "layernorm_rows":
            xs = shares_of(enc(rng.normal(0.0, 1.0, size=(n, d_in))), 3)
        elif comp.startswith("softmax_rows_k"):
            xs = shares_of(enc(rng.uniform(-4.0, 4.0, size=(n, d_in))), 3)
        elif comp == "gelu_clip12":
            xs = shares_of(enc(rng.uniform(-6.0, 6.0, size=(n, d_in))), 3)
        elif comp in ("trunc_faithful", "ring_to_field", "a2h_exact"):
            xs = shares_of(enc(rng.uniform(-1000.0, 1000.0, size=(n, d_in))), 3)
        else:
            xs = shares_of(enc(rng.uniform(-1.0, 1.0, size=(n, d_in))), 3)
        # one-time preparation outside the measured call
        if comp in ("linear_wmask", "embedding_wmask"):
            h.run_same(lambda p: p.install_wmask("M", W if p.party == 0 else None))
        elif comp in ("linear_generic", "embedding_generic"):
            h.run_same(lambda p: p.vector_share("M").shape)
        elif comp == "a2h_exact":
            warm = shares_of(enc(np.zeros(2)), 4)
            h.run_same(lambda p: p.a2h_exact(warm[p.party], T_FIELD) is None)   # sends the public BFV context once
        if comp == "linear_wmask":
            fn = lambda p: p.linear_wmask(xs[p.party], "M")
        elif comp == "embedding_wmask":
            fn = lambda p: p.linear_wmask(xs[p.party], "M", trunc=False)
        elif comp == "linear_generic":
            fn = lambda p: p.linear_generic(xs[p.party], "M")
        elif comp == "embedding_generic":
            fn = lambda p: p.linear_generic(xs[p.party], "M", trunc=False)
        elif comp == "matmul_shared":
            ws = shares_of(W, 5)
            fn = lambda p: p.matmul_shared(xs[p.party], ws[p.party])
        elif comp == "install_wmask":
            fn = lambda p: p.install_wmask("M", W if p.party == 0 else None)["rounds"]
        elif comp == "gelu_clip12":
            fn = lambda p: p.gelu(xs[p.party])
        elif comp.startswith("softmax_rows_k"):
            fn = lambda p: p.softmax_rows(xs[p.party], None, k_max=k)
        elif comp == "layernorm_rows":
            gs = shares_of(enc(rng.uniform(0.5, 1.5, size=d_in)), 6)
            bs = shares_of(enc(rng.uniform(-0.5, 0.5, size=d_in)), 7)
            fn = lambda p: p.layernorm_rows(xs[p.party], gs[p.party], bs[p.party])
        elif comp.startswith("argmax_k"):
            fn = lambda p: p.argmax_row(xs[p.party], k=k, release=False)
        elif comp == "trunc_faithful":
            fn = lambda p: p.trunc_faithful(xs[p.party])
        elif comp == "ring_to_field":
            fn = lambda p: p.ring_to_field(xs[p.party], T_FIELD)
        elif comp == "a2h_exact":
            fn = lambda p: p.a2h_exact(xs[p.party], T_FIELD) is None
        else:  # pragma: no cover
            raise AssertionError(comp)
        return meter.run(fn)
    finally:
        h.close()


_MEASURED_CASES = [(comp, n, d_in, d_out) for comp in cm.COMPONENTS for (n, d_in, d_out) in cm.FORMULA_TEST_SHAPES[comp]]


def test_every_component_has_three_measured_shapes():
    assert set(cm.FORMULA_TEST_SHAPES) == set(cm.COMPONENTS)
    for comp, shapes in cm.FORMULA_TEST_SHAPES.items():
        assert len(set(shapes)) >= 3, comp


@pytest.mark.parametrize("comp,n,d_in,d_out", _MEASURED_CASES, ids=[f"{c}-{n}x{a}x{b}" for c, n, a, b in _MEASURED_CASES])
def test_measured_channel_counters_equal_formulas(comp, n, d_in, d_out):
    """One call of the gadget: rounds, messages, payload bytes and material equal the formulas exactly."""
    m = _run_component(comp, n, d_in, d_out, seed=900 + len(comp) + n + d_in)
    assert m["rounds"] == cm.rounds_formula(comp, n, d_in, d_out)
    assert m["messages"] == cm.messages_formula(comp, n, d_in, d_out)
    # framing rule: an array frame of ndim dimensions carries FRAME_BYTES + 8 (ndim - 1) bytes of framing
    array_frames = [(f, r, nd) for f, r, nd in m["frames"] if nd is not None]
    for f, r, nd in array_frames:
        assert f - r == cm.FRAME_BYTES + cm.FRAME_DIM_BYTES * (nd - 1), (f, r, nd)
    measured_payload = m["total"] - m["framing"]
    assert measured_payload == m["payload"]
    if comp == "a2h_exact":
        N = n * d_in
        n_ct = -(-N // cm.BFV_N)
        assert len(array_frames) == 2 and sum(r for _, r, _ in array_frames) == 16 * N      # ring_to_field opening
        ct_frames = [(f, r) for f, r, nd in m["frames"] if nd is None]
        assert len(ct_frames) == 1                                                           # one frame P1 -> P0
        ct_bytes = ct_frames[0][1]
        # the TenSEAL serialization is compressed: its size varies around the measured value
        assert abs(ct_bytes - n_ct * cm.bfv_ct_bytes()) <= cm.BFV_CT_REL_TOL * n_ct * cm.bfv_ct_bytes(), ct_bytes
        assert abs(measured_payload - cm.payload_bytes(comp, n, d_in, d_out)) <= cm.BFV_CT_REL_TOL * n_ct * cm.bfv_ct_bytes()
    else:
        assert measured_payload == cm.payload_bytes(comp, n, d_in, d_out)
        extra_dims = sum(cm.FRAME_DIM_BYTES * (nd - 1) for _, _, nd in array_frames)
        assert m["total"] == cm.bytes_formula(comp, n, d_in, d_out) + extra_dims
    want_material = cm.material_formula(comp, n, d_in, d_out, CONTRACT)
    assert m["material"] == [want_material, want_material]


def test_argmax_release_is_one_extra_message_outside_the_component():
    """argmax_k* prices the tournament; RELEASE_FINAL adds one frame of 8 bytes per row and one round."""
    h = TwoPartyHarness(seed=77, release=("token", 4))
    try:
        meter = _Meter(h)
        xs = shares_of(enc(np.arange(8, dtype=np.float64)), 3)
        m = meter.run(lambda p: p.argmax_row(xs[p.party], k=4))
        assert m["rounds"] == cm.rounds_formula("argmax_k4", 1, 8, 1) + 1
        assert m["messages"] == cm.messages_formula("argmax_k4", 1, 8, 1) + 1
        assert m["payload"] == cm.payload_bytes("argmax_k4", 1, 8, 1) + 8
    finally:
        h.close()


def test_public_mask_only_removes_pairwise_comparisons():
    """The formulas price the unmasked row; a public mask lowers bytes, material and rounds, never raises them."""
    h = TwoPartyHarness(seed=78)
    try:
        meter = _Meter(h)
        x = np.random.default_rng(0).uniform(-4.0, 4.0, size=(3, 17))
        xs = shares_of(enc(x), 3)
        mask = np.tril(np.ones((3, 17), bool), k=13)
        m = meter.run(lambda p: p.softmax_rows(xs[p.party], mask, k_max=4))
        # a level whose comparisons are all decided by the public mask needs no comparison round
        assert m["rounds"] <= cm.rounds_formula("softmax_rows_k4", 3, 17, 17)
        assert m["payload"] < cm.payload_bytes("softmax_rows_k4", 3, 17, 17)
        assert m["material"][0] < cm.material_formula("softmax_rows_k4", 3, 17, 17, CONTRACT)
    finally:
        h.close()


# ---------------------------------------------------------------------------
# shared formula file and the measured BFV ciphertext size
# ---------------------------------------------------------------------------
def test_formula_shapes_cover_gpt2_and_the_measured_shapes():
    shapes = cm.formula_shapes()
    assert len(shapes) == len(set(shapes))
    comps = {c for c, _, _, _ in shapes}
    assert comps == set(cm.COMPONENTS)
    for want in [("embedding_generic", 1, 50257, 768), ("embedding_wmask", 16, 50257, 768), ("linear_wmask", 1, 768, 2304),
                 ("linear_generic", 16, 768, 3072), ("linear_wmask", 1, 768, 50257), ("install_wmask", 1, 50257, 768),
                 ("matmul_shared", 1, 64, 32), ("softmax_rows_k4", 12, 32, 32), ("softmax_rows_k2", 192, 16, 16),
                 ("layernorm_rows", 1, 768, 768), ("gelu_clip12", 16, 3072, 3072), ("argmax_k2", 1, 50257, 1),
                 ("argmax_k4", 1, 50257, 1), ("argmax_k8", 1, 50257, 1), ("a2h_exact", 1, 8192, 8192)]:
        assert want in shapes, want
    for comp, small in cm.FORMULA_TEST_SHAPES.items():
        assert all((comp,) + tuple(sh) in shapes for sh in small)


def test_write_formulas_json_round_trip(tmp_path):
    out = tmp_path / "formulas.json"
    doc = cm.write_formulas_json(out, measure=False)
    loaded = cm.load_formulas(out)
    assert loaded == json.loads(json.dumps(doc))
    assert loaded["schema"] == cm.FORMULAS_SCHEMA and loaded["bfv"]["ct_bytes"] == cm.BFV_CT_BYTES
    assert loaded["bfv"]["status"] == "MEASURED_NATIVE" and "provenance" in loaded["bfv"]
    assert loaded["bfv"]["provenance"]["confirmed_this_run"] is False
    assert loaded["framing"]["frame_bytes"] == cm.FRAME_BYTES
    assert len(loaded["shapes"]) == len(cm.formula_shapes())
    for rec in loaded["shapes"]:
        c, n, a, b = rec["component"], rec["n"], rec["d_in"], rec["d_out"]
        assert rec["bytes"] == cm.bytes_formula(c, n, a, b) == rec["payload_bytes"] + cm.FRAME_BYTES * rec["messages"]
        assert rec["payload_bytes"] == cm.payload_bytes(c, n, a, b)
        assert rec["rounds"] == cm.rounds_formula(c, n, a, b) and rec["messages"] == cm.messages_formula(c, n, a, b)
        assert rec["material_bytes"] == cm.material_formula(c, n, a, b) == sum(rec["material"].values())
        assert rec["material"] == cm.material_breakdown(c, n, a, b)
        assert rec["gates"] == cm.gate_counts(c, n, a, b).to_dict() and rec["macs"] == cm.macs(c, n, a, b)
    for row in loaded["tournament"]:
        assert (row["levels"], row["drelu_p1"], row["drelu_p2"], row["beaver_select"]) == \
            cm.tournament_gates(row["n"], row["k"], row["with_index"])
    text = out.read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text
    # a second write is byte-identical: the document is a pure function of the module when nothing is measured
    cm.write_formulas_json(tmp_path / "again.json", measure=False)
    assert (tmp_path / "again.json").read_text(encoding="utf-8") == text


def test_committed_formulas_json_is_current():
    """results/calibration/formulas.json (shared with the C++ cost model) equals a fresh generation."""
    path = cm.formulas_path()
    assert path.is_file(), f"FORMULAS_JSON_MISSING: run costmodel.write_formulas_json() ({path})"
    committed = cm.load_formulas(path)
    fresh = json.loads(json.dumps(cm.formulas_document(None)))
    for key in ("schema", "contract", "conventions", "framing", "constants", "tournament", "shapes"):
        assert committed[key] == fresh[key], key
    assert committed["bfv"]["ct_bytes"] == fresh["bfv"]["ct_bytes"] == cm.bfv_ct_bytes()
    prov = committed["bfv"]["provenance"]
    assert prov["declared_value"] == cm.BFV_CT_BYTES and "tenseal" in prov["declared_source"]
    if prov.get("confirmed_this_run"):
        r = prov["remeasured"]
        assert r["samples"] >= 8 and r["min"] <= r["mean"] <= r["max"]
        assert abs(r["mean"] - cm.BFV_CT_BYTES) <= cm.BFV_CT_REL_TOL * cm.BFV_CT_BYTES


def test_load_formulas_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="^FORMULAS_JSON_MISSING"):
        cm.load_formulas(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="^FORMULAS_JSON_INVALID"):
        cm.load_formulas(bad)
    bad.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="^FORMULAS_JSON_INVALID"):
        cm.load_formulas(bad)
    bad.write_text(json.dumps({"schema": cm.FORMULAS_SCHEMA, "bfv": {"ct_bytes": 0}, "shapes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="^FORMULAS_JSON_INVALID"):
        cm.load_formulas(bad)


def test_bfv_ciphertext_size_is_measured():
    """Re-measure the serialized BFV ciphertext with TenSEAL and confirm the declared value."""
    m = cm.measure_bfv_ct_bytes(samples=12, seed=1)
    assert m["samples"] == 12 and m["N"] == 8192 and m["t"] == 68720050177
    assert m["min"] <= m["mean"] <= m["max"]
    assert abs(m["mean"] - cm.BFV_CT_BYTES) <= cm.BFV_CT_REL_TOL * cm.BFV_CT_BYTES
    # the retired nominal figures of both languages are far outside the measured range
    assert m["max"] < 524288 < 655360
    with pytest.raises(ValueError, match="^SHAPE_PARAM"):
        cm.measure_bfv_ct_bytes(samples=0)

