"""Tests for opencrab2.highlevel.dealer (paper, Appendix B, DealerDelivery)."""

from __future__ import annotations

import numpy as np
import pytest

from opencrab2.errors import (ContractViolation, MaterialError, MaterialReuse, MaterialStall,
                              MaterialVersionMismatch)
from opencrab2.highlevel.contract import DEFAULT_CONTRACT, SMOLLM_CONTRACT
from opencrab2.highlevel import dealer as dealer_mod
from opencrab2.highlevel.dealer import (DEFAULT_T, HEADER_KEYS, MATERIAL_TYPES, MODE_INSECURE, MODE_ISOLATED,
                                        BeaverShare, Dealer, DealerClient, MatrixTripleShare, WMaskCall,
                                        estimate_total_bytes, material_estimate, material_estimate_detail,
                                        merge_estimates, tournament_shape)
from opencrab2.highlevel.ledger import AVAILABLE, BURNED, SPENT, MaterialLedger
from opencrab2.lowlevel import dcf, drelu, ring, sharing, trunc
from opencrab2.lowlevel import ring_to_field as r2f

C = DEFAULT_CONTRACT
KEYS = {0: bytes(range(32)), 1: bytes(range(32, 64))}
PROFILE = "SH2P-TrustedOffline-v1"


def binding(setup_id="su", model_version="v0", contract=C, session_id="s1"):
    return {"session_id": session_id, "setup_id": setup_id, "model_version": model_version,
            "numeric_contract_hash": contract.contract_hash(), "profile_id": PROFILE}


@pytest.fixture
def dealer():
    return Dealer(C, np.random.default_rng(1234), setup_id="su", model_version="v0")


@pytest.fixture
def clients(tmp_path):
    out = []
    for p in (0, 1):
        L = MaterialLedger(str(tmp_path / f"ledger{p}.jsonl"), binding())
        out.append(DealerClient(p, L, KEYS[p], C))
    yield out
    for c in out:
        c.ledger.close()


def deliver(dealer, clients, pairs):
    for h0, p0, h1, p1 in pairs:
        clients[0].ingest(*dealer.seal(0, h0, p0, KEYS[0]))
        clients[1].ingest(*dealer.seal(1, h1, p1, KEYS[1]))
    return len(pairs)


# --------------------------------------------------------------------------
# headers and sealing
# --------------------------------------------------------------------------
def test_header_schema_and_pairing(dealer):
    (h0, p0, h1, p1), = dealer.make_beaver((2, 3), 1)
    for h, p in ((h0, p0), (h1, p1)):
        assert set(HEADER_KEYS) <= set(h)
        assert h["profile_id"] == PROFILE and h["setup_id"] == "su" and h["model_version"] == "v0"
        assert h["numeric_contract_hash"] == C.contract_hash()
        assert h["material_type"] == "beaver" and h["count"] == 6 and h["bytes"] == len(p) == 3 * 6 * 8
        # the fixture dealer runs on a caller-seeded rng: isolated routing, but labelled insecure
        assert h["dealer_mode"] == MODE_ISOLATED and h["insecure"] is True and h["gate_id"] == "*"
        assert h["params"] == {"shape": [2, 3], "elements": 6}
        assert h["parts"] == ["a", "b", "c"]             # payload parts in payload order
    assert h0["recipient"] == 0 and h1["recipient"] == 1
    assert h0["material_id"] == h1["material_id"]
    (g0, _, _, _), = dealer.make_beaver((1,), 1)
    assert g0["material_id"] != h0["material_id"]
    assert dealer.produced_summary()["beaver"] == (7, 3 * 6 * 8 + 24)


def test_seal_checks_and_tags(dealer):
    (h0, p0, h1, p1), = dealer.make_beaver((4,), 1)
    hh, pp, tag = dealer.seal(0, h0, p0, KEYS[0])
    assert hh is h0 and pp == p0 and len(tag) == 64
    assert dealer.seal(0, h0, p0, KEYS[1])[2] != tag                 # key dependent
    with pytest.raises(MaterialError) as ei:
        dealer.seal(1, h0, p0, KEYS[1])
    assert ei.value.code == "BUNDLE_WRONG_RECIPIENT"
    with pytest.raises(MaterialError) as ei:
        dealer.seal(0, h0, p0[:-8], KEYS[0])
    assert ei.value.code == "BUNDLE_MALFORMED"


def test_modes_and_os_seeded_dealers():
    with pytest.raises(MaterialError):
        Dealer(C, mode="SOMETHING_ELSE")
    d1 = Dealer(C)
    d2 = Dealer(C)
    (h, p, _, _), = d1.make_beaver((8,), 1)
    (h2, p2, _, _), = d2.make_beaver((8,), 1)
    assert p != p2 and h["insecure"] is False
    assert not hasattr(d1, "seed") and not hasattr(d1, "master_seed")
    assert d1.insecure is False and d1.rng_from_os_entropy is True
    ins = Dealer(C, np.random.default_rng(0), MODE_INSECURE)
    (h, _, h1, _), = ins.make_beaver((8,), 1)
    assert h["insecure"] is True and h1["insecure"] is True and h["dealer_mode"] == MODE_INSECURE
    with pytest.raises(ContractViolation):
        Dealer("not a contract")   # type: ignore[arg-type]
    with pytest.raises(ContractViolation):
        Dealer(C, rng=1234)        # type: ignore[arg-type]


def test_caller_seeded_rng_is_labelled_insecure_unless_declared_os_entropy():
    # whoever seeds the generator can replay the dealer, so the label does not follow the mode alone
    a = Dealer(C, np.random.default_rng(0), MODE_ISOLATED)
    b = Dealer(C, np.random.default_rng(0), MODE_ISOLATED)
    (ha, pa, ha1, _), = a.make_beaver((8,), 1)
    (hb, pb, _, _), = b.make_beaver((8,), 1)
    assert pa == pb                                      # the replay the label warns about
    assert a.insecure is True and a.rng_from_os_entropy is False
    assert ha["insecure"] is True and ha1["insecure"] is True and ha["dealer_mode"] == MODE_ISOLATED
    for make in (lambda d: d.make_drelu(2), lambda d: d.make_trunc(2), lambda d: d.make_r2f(2),
                 lambda d: d.make_matrix_triples((1, 2), (2, 1), 1), lambda d: d.make_wmask("m", (2, 2)),
                 lambda d: d.make_wmask_calls("m", 1, 1), lambda d: d.make_vector_share("v", np.arange(3, dtype=np.uint64)),
                 lambda d: d.make_kary_select((2,), 1)):
        for h0, _p0, h1, _p1 in make(a):
            assert h0["insecure"] is True and h1["insecure"] is True
    # an OS-keyed ChaCha20 stream supplied by the caller is eligible
    from opencrab2.lowlevel import prg
    c = Dealer(C, prg.SecureRandom.from_os(), MODE_ISOLATED, rng_is_os_entropy=True)
    (hc, _, _, _), = c.make_beaver((8,), 1)
    assert c.insecure is False and hc["insecure"] is False and hc["dealer_mode"] == MODE_ISOLATED
    # a numpy Generator is never eligible, whatever seeded it: its state is recoverable from the bundles
    rng = prg.generator_from_bytes(prg.os_random_bytes(32))
    assert Dealer(C, rng, MODE_ISOLATED).insecure is True
    with pytest.raises(ContractViolation) as ei:
        Dealer(C, rng, MODE_ISOLATED, rng_is_os_entropy=True)
    assert ei.value.code == "DEALER_PARAMS"
    # a ChaCha20 stream keyed by the caller is replayable: insecure, and the flag cannot claim otherwise
    assert Dealer(C, prg.SecureRandom.from_seed(3), MODE_ISOLATED).insecure is True
    with pytest.raises(ContractViolation):
        Dealer(C, prg.SecureRandom.from_seed(3), MODE_ISOLATED, rng_is_os_entropy=True)
    # the insecure mode is never upgraded
    d = Dealer(C, prg.SecureRandom.from_os(), MODE_INSECURE, rng_is_os_entropy=True)
    assert d.insecure is True and d.make_beaver((1,), 1)[0][0]["insecure"] is True
    assert Dealer(C, None, MODE_ISOLATED, rng_is_os_entropy=False).insecure is False
    assert Dealer(C, None, MODE_INSECURE).insecure is True


# --------------------------------------------------------------------------
# every material type reconstructs its correlation
# --------------------------------------------------------------------------
def test_beaver_and_kary_select_reconstruct(dealer, clients):
    deliver(dealer, clients, dealer.make_beaver((5, 2), 3))
    deliver(dealer, clients, dealer.make_kary_select((7,), 1))
    c0, c1 = clients
    assert c0.stock()["beaver"] == (30, 720) and c0.stock()["kary_select"] == (7, 168)
    b0 = c0.take_beaver((3, 4))              # 12 elements span two bundles of 10
    b1 = c1.take_beaver((3, 4))
    assert isinstance(b0, BeaverShare) and b0.shape == (3, 4) and b0.a.shape == (3, 4)
    a = ring.add(b0.a, b1.a)
    b = ring.add(b0.b, b1.b)
    c = ring.add(b0.c, b1.c)
    assert np.array_equal(ring.mul(a, b), c)
    k0 = c0.take_kary_select((7,))
    k1 = c1.take_kary_select((7,))
    assert np.array_equal(ring.mul(ring.add(k0.a, k1.a), ring.add(k0.b, k1.b)), ring.add(k0.c, k1.c))
    assert c0.stock()["beaver"] == (18, 432) and c0.stock()["kary_select"] == (0, 0)


def test_matrix_triple_reconstructs(dealer, clients):
    deliver(dealer, clients, dealer.make_matrix_triples((3, 5), (5, 4), 2))
    c0, c1 = clients
    t0 = c0.take_matrix_triple((3, 5), (5, 4))
    t1 = c1.take_matrix_triple((3, 5), (5, 4))
    assert isinstance(t0, MatrixTripleShare)
    A = ring.add(t0.a, t1.a)
    B = ring.add(t0.b, t1.b)
    assert np.array_equal(ring.ring_matmul(A, B), ring.add(t0.c, t1.c))
    assert c0.stock()["matrix_triple"] == (1, 8 * (15 + 20 + 12))
    with pytest.raises(MaterialStall):
        c0.take_matrix_triple((3, 5), (5, 5))
    with pytest.raises(ContractViolation):
        dealer.make_matrix_triples((3, 5), (4, 4))


def test_wmask_mask_and_calls_reconstruct(dealer, clients):
    W = ring.rand_u64((6, 3), np.random.default_rng(9))
    deliver(dealer, clients, dealer.make_wmask("fc", W))
    deliver(dealer, clients, dealer.make_wmask_calls("fc", 2, 2))
    c0, c1 = clients
    B = ring.add(c0.wmask_mask_share("fc"), c1.wmask_mask_share("fc"))
    assert B.shape == (6, 3)
    # install protocol (Party): P1 sends B1, P0 returns E = W - B0 - B1; E is public and W never left P0
    E = ring.sub(W, B)
    assert np.array_equal(ring.add(E, B), W)
    for _ in range(2):
        w0 = c0.take_wmask_call("fc", 2)
        w1 = c1.take_wmask_call("fc", 2)
        assert isinstance(w0, WMaskCall) and w0.mat_id == "fc"
        A = ring.add(w0.a, w1.a)
        assert np.array_equal(ring.ring_matmul(A, B), ring.add(w0.c, w1.c))
    with pytest.raises(MaterialStall):
        c0.take_wmask_call("fc", 2)
    with pytest.raises(MaterialStall) as ei:
        c0.take_wmask_call("nope", 2)
    assert ei.value.code == "MATERIAL_STALL"
    with pytest.raises(MaterialError) as ei:
        dealer.make_wmask_calls("nope", 2)
    assert ei.value.code == "WMASK_MASK_MISSING"
    # a second mask bundle for the same mat_id (same model version) is refused by the client
    with pytest.raises(MaterialReuse):
        deliver(dealer, clients, dealer.make_wmask("fc", (6, 3)))
    # the mask is spent exactly once in the ledger although it is read twice
    assert c0.wmask_mask_share("fc") is not None
    assert c0.ledger.counts()[SPENT] == 3
    assert c0.wmask_mask_version("fc")["model_version"] == "v0"


def test_fresh_mask_per_model_version_prevents_E_difference_leak(dealer):
    rng = np.random.default_rng(3)
    W0 = ring.rand_u64((4, 4), rng)
    W1 = ring.rand_u64((4, 4), rng)
    (h0, p0, h1, p1), = dealer.make_wmask("lm", W0)
    dealer.bind(model_version="v1")
    (g0, q0, g1, q1), = dealer.make_wmask("lm", W1)
    B0 = ring.add(np.frombuffer(p0, "<u8").astype(np.uint64), np.frombuffer(p1, "<u8").astype(np.uint64)).reshape(4, 4)
    B1 = ring.add(np.frombuffer(q0, "<u8").astype(np.uint64), np.frombuffer(q1, "<u8").astype(np.uint64)).reshape(4, 4)
    assert not np.array_equal(B0, B1)
    E0 = ring.sub(W0, B0)
    E1 = ring.sub(W1, B1)
    assert not np.array_equal(ring.sub(E1, E0), ring.sub(W1, W0))
    assert h0["model_version"] == "v0" and g0["model_version"] == "v1"


def test_vector_share(dealer, clients):
    v = ring.encode(np.array([0.5, -1.25, 3.0]), C)
    deliver(dealer, clients, dealer.make_vector_share("ln.beta", v))
    c0, c1 = clients
    assert np.array_equal(ring.add(c0.take_vector_share("ln.beta"), c1.take_vector_share("ln.beta")), v)
    assert np.array_equal(c0.take_vector_share("ln.beta"), c0.take_vector_share("ln.beta"))   # fixed value, reusable
    assert c0.ledger.counts()[SPENT] == 1
    with pytest.raises(MaterialStall):
        c0.take_vector_share("missing")


def test_drelu_material_reconstructs_sign(dealer, clients):
    deliver(dealer, clients, dealer.make_drelu(16))
    deliver(dealer, clients, dealer.make_drelu(4, with_product=False))
    c0, c1 = clients
    d0 = c0.take_drelu(9)
    d1 = c1.take_drelu(9)
    assert isinstance(d0, drelu.DReLUMaterial) and d0.M == 9 and d0.party == 0 and d1.party == 1
    xr = np.array([-3.0, 2.0, 0.0, -0.5, 1.0, 5.0, -7.0, 1e-3, -1e-3])
    x = ring.encode(xr, C)
    y = ring.add(x, ring.add(d0.r_share, d1.r_share))
    b0, rb0 = drelu.drelu_eval(d0, y)
    b1, rb1 = drelu.drelu_eval(d1, y)
    assert np.array_equal(ring.add(b0, b1), (xr >= 0).astype(np.uint64))
    relu = ring.add(drelu.relu_from_drelu(y, b0, rb0), drelu.relu_from_drelu(y, b1, rb1))
    # exact in the ring: ReLU of the encoded value (1e-3 encodes to 66 units, i.e. 1.0071e-3)
    assert np.array_equal(ring.to_signed(relu), np.maximum(ring.to_signed(x), 0))
    e0 = c0.take_drelu(4, with_product=False)
    e1 = c1.take_drelu(4, with_product=False)
    assert e0.key_p.P == 1
    y = ring.add(x[:4], ring.add(e0.r_share, e1.r_share))
    s0, _ = drelu.drelu_eval(e0, y)
    s1, _ = drelu.drelu_eval(e1, y)
    assert np.array_equal(ring.add(s0, s1), (xr[:4] >= 0).astype(np.uint64))
    assert c0.stock()["drelu"] == (7, 7 * 2200)
    with pytest.raises(MaterialStall):
        c0.take_drelu(8)
    with pytest.raises(MaterialStall):
        c0.take_drelu(1, n_bits=16)
    # a bit-only request is served from the product-payload pool when its own pool is empty
    f0 = c0.take_drelu(2, with_product=False)
    f1 = c1.take_drelu(2, with_product=False)
    assert f0.key_p.P == 2 and f1.key_p.P == 2
    y = ring.add(x[:2], ring.add(f0.r_share, f1.r_share))
    s0, _ = drelu.drelu_eval(f0, y)
    s1, _ = drelu.drelu_eval(f1, y)
    assert np.array_equal(ring.add(s0, s1), (xr[:2] >= 0).astype(np.uint64))
    assert c0.stock()["drelu"] == (5, 5 * 2200)


def test_estimate_detail_splits_by_dealer_parameters():
    detail = material_estimate_detail("softmax_rows", (2, 8), C, k_max=2)
    agg = material_estimate("softmax_rows", (2, 8), C, k_max=2)
    by_type = {}
    for ent in detail:
        c, b = by_type.get(ent["material_type"], (0, 0))
        by_type[ent["material_type"]] = (c + ent["count"], b + ent["bytes"])
    assert by_type == agg
    drelus = {tuple(sorted(e["params"].items())): e for e in detail if e["material_type"] == "drelu"}
    assert set(drelus) == {(("n_bits", 32), ("with_product", True)), (("n_bits", 32), ("with_product", False))}
    # binary max tournament: 7 pairwise comparisons per row without payload (Beaver selects, no payload DReLU)
    assert drelus[(("n_bits", 32), ("with_product", True))]["count"] == 16             # exp clamp only
    assert drelus[(("n_bits", 32), ("with_product", False))]["count"] == 2 * 7 + 2 * 11  # max pairs + recip thresholds
    tr = material_estimate_detail("layernorm_rows", (1, 4), C)
    assert [e for e in tr if e["material_type"] == "trunc_faithful"][0]["params"] == {"f": 16, "L": 40}
    r = material_estimate_detail("ring_to_field", (3,), C, t=17)
    assert r == [{"material_type": "ring_to_field", "params": {"t": 17}, "count": 3, "bytes": 3 * r2f.r2f_bytes_per_element(17)}]
    assert material_estimate_detail("beaver_mul", (2,), C) == [{"material_type": "beaver", "params": {}, "count": 2, "bytes": 48}]


def test_trunc_material_reconstructs_faithful_floor(dealer, clients):
    deliver(dealer, clients, dealer.make_trunc(8))
    c0, c1 = clients
    t0 = c0.take_trunc(6)
    t1 = c1.take_trunc(6)
    assert isinstance(t0, trunc.TruncMaterial) and t0.f == 16 and t0.L == 40 and t0.M == 6
    xs = ring.encode(np.array([-13.0, 1.25, 1000.5, -0.75, 3.0, -2.0]), C)
    x0, x1 = sharing.share(xs, np.random.default_rng(5))
    y0 = ring.add(trunc.trunc_offset(x0, 0, 40), t0.r_share)
    y1 = ring.add(trunc.trunc_offset(x1, 1, 40), t1.r_share)
    y = ring.add(y0, y1)
    z = ring.add(trunc.trunc_faithful_eval(t0, y, 0), trunc.trunc_faithful_eval(t1, y, 1))
    assert np.array_equal(ring.to_signed(z), ring.to_signed(xs) >> np.int64(16))
    assert c0.stock()["trunc_faithful"] == (2, 2 * 2064)
    with pytest.raises(ContractViolation):
        dealer.make_trunc(2, f=40, L=40)


def test_r2f_material_reconstructs_signed_mod(dealer, clients):
    t = DEFAULT_T
    deliver(dealer, clients, dealer.make_r2f(6, t))
    c0, c1 = clients
    m0 = c0.take_r2f(5, t)
    m1 = c1.take_r2f(5, t)
    assert isinstance(m0, r2f.R2FMaterial) and m0.t == t and m0.M == 5
    u = np.array([2 ** 64 - 1, 5, 2 ** 63, 2 ** 63 - 1, 0], dtype=np.uint64)   # -1, 5, -2^63, 2^63-1, 0
    a0, a1 = sharing.share(u, np.random.default_rng(2))
    y = ring.add(ring.add(a0, m0.r_share), ring.add(a1, m1.r_share))
    z = sharing.reconstruct_mod(r2f.r2f_eval(m0, y, 0), r2f.r2f_eval(m1, y, 1), t)
    assert np.array_equal(z, r2f.signed_mod(u, t))
    assert c0.stock()["ring_to_field"] == (1, 3272)
    with pytest.raises(MaterialStall):
        c0.take_r2f(1, t=17)


# --------------------------------------------------------------------------
# ingest rejections
# --------------------------------------------------------------------------
def test_ingest_rejections(dealer, clients, tmp_path):
    c0, c1 = clients
    (h0, p0, h1, p1), = dealer.make_beaver((4,), 1)
    sealed1 = dealer.seal(1, h1, p1, KEYS[1])
    # wrong key
    with pytest.raises(MaterialError) as ei:
        c0.ingest(*dealer.seal(0, h0, p0, KEYS[1]))
    assert ei.value.code == "BUNDLE_BAD_TAG"
    # cross recipient under the correct key of the ingesting party
    with pytest.raises(MaterialError) as ei:
        c1.ingest(*dealer.seal(0, h0, p0, KEYS[1]))
    assert ei.value.code == "BUNDLE_WRONG_RECIPIENT"
    # tampered payload / header
    hh, pp, tg = sealed1
    with pytest.raises(MaterialError) as ei:
        c1.ingest(hh, pp[:-1] + bytes([pp[-1] ^ 1]), tg)
    assert ei.value.code == "BUNDLE_BAD_TAG"
    with pytest.raises(MaterialError) as ei:
        c1.ingest({**hh, "count": 5}, pp, tg)
    assert ei.value.code == "BUNDLE_BAD_TAG"
    # malformed header
    with pytest.raises(MaterialError) as ei:
        c1.ingest({"recipient": 1}, pp, tg)
    assert ei.value.code == "BUNDLE_MALFORMED"
    # payload that does not match the declared shape, resealed by the dealer
    bad_h = {**hh, "bytes": len(pp) - 8}
    with pytest.raises(MaterialError) as ei:
        c1.ingest(*dealer.seal(1, bad_h, pp[:-8], KEYS[1]))
    assert ei.value.code == "BUNDLE_PAYLOAD_MISMATCH"
    # nothing above left a trace
    assert c1.ledger.counts()[AVAILABLE] == 0 and c1.stock()["beaver"] == (0, 0)
    # the good one
    c1.ingest(*sealed1)
    with pytest.raises(MaterialReuse) as ei:
        c1.ingest(*sealed1)
    assert ei.value.code == "BUNDLE_DUPLICATE_MATERIAL_ID"
    # contract mismatch
    other = Dealer(SMOLLM_CONTRACT, np.random.default_rng(0), setup_id="su", model_version="v0")
    (g0, q0, _, _), = other.make_beaver((2,), 1)
    with pytest.raises(MaterialVersionMismatch) as ei:
        c0.ingest(*other.seal(0, g0, q0, KEYS[0]))
    assert ei.value.code == "BUNDLE_CONTRACT_MISMATCH"
    # foreign setup / model version is refused by the ledger binding
    foreign = Dealer(C, np.random.default_rng(0), setup_id="su", model_version="v1")
    (g0, q0, _, _), = foreign.make_beaver((2,), 1)
    with pytest.raises(MaterialVersionMismatch):
        c0.ingest(*foreign.seal(0, g0, q0, KEYS[0]))
    assert c0.stock()["beaver"] == (0, 0)
    # wrong profile id
    with pytest.raises(MaterialVersionMismatch) as ei:
        (g0, q0, _, _), = dealer.make_beaver((2,), 1)
        g0 = {**g0, "profile_id": "OTHER"}
        c0.ingest(*dealer.seal(0, g0, q0, KEYS[0]))
    assert ei.value.code == "BUNDLE_PROFILE_MISMATCH"


def test_insecure_mode_is_labelled_and_can_be_refused(tmp_path):
    ins = Dealer(C, np.random.default_rng(0), MODE_INSECURE, setup_id="su", model_version="v0")
    L = MaterialLedger(str(tmp_path / "l.jsonl"), binding())
    c = DealerClient(0, L, KEYS[0], C)
    assert c.security_eligible
    (h0, p0, _, _), = ins.make_beaver((2,), 1)
    c.ingest(*ins.seal(0, h0, p0, KEYS[0]))
    assert not c.security_eligible and c.insecure_material_seen
    strict = DealerClient(1, MaterialLedger(str(tmp_path / "s.jsonl"), binding()), KEYS[1], C, allow_insecure=False)
    (_, _, h1, p1), = ins.make_beaver((2,), 1)
    with pytest.raises(MaterialError) as ei:
        strict.ingest(*ins.seal(1, h1, p1, KEYS[1]))
    assert ei.value.code == "BUNDLE_INSECURE_MODE"
    L.close()
    strict.ledger.close()


def test_default_client_refuses_insecure_bundles_outside_a_test_harness(tmp_path, monkeypatch):
    # the default DealerClient (what launch.py, model_server.py and sidecar.py build) fails closed.
    seeded = Dealer(C, np.random.default_rng(3), MODE_ISOLATED, setup_id="su", model_version="v0")
    shared = Dealer(C, np.random.default_rng(3), MODE_INSECURE, setup_id="su", model_version="v0")
    osd = Dealer(C, setup_id="su", model_version="v0")
    # building an insecure dealer in this process turned the test-only switch on (in-process harness)
    assert dealer_mod.insecure_material_allowed() is True
    assert dealer_mod.insecure_material_allowed(False) is False

    def fresh(name, **kw):
        return DealerClient(0, MaterialLedger(str(tmp_path / f"{name}.jsonl"), binding()), KEYS[0], C, **kw)

    # a party process hosts no dealer and has no environment switch: emulate it
    monkeypatch.setitem(dealer_mod._TEST_ONLY_STATE, "in_process_insecure_dealer", False)
    monkeypatch.delenv(dealer_mod.ENV_ALLOW_INSECURE, raising=False)
    assert dealer_mod.insecure_material_allowed() is False
    party = fresh("party")
    assert party.allow_insecure is None
    for d in (seeded, shared):
        (h0, p0, _, _), = d.make_beaver((2,), 1)
        with pytest.raises(MaterialError) as ei:
            party.ingest(*d.seal(0, h0, p0, KEYS[0]))
        assert ei.value.code == "BUNDLE_INSECURE_MODE"
    assert party.ledger.counts()[AVAILABLE] == 0 and party.stock()["beaver"] == (0, 0)     # nothing left a trace
    assert party.security_eligible
    # a header that does not positively say "isolated and not insecure" is treated as insecure
    (h0, p0, _, _), = osd.make_beaver((2,), 1)
    for tampered in ({**h0, "insecure": 0}, {**h0, "insecure": None}, {**h0, "dealer_mode": "SOMETHING_ELSE"},
                     {k: v for k, v in h0.items() if k != "insecure"} | {"insecure": "false"}):
        with pytest.raises(MaterialError) as ei:
            party.ingest(*osd.seal(0, tampered, p0, KEYS[0]))
        assert ei.value.code == "BUNDLE_INSECURE_MODE"
    # material of an OS-seeded isolated dealer is accepted and keeps the record eligible
    party.ingest(*osd.seal(0, h0, p0, KEYS[0]))
    assert party.security_eligible and party.stock()["beaver"] == (2, 48)
    # the three ways to turn the test-only switch on
    (g0, q0, _, _), = shared.make_beaver((2,), 1)
    explicit = fresh("explicit", allow_insecure=True)
    explicit.ingest(*shared.seal(0, g0, q0, KEYS[0]))
    assert not explicit.security_eligible
    monkeypatch.setenv(dealer_mod.ENV_ALLOW_INSECURE, "1")
    (g0, q0, _, _), = shared.make_beaver((2,), 1)
    env_client = fresh("env")
    env_client.ingest(*shared.seal(0, g0, q0, KEYS[0]))
    assert not env_client.security_eligible
    monkeypatch.setenv(dealer_mod.ENV_ALLOW_INSECURE, "0")
    (g0, q0, _, _), = shared.make_beaver((2,), 1)
    with pytest.raises(MaterialError):
        fresh("env0").ingest(*shared.seal(0, g0, q0, KEYS[0]))
    monkeypatch.delenv(dealer_mod.ENV_ALLOW_INSECURE)
    Dealer(C, np.random.default_rng(9), MODE_INSECURE)          # an in-process insecure dealer: harness detected
    assert dealer_mod.insecure_material_allowed() is True
    for c in (party, explicit, env_client):
        c.ledger.close()


def test_bundle_parts_are_checked(dealer, clients):
    c0, _ = clients
    (h0, p0, _, _), = dealer.make_drelu(2)
    assert h0["parts"] == ["drelu_material"]
    assert dealer.make_trunc(1)[0][0]["parts"] == ["trunc_material"]
    assert dealer.make_r2f(1)[0][0]["parts"] == ["r2f_material"]
    assert dealer.make_matrix_triples((1, 2), (2, 1), 1)[0][0]["parts"] == ["A", "B", "C"]
    assert dealer.make_wmask("m", (2, 2))[0][0]["parts"] == ["B_share"]
    assert dealer.make_wmask_calls("m", 1, 1)[0][0]["parts"] == ["A", "C"]
    assert dealer.make_vector_share("v", np.arange(3, dtype=np.uint64))[0][0]["parts"] == ["share"]
    assert set(dealer_mod.PAYLOAD_PARTS) == set(MATERIAL_TYPES)
    with pytest.raises(MaterialError) as ei:
        c0.ingest(*dealer.seal(0, {**h0, "parts": ["a", "b", "c"]}, p0, KEYS[0]))
    assert ei.value.code == "BUNDLE_MALFORMED"
    # a header without the optional parts field (older writer) is still accepted
    c0.ingest(*dealer.seal(0, {k: v for k, v in h0.items() if k != "parts"}, p0, KEYS[0]))
    assert c0.stock()["drelu"][0] == 2


# --------------------------------------------------------------------------
# dealer dialogue frames
# --------------------------------------------------------------------------
def test_dialogue_frames_and_auth_tag(dealer, clients):
    import hashlib
    import hmac as _hmac
    from opencrab2.highlevel.channel import decode_payload, encode_payload
    from opencrab2.highlevel.dealer import (AUTH_NONCE_BYTES, auth_frame, dealer_auth_tag, hello_frame,
                                            parse_bundle_frame, parse_challenge, raise_if_error_frame, req_frame,
                                            verify_dealer_auth_tag)
    assert hello_frame(1, "su", "ab" * 32) == {"type": "HELLO", "recipient": 1, "setup_id": "su", "contract_hash": "ab" * 32}
    assert hello_frame(0, None, None) == {"type": "HELLO", "recipient": 0, "setup_id": "", "contract_hash": ""}
    with pytest.raises(MaterialError):
        hello_frame(2, "su", "x")
    # AUTH tag = hex(HMAC-SHA256(key_recipient, nonce)) over the 32 raw nonce bytes
    nonce = bytes(range(200, 232))
    assert AUTH_NONCE_BYTES == 32 and len(nonce) == 32
    tag = dealer_auth_tag(KEYS[0], nonce)
    assert tag == _hmac.new(KEYS[0], nonce, hashlib.sha256).hexdigest()
    # pinned vector: key = bytes 00..1f, nonce = bytes c8..e7
    assert tag == "374c89e32be4d971465bfdd351080c1a42a12225bfa9822dd1cd22abbbd0c3a9"
    assert auth_frame(KEYS[0], nonce) == {"type": "AUTH", "tag": tag}
    assert verify_dealer_auth_tag(KEYS[0], nonce, tag)
    assert not verify_dealer_auth_tag(KEYS[1], nonce, tag)                       # the key binds the recipient
    assert not verify_dealer_auth_tag(KEYS[0], bytes(32), tag)
    for bad in (tag[:-1], tag.upper() + "x", "zz" * 32, None, 5, b"x" * 64):
        assert not verify_dealer_auth_tag(KEYS[0], nonce, bad)
    for bad_nonce in (b"", bytes(31), bytes(33), "00" * 32):
        with pytest.raises(MaterialError) as ei:
            dealer_auth_tag(KEYS[0], bad_nonce)      # never sign anything but a 32-byte challenge
        assert ei.value.code == "DEALER_BAD_MESSAGE"
    with pytest.raises(MaterialError):
        dealer_auth_tag(b"", nonce)
    # CHALLENGE parsing
    assert parse_challenge({"type": "CHALLENGE", "nonce": nonce, "dealer_mode": MODE_ISOLATED}) == nonce
    for bad in ({"type": "CHALLENGE"}, {"type": "CHALLENGE", "nonce": "00" * 32}, {"type": "CHALLENGE", "nonce": b"short"},
                {"type": "WELCOME", "nonce": nonce}, None, [1]):
        with pytest.raises(MaterialError) as ei:
            parse_challenge(bad)
        assert ei.value.code == "DEALER_BAD_MESSAGE"
    for code in ("AUTH_FAILED", "CONTRACT_MISMATCH", "SETUP_MISMATCH", "DEALER_UNKNOWN_RECIPIENT"):
        with pytest.raises(MaterialError) as ei:
            parse_challenge({"type": "ERROR", "code": code})
        assert ei.value.code == code
        with pytest.raises(MaterialError) as ei:
            raise_if_error_frame({"type": "ERROR", "code": code, "message": f"{code}: detail"})
        assert ei.value.code == code and str(ei.value) == f"{code}: detail"
    with pytest.raises(MaterialError) as ei:
        parse_challenge({"type": "ERROR", "code": "not a code; rm -rf"})
    assert ei.value.code == "DEALER_ERROR"
    raise_if_error_frame({"type": "BUNDLE"})
    # REQ
    assert req_frame("beaver", {"shape": [4]}, 2) == {"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 2}
    assert req_frame("drelu", None, 1, "g", "r1") == {"type": "REQ", "material_type": "drelu", "params": {}, "count": 1,
                                                       "gate_id": "g", "request_id": "r1"}
    with pytest.raises(MaterialError) as ei:
        req_frame("beaver", {}, 0)
    assert ei.value.code == "DEALER_BAD_REQUEST"
    # every dialogue frame survives the binding codec unchanged
    for frame in (hello_frame(1, "su", "ab" * 32), {"type": "CHALLENGE", "nonce": nonce}, auth_frame(KEYS[0], nonce),
                  req_frame("beaver", {"shape": [4]}, 2, "g", "r")):
        assert decode_payload(encode_payload(frame)) == frame
    # BUNDLE{header, tag} followed by one bytes frame
    (h0, p0, _, _), = dealer.make_beaver((3,), 1)
    hdr, pay, tg = dealer.seal(0, h0, p0, KEYS[0])
    msg = decode_payload(encode_payload({"type": "BUNDLE", "header": hdr, "tag": tg}))
    assert parse_bundle_frame(msg) == (hdr, tg, 0)
    assert parse_bundle_frame({"type": "BUNDLE", "header": hdr, "tag": tg, "remaining": 3})[2] == 3
    for bad in ({"type": "BUNDLE", "header": hdr}, {"type": "BUNDLE", "header": [], "tag": tg},
                {"type": "BUNDLE", "header": hdr, "tag": tg, "remaining": -1},
                {"type": "BUNDLE", "header": hdr, "tag": tg, "remaining": True}, {"type": "END"}, b"x"):
        with pytest.raises(MaterialError) as ei:
            parse_bundle_frame(bad)
        assert ei.value.code == "DEALER_BAD_MESSAGE"
    with pytest.raises(MaterialError) as ei:
        parse_bundle_frame({"type": "ERROR", "code": "MATERIAL_STALL"})
    assert ei.value.code == "MATERIAL_STALL"
    c0, c1 = clients
    with pytest.raises(MaterialError) as ei:
        c0.ingest_frame(msg, "not bytes")
    assert ei.value.code == "DEALER_BAD_MESSAGE"
    with pytest.raises(MaterialError) as ei:
        c1.ingest_frame(msg, decode_payload(encode_payload(pay)))               # foreign recipient
    assert ei.value.code == "BUNDLE_BAD_TAG"
    c0.ingest_frame(msg, decode_payload(encode_payload(pay)))
    assert c0.stock()["beaver"] == (3, 72)


# --------------------------------------------------------------------------
# ledger coupling
# --------------------------------------------------------------------------
def test_takes_spend_in_the_ledger_before_returning(dealer, clients):
    c0, _ = clients
    deliver(dealer, clients, dealer.make_beaver((10,), 2))
    c0.set_context("mlp.fc1", 4)
    ids = c0.ledger.ids_in_state(AVAILABLE)
    assert len(ids) == 2
    c0.take_beaver((4,))
    assert c0.ledger.state(ids[0]) == SPENT and c0.ledger.state(ids[1]) == AVAILABLE
    c0.take_beaver((4,))                      # remainder of the first bundle: no second spend
    assert c0.ledger.counts()[SPENT] == 1
    c0.take_beaver((4,))                      # 2 left in bundle 1 + 2 from bundle 2
    assert c0.ledger.counts()[SPENT] == 2 and c0.stock()["beaver"] == (8, 192)
    import json
    with open(c0.ledger.journal_path, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    reserved = [r for r in recs if r.get("state") == "RESERVED"]
    assert all(r["gate_id"] == "mlp.fc1" and r["epoch"] == 4 for r in reserved)
    assert c0.consumed_bytes == 24 * 12 and c0.ingested_bundles == 2
    with pytest.raises(MaterialStall):
        c0.take_beaver((9,))
    assert c0.stalls == 1
    with pytest.raises(ContractViolation):
        c0.take_beaver((0,))


def test_take_refused_when_ledger_needs_recovery(tmp_path, dealer):
    L = MaterialLedger(str(tmp_path / "l.jsonl"), binding())
    c = DealerClient(0, L, KEYS[0], C)
    (h0, p0, _, _), = dealer.make_beaver((3,), 1)
    c.ingest(*dealer.seal(0, h0, p0, KEYS[0]))
    L.reserve([h0["material_id"]], "g", 0)   # a reservation left behind by a crash
    L.close()
    L2 = MaterialLedger(str(tmp_path / "l.jsonl"), binding())
    c2 = DealerClient(0, L2, KEYS[0], C)
    (h, p, _, _), = dealer.make_beaver((3,), 1)
    with pytest.raises(Exception) as ei:
        c2.ingest(*dealer.seal(0, h, p, KEYS[0]))
    assert "LEDGER_RECOVERY_REQUIRED" in str(ei.value)
    assert L2.recover()["reserved_burned"] == 1
    c2.ingest(*dealer.seal(0, h, p, KEYS[0]))
    assert c2.stock()["beaver"] == (3, 72)
    L2.close()


def test_discard_burns_untouched_bundles(dealer, clients):
    c0, _ = clients
    deliver(dealer, clients, dealer.make_beaver((5,), 2))
    deliver(dealer, clients, dealer.make_drelu(3))
    deliver(dealer, clients, dealer.make_matrix_triples((2, 2), (2, 2), 1))
    c0.take_beaver((2,))
    wasted = c0.discard(reason="PLAN_SWITCH")
    assert wasted == 24 * 8 + 3 * 2200 + 8 * 12
    assert c0.stock()["beaver"] == (0, 0) and c0.stock()["drelu"] == (0, 0) and c0.stock()["matrix_triple"] == (0, 0)
    counts = c0.ledger.counts()
    assert counts[BURNED] == 3 and counts[SPENT] == 1 and counts[AVAILABLE] == 0
    assert c0.wasted_bytes == wasted
    with pytest.raises(MaterialStall):
        c0.take_beaver((1,))


def test_client_param_validation(tmp_path):
    L = MaterialLedger(str(tmp_path / "l.jsonl"), binding())
    with pytest.raises(MaterialError):
        DealerClient(2, L, KEYS[0], C)
    with pytest.raises(ContractViolation):
        DealerClient(0, L, KEYS[0], "x")   # type: ignore[arg-type]
    L.close()


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------
def test_make_dispatcher(dealer):
    assert dealer.make("beaver", {"shape": [3]}, 2)[0][0]["material_type"] == "beaver"
    assert dealer.make("kary_select", {"shape": [3]}, 1)[0][0]["material_type"] == "kary_select"
    assert dealer.make("matrix_triple", {"shape_a": [2, 3], "shape_b": [3, 1]}, 1)[0][0]["params"]["shape_b"] == [3, 1]
    out = dealer.make("drelu", {"M": 2, "with_product": False}, 2)
    assert len(out) == 2 and out[0][0]["params"] == {"n_bits": 32, "with_product": False, "P": 1}
    assert dealer.make("trunc_faithful", {"M": 2}, 1)[0][0]["params"] == {"f": 16, "L": 40}
    assert dealer.make("ring_to_field", {"M": 2, "t": 17}, 1)[0][0]["params"] == {"t": 17}
    assert dealer.make("wmask_mask", {"mat_id": "w", "shape": [2, 2]})[0][0]["params"]["mat_id"] == "w"
    assert len(dealer.make("wmask_call", {"mat_id": "w", "n_rows": 1}, 3)) == 3
    assert dealer.make("vector_share", {"name": "b", "values": [1, 2, 3]})[0][0]["count"] == 3
    with pytest.raises(MaterialError):
        dealer.make("unknown", {}, 1)
    assert set(dealer.produced_summary()) <= set(MATERIAL_TYPES)


# --------------------------------------------------------------------------
# material_estimate
# --------------------------------------------------------------------------
def test_estimate_elementwise_and_linear():
    assert material_estimate("beaver_mul", (4, 5), C) == {"beaver": (20, 480)}
    assert material_estimate("linear_generic", (2, 768, 2304), C) == {"matrix_triple": (2 * 768 + 768 * 2304 + 2 * 2304, 8 * (2 * 768 + 768 * 2304 + 2 * 2304))}
    assert material_estimate("linear_wmask", (2, 768, 2304), C) == {"wmask_call": (2 * 768 + 2 * 2304, 8 * (2 * 768 + 2 * 2304))}
    assert material_estimate("embedding_wmask", (1, 50257, 768), C) == {"wmask_call": (50257 + 768, 8 * (50257 + 768))}
    assert material_estimate("embedding_generic", (1, 50257, 768), C)["matrix_triple"][0] == 50257 + 50257 * 768 + 768
    assert material_estimate("install_wmask", (768, 50257), C) == {"wmask_mask": (768 * 50257, 8 * 768 * 50257)}
    assert material_estimate("matmul_shared", ((12, 4, 64), (12, 64, 4)), C) == {"matrix_triple": (12 * (256 + 256 + 16), 8 * 12 * 528)}


def test_estimate_gate_bytes_match_lowlevel_formulas():
    assert material_estimate("drelu", (10,), C) == {"drelu": (10, 10 * drelu.drelu_bytes_per_element(32, False))}
    assert material_estimate("relu", (10,), C) == {"drelu": (10, 10 * 2200)}
    assert material_estimate("trunc_faithful", (3, 3), C) == {"trunc_faithful": (9, 9 * 2064)}
    assert material_estimate("ring_to_field", (4,), C) == {"ring_to_field": (4, 4 * 3272)}
    assert material_estimate("a2h_exact", (4,), C, t=17) == {"ring_to_field": (4, 4 * r2f.r2f_bytes_per_element(17))}
    assert material_estimate("drelu", (1,), C, n_bits=64)["drelu"][1] == drelu.drelu_bytes_per_element(64, False)


def test_estimate_matches_dealer_production(dealer):
    (h0, p0, _, _), = dealer.make_drelu(12)
    est = material_estimate("relu", (12,), C)
    assert est["drelu"][0] == h0["count"] == 12
    mat = drelu.DReLUMaterial.from_bytes(p0)
    assert est["drelu"][1] == mat.nbytes() == 12 * 2200                          # the charged size (declared size formula)
    # the payload embeds the two DCF keys in the binding wire layout:
    # one u64 word per control bit, i.e. 15 bytes per level and key more than the charged formula
    assert h0["bytes"] == len(p0) == mat.wire_nbytes()
    assert len(p0) == drelu.HEADER_BYTES + 24 * 12 + 2 * dcf.wire_bytes(12, 32, 2)
    assert len(p0) - est["drelu"][1] == drelu.HEADER_BYTES + 2 * dcf.HEADER_BYTES + 2 * 12 * 15 * 32
    (h0, p0, _, _), = dealer.make_trunc(5)
    tm = trunc.TruncMaterial.from_bytes(p0)
    assert material_estimate("trunc_faithful", (5,), C)["trunc_faithful"][1] == tm.nbytes() == 5 * 2064
    assert h0["bytes"] == len(p0) == tm.wire_nbytes() == trunc.HEADER_BYTES + 16 * 5 + dcf.wire_bytes(5, 64, 1) + dcf.wire_bytes(5, 16, 1)
    (h0, p0, _, _), = dealer.make_r2f(3)
    rm = r2f.R2FMaterial.from_bytes(p0)
    assert material_estimate("ring_to_field", (3,), C)["ring_to_field"][1] == rm.nbytes() == 3 * 3272
    assert h0["bytes"] == len(p0) == rm.wire_nbytes() == r2f.HEADER_BYTES + 24 * 3 + 2 * dcf.wire_bytes(3, 64, 1)
    (h0, p0, _, _), = dealer.make_beaver((5,), 1)
    assert material_estimate("beaver_mul", (5,), C)["beaver"][1] == len(p0)


def test_estimate_compositions():
    R, n = 4, 16
    sm = material_estimate("softmax_rows", (R, n), C, k_max=4)
    mx = material_estimate("max_rows", (R, n), C, k=4)
    ex = material_estimate("exp_neg", (R, n), C)
    rc = material_estimate("recip", (R,), C)
    expect = merge_estimates(mx, ex, rc, {"beaver": (R * n, 24 * R * n)})
    assert sm == expect
    assert rc == {"drelu": (11 * R, 11 * R * 1672), "beaver": (11 * R, 24 * 11 * R)}
    assert ex == {"drelu": (R * n, R * n * 2200), "beaver": (9 * R * n, 24 * 9 * R * n)}
    ln = material_estimate("layernorm_rows", (R, 8), C)
    rs = material_estimate("rsqrt", (R,), C)
    assert ln["trunc_faithful"] == (R * 8, R * 8 * 2064)
    assert ln["drelu"] == rs["drelu"] == (15 * R, 15 * R * 1672)
    assert ln["beaver"][0] == rs["beaver"][0] + 3 * R * 8 and rs["beaver"][0] == 13 * R
    assert "trunc_faithful" not in material_estimate("layernorm_rows", (R, 8), C, exact_square=False)
    assert material_estimate("rmsnorm_rows", (R, 8), C) == ln
    assert material_estimate("gelu", (2, 3), C) == {"drelu": (12, 12 * 1672), "beaver": (14 * 6, 24 * 14 * 6)}
    assert material_estimate("silu", (6,), C) == material_estimate("gelu", (6,), C)
    assert material_estimate("poly_horner", (6,), C, degree=5) == {"beaver": (30, 720)}
    assert estimate_total_bytes(ln) == sum(b for _, b in ln.values())
    with pytest.raises(MaterialError) as ei:
        material_estimate("teleport", (1,), C)
    assert ei.value.code == "MATERIAL_ESTIMATE_UNKNOWN_OP"
    with pytest.raises(ContractViolation):
        material_estimate("beaver_mul", (1,), "nope")   # type: ignore[arg-type]


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 9, 64, 65, 50257])
def test_tournament_shape(n):
    import math
    for k in (2, 4, 8):
        rounds, cmp = tournament_shape(n, k)
        assert cmp == n - 1
        assert rounds == (0 if n == 1 else math.ceil(math.log(n, k) - 1e-12))
    assert tournament_shape(50257, 8) == (6, 50256) and tournament_shape(50257, 2) == (16, 50256)
    with pytest.raises(ContractViolation):
        tournament_shape(n, 3)


def test_estimate_argmax_k_prices_the_fused_realization():
    """material_estimate charges exactly the gates party.Party._tournament issues (session.tournament_counts)."""
    from opencrab2.highlevel.dealer import tournament_material
    from opencrab2.highlevel.session import tournament_counts
    n = 50257
    for k in (2, 4, 8):
        for with_index in (False, True):
            assert tournament_material(n, k, with_index) == tournament_counts(n, k, with_index)
    # k = 2 closed form: n - 1 pairwise comparisons (P = 1), no payload DReLU, one value select per
    # comparison plus one index select per comparison after the first level
    p1, p2, bv = tournament_material(n, 2, True)
    first_level_pairs = n // 2
    assert (p1, p2) == (n - 1, 0) and bv == (n - 1) + (n - 1 - first_level_pairs)
    ests = {k: material_estimate("argmax_k", (1, n), C, k=k) for k in (2, 4, 8)}
    assert ests[2]["drelu"] == (n - 1, (n - 1) * 1672)
    assert ests[2]["kary_select"] == (bv, 24 * bv)
    for k in (4, 8):
        p1, p2, bv = tournament_material(n, k, True)
        assert ests[k]["drelu"] == (p1 + p2, p1 * 1672 + p2 * 2200)
        assert ests[k].get("kary_select", (0, 0)) == ((bv, 24 * bv) if bv else (0, 0))
    # 3 rows of 10 at k = 4: nodes (4, 4, 2) then (3): pairs 6 + 6 + 1 + 3 = 16 (P = 1),
    # payload DReLUs 3 + 3 + 2 = 8 (P = 2), one Beaver select for the binary node
    mx = material_estimate("max_rows", (3, 10), C, k=4)
    assert mx == {"drelu": (3 * 24, 3 * (16 * 1672 + 8 * 2200)), "kary_select": (3, 3 * 24)}
    with pytest.raises(ContractViolation):
        material_estimate("argmax_k", (1, 10), C, k=5)
