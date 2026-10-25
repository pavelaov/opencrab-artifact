"""Party tests and the in-process two-party harness shared by the gadget, HE, bridge and transformer tests.

Harness (test only, INSECURE_TEST_ONLY_SHARED_SEED): two threads, one
``socket.socketpair`` wrapped in two ``Channel`` objects, one ``MaterialLedger``
per party (journal in a temp dir), one real ``Dealer`` in insecure mode whose
bundle PAIRS are sealed per recipient and ingested into both real
``DealerClient`` objects on demand (``AutoProvisionClient`` catches
``MaterialStall`` and asks the shared dealer for exactly the missing bundle
pair under one lock, so both clients see identical bundle order). Every
record produced this way is DIAGNOSTIC and never security eligible; the
material path itself (ledger reserve/spend before the first message) is the
admitted one.
"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from opencrab2.errors import (ContractViolation, MaterialError, MaterialStall, MaterialVersionMismatch,
                              UnauthorizedRelease)
from opencrab2.highlevel.channel import Channel, frame_size
from opencrab2.highlevel.contract import DEFAULT_CONTRACT, NumericContract
from opencrab2.highlevel.dealer import MODE_INSECURE, PROFILE_ID, Dealer, DealerClient
from opencrab2.highlevel.ledger import MaterialLedger
from opencrab2.highlevel.party import Party, ReleasePolicy
from opencrab2.lowlevel import ring, sharing

CONTRACT = DEFAULT_CONTRACT
F = CONTRACT.frac_bits


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
class ProvisionHub:
    """Shared INSECURE dealer plus both clients; provisions bundle pairs on demand."""

    def __init__(self, contract: NumericContract, seed: int, setup_id: str, model_version: str,
                 secrets: Optional[Dict[str, np.ndarray]] = None, mask_shapes: Optional[Dict[str, tuple]] = None):
        self.contract = contract
        self.dealer = Dealer(contract, np.random.default_rng(seed), MODE_INSECURE,
                             setup_id=setup_id, model_version=model_version)
        self.keys = (b"k0" * 16, b"k1" * 16)
        self.clients: List[Optional[DealerClient]] = [None, None]
        self.lock = threading.RLock()
        self.secrets = dict(secrets or {})          # name -> encoded vector/matrix (dealer-side plaintext)
        self.mask_shapes = dict(mask_shapes or {})  # mat_id -> (d, m)
        self.bundles = 0

    def set_model_version(self, model_version: str) -> None:
        self.dealer.bind(model_version=model_version)

    def provision(self, make: Callable[[Dealer], list]) -> None:
        with self.lock:
            for h0, p0, h1, p1 in make(self.dealer):
                for r, (h, p) in enumerate(((h0, p0), (h1, p1))):
                    hh, pp, tag = self.dealer.seal(r, h, p, self.keys[r])
                    self.clients[r].ingest(hh, pp, tag)
                self.bundles += 1


class AutoProvisionClient:
    """Test-only wrapper: real DealerClient semantics, on MaterialStall provisions the missing bundle pair."""

    BEAVER_CHUNK = 1 << 14
    DRELU_CHUNK = 256
    TRUNC_CHUNK = 256

    def __init__(self, client: DealerClient, hub: ProvisionHub):
        self.client = client
        self.hub = hub
        self.party = client.party
        self.ledger = client.ledger
        self.contract = client.contract

    def _retry(self, fn: Callable[[], Any], make: Callable[[Dealer], list]) -> Any:
        try:
            return fn()
        except MaterialStall:
            pass
        with self.hub.lock:
            try:
                return fn()
            except MaterialStall:
                self.hub.provision(make)
                return fn()

    def take_beaver(self, shape):
        shp = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
        n = int(np.prod(shp)) if shp else 1
        return self._retry(lambda: self.client.take_beaver(shp),
                           lambda d: d.make_beaver((max(n, self.BEAVER_CHUNK),), 1))

    def take_matrix_triple(self, shape_a, shape_b):
        sa, sb = tuple(shape_a), tuple(shape_b)
        return self._retry(lambda: self.client.take_matrix_triple(sa, sb),
                           lambda d: d.make_matrix_triples(sa, sb, 1))

    def take_wmask_call(self, mat_id, n_rows):
        self.wmask_mask_share(mat_id)   # make sure the mask exists on the dealer side
        return self._retry(lambda: self.client.take_wmask_call(mat_id, n_rows),
                           lambda d: d.make_wmask_calls(mat_id, int(n_rows), 1))

    def wmask_mask_share(self, mat_id, for_version=None):
        def make(d: Dealer):
            if mat_id not in self.hub.mask_shapes:
                raise MaterialStall(f"MATERIAL_STALL: harness knows no wmask shape for {mat_id!r}")
            return d.make_wmask(mat_id, tuple(self.hub.mask_shapes[mat_id]))
        return self._retry(lambda: self.client.wmask_mask_share(mat_id, for_version=for_version), make)

    def wmask_mask_version(self, mat_id):
        return self.client.wmask_mask_version(mat_id)

    def take_drelu(self, count, with_product=True, n_bits=None):
        n = int(count)
        return self._retry(lambda: self.client.take_drelu(n, with_product=with_product, n_bits=n_bits),
                           lambda d: d.make_drelu(max(n, self.DRELU_CHUNK), n_bits, with_product))

    def take_trunc(self, count, f=None, L=None):
        n = int(count)
        return self._retry(lambda: self.client.take_trunc(n, f, L),
                           lambda d: d.make_trunc(max(n, self.TRUNC_CHUNK), f, L))

    def take_r2f(self, count, t):
        n = int(count)
        return self._retry(lambda: self.client.take_r2f(n, int(t)),
                           lambda d: d.make_r2f(max(n, 64), int(t)))

    def take_vector_share(self, name):
        def make(d: Dealer):
            if name not in self.hub.secrets:
                raise MaterialStall(f"VECTOR_SHARE_MISSING: harness knows no vector named {name!r}")
            return d.make_vector_share(name, self.hub.secrets[name])
        return self._retry(lambda: self.client.take_vector_share(name), make)

    def stock(self):
        return self.client.stock()


class TwoPartyHarness:
    """Runs fn0(party0) and fn1(party1) concurrently over a socketpair; returns (out0, out1)."""

    def __init__(self, seed: int = 7, contract: NumericContract = CONTRACT, secrets=None, mask_shapes=None,
                 release: Optional[Tuple[str, int]] = ("token", 8), model_version: str = "model-v1",
                 session_id: str = "session-test"):
        self.contract = contract
        self.seed = int(seed)
        self.model_version = model_version
        self.setup_id = "setup-test"
        self.tmp = tempfile.mkdtemp(prefix="oc2-party-")
        self.hub = ProvisionHub(contract, seed, self.setup_id, model_version, secrets, mask_shapes)
        binding = {"session_id": session_id, "setup_id": self.setup_id, "model_version": model_version,
                   "numeric_contract_hash": contract.contract_hash(), "profile_id": PROFILE_ID}
        self.ledgers = [MaterialLedger(os.path.join(self.tmp, f"ledger{p}.jsonl"), binding) for p in (0, 1)]
        self.clients = [DealerClient(p, self.ledgers[p], self.hub.keys[p], contract) for p in (0, 1)]
        self.hub.clients = self.clients
        self.auto = [AutoProvisionClient(self.clients[p], self.hub) for p in (0, 1)]
        self.release_spec = release
        s0, s1 = socket.socketpair()
        self.socks = (s0, s1)
        self.channels = [Channel(s0, 0), Channel(s1, 1)]
        self.parties: List[Party] = []
        for p in (0, 1):
            pol = ReleasePolicy(1, release[0], release[1]) if release is not None else None
            self.parties.append(Party(p, self.channels[p], self.auto[p], contract, pol,
                                      np.random.default_rng(1000 + p + 10 * self.seed),
                                      model_version=model_version))

    def run(self, fn0: Callable[[Party], Any], fn1: Callable[[Party], Any], timeout_s: float = 600.0) -> Tuple[Any, Any]:
        out: List[Any] = [None, None]
        err: List[Optional[BaseException]] = [None, None]

        def worker(i: int, fn: Callable[[Party], Any]) -> None:
            try:
                out[i] = fn(self.parties[i])
            except BaseException as exc:  # noqa: BLE001 (re-raised in the main thread)
                err[i] = exc
                try:
                    self.socks[1 - i].close()
                    self.socks[i].close()
                except OSError:
                    pass

        threads = [threading.Thread(target=worker, args=(i, f), daemon=True) for i, f in enumerate((fn0, fn1))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout_s)
        for i in (0, 1):
            if err[i] is not None and not isinstance(err[i], (ConnectionError,)):
                raise err[i]
        for i in (0, 1):
            if err[i] is not None:
                raise err[i]
        if any(t.is_alive() for t in threads):
            raise TimeoutError("HARNESS_TIMEOUT: a party did not finish")
        return out[0], out[1]

    def run_same(self, fn: Callable[[Party], Any], timeout_s: float = 600.0) -> Tuple[Any, Any]:
        return self.run(fn, fn, timeout_s)

    def close(self) -> None:
        for ch in self.channels:
            ch.close()
        for l in self.ledgers:
            l.close()


def enc(x) -> np.ndarray:
    return ring.encode(np.asarray(x, np.float64), CONTRACT)


def dec(v) -> np.ndarray:
    return ring.decode(np.asarray(v, np.uint64), CONTRACT)


def shares_of(x_enc: np.ndarray, seed: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    return sharing.share(np.asarray(x_enc, np.uint64), np.random.default_rng(seed))


def run_shared(h: TwoPartyHarness, x_enc: np.ndarray, gadget: Callable[[Party, np.ndarray], Any], seed: int = 11):
    """Share x, let each party apply ``gadget(party, share)``, reconstruct the outputs."""
    s0, s1 = shares_of(x_enc, seed)
    o0, o1 = h.run(lambda p: gadget(p, s0), lambda p: gadget(p, s1))
    return ring.add(o0, o1)


# ---------------------------------------------------------------------------
# tests: sharing, opening, release policy
# ---------------------------------------------------------------------------
def test_share_input_and_reveal_roundtrip():
    h = TwoPartyHarness(seed=1)
    try:
        x = enc(np.array([1.5, -2.25, 1000.0, -0.0001]))

        def fn(p: Party):
            s = p.share_input(x if p.party == 1 else None, owner=1, shape=(4,))
            return p.reveal(s)

        o0, o1 = h.run_same(fn)
        assert np.array_equal(o0, x) and np.array_equal(o1, x)
        assert h.channels[0].stats.rounds == h.channels[1].stats.rounds == 2
    finally:
        h.close()


def test_reveal_final_only_to_p1_and_policy_counts():
    h = TwoPartyHarness(seed=2, release=("token", 2))
    try:
        x = np.array([42], np.uint64)
        s0, s1 = shares_of(x)

        def fn(p: Party):
            s = s0 if p.party == 0 else s1
            a = p.reveal_final(s, "token")
            b = p.reveal_final(s, "token")
            with pytest.raises(UnauthorizedRelease):
                p.reveal_final(s, "token")          # budget of 2 exhausted
            with pytest.raises(UnauthorizedRelease):
                p.reveal_final(s, "class")          # wrong kind
            return a, b

        (a0, b0), (a1, b1) = h.run_same(fn)
        assert a0 is None and b0 is None
        assert int(a1[0]) == 42 and int(b1[0]) == 42
        assert h.parties[0].release.count == 2 and h.parties[1].release.count == 2
    finally:
        h.close()


def test_release_policy_validation():
    with pytest.raises(ContractViolation):
        ReleasePolicy(0, "token", 1)          # only P1 may receive outputs
    with pytest.raises(ContractViolation):
        ReleasePolicy(1, "logits", 1)         # undeclared kind
    pol = ReleasePolicy(1, "class", 1)
    pol.authorize("class")
    with pytest.raises(UnauthorizedRelease):
        pol.authorize("class")
    assert pol.remaining() == 0


def test_party_without_policy_cannot_release():
    h = TwoPartyHarness(seed=3, release=None)
    try:
        def fn(p: Party):
            with pytest.raises(UnauthorizedRelease):
                p.reveal_final(np.zeros(1, np.uint64), "token")
            return True
        assert h.run_same(fn) == (True, True)
    finally:
        h.close()


# ---------------------------------------------------------------------------
# tests: linear algebra
# ---------------------------------------------------------------------------
def test_local_ops_and_public_real_multiply():
    h = TwoPartyHarness(seed=4)
    try:
        rng = np.random.default_rng(0)
        x = rng.uniform(-100, 100, size=(5, 7))
        for c in (3.0, -2.0, 0.37, 1.0 / 768.0, -1.0 / 12.0, 1234.5678, 2.0 ** -11):
            out = run_shared(h, enc(x), lambda p, s: p.mul_public_real(s, c))
            # mantissa rounding: relative error <= 2^-f on x*c, plus 3 ulp of truncation and encoding
            tol = np.max(np.abs(x * c)) * 2.0 ** -F + 3.0 / 2 ** F
            assert np.max(np.abs(dec(out) - x * c)) <= tol, c
        out = run_shared(h, enc(x), lambda p, s: p.add_public_real(p.add(s, s), 1.25))
        assert np.allclose(dec(out), 2 * x + 1.25, atol=1e-9)
        out = run_shared(h, enc(x), lambda p, s: p.sub(s, p.const(1.0, s.shape)))
        assert np.allclose(dec(out), x - 1.0, atol=1e-9)
    finally:
        h.close()


def test_beaver_and_mul_fxp_against_reference():
    h = TwoPartyHarness(seed=5)
    try:
        rng = np.random.default_rng(1)
        x = rng.uniform(-50, 50, size=(3, 64))
        y = rng.uniform(-50, 50, size=(3, 64))
        xs = shares_of(enc(x), 1)
        ys = shares_of(enc(y), 2)

        def fn(p: Party):
            sx, sy = xs[p.party], ys[p.party]
            z = p.mul_fxp(sx, sy)
            z_exact = p.beaver_mul(sx, p.const(3.0, sx.shape))
            return z, z_exact

        (z0, e0), (z1, e1) = h.run_same(fn)
        xq, yq = dec(enc(x)), dec(enc(y))        # the gadget sees the quantised inputs
        assert np.max(np.abs(dec(ring.add(z0, z1)) - xq * yq)) < 2.0 / 2 ** F
        assert np.allclose(dec(ring.add(e0, e1)) / 2 ** F, 3.0 * xq, atol=1e-9)
        kinds = [s["kind"] for s in h.parties[0].all_stats]
        assert kinds.count("beaver_mul") == 0          # beaver_mul is a primitive, tracked by callers
        assert h.parties[0]._material_bytes > 0
    finally:
        h.close()


def test_matmul_shared_batched_and_linear_generic():
    secrets = {"W1": enc(np.random.default_rng(3).uniform(-1, 1, size=(16, 6))),
               "W1.b": enc(np.arange(6) * 0.5)}
    h = TwoPartyHarness(seed=6, secrets=secrets)
    try:
        rng = np.random.default_rng(2)
        A = rng.uniform(-4, 4, size=(2, 3, 5))
        B = rng.uniform(-4, 4, size=(2, 5, 4))
        As = shares_of(enc(A), 1)
        Bs = shares_of(enc(B), 2)
        X = rng.uniform(-3, 3, size=(4, 16))
        Xs = shares_of(enc(X), 3)

        def fn(p: Party):
            z = p.matmul_shared(As[p.party], Bs[p.party])
            y = p.linear_generic(Xs[p.party], "W1")
            return z, y

        (z0, y0), (z1, y1) = h.run_same(fn)
        assert np.max(np.abs(dec(ring.add(z0, z1)) - A @ B)) < 8.0 / 2 ** F
        W1 = dec(secrets["W1"])
        ref = X @ W1 + np.arange(6) * 0.5
        assert np.max(np.abs(dec(ring.add(y0, y1)) - ref)) < 16.0 / 2 ** F
        st = [s for s in h.parties[0].stats if s["kind"] == "matmul_shared"][0]
        assert st["rounds"] == 1 and st["bytes"] > 0
    finally:
        h.close()


def test_trunc_faithful_exact_including_large_squares():
    h = TwoPartyHarness(seed=7)
    try:
        rng = np.random.default_rng(5)
        # signed inputs across magnitudes, plus non-negative squares up to 2^60
        mags = [2 ** e for e in (10, 20, 30, 38)]
        vals = np.concatenate([rng.integers(-m, m, size=64) for m in mags])
        vals = np.concatenate([vals, np.array([2 ** 39 - 1, -(2 ** 39), 0, -1, 1, 2 ** 16 - 1, -(2 ** 16)])])
        x = ring.from_signed(vals.astype(np.int64))
        out = run_shared(h, x, lambda p, s: p.trunc_faithful(s))
        want = np.floor_divide(vals.astype(np.int64), 2 ** F)
        assert np.array_equal(ring.to_signed(out), want)
        big = rng.integers(0, 2 ** 60, size=256, dtype=np.int64)
        out = run_shared(h, ring.from_signed(big), lambda p, s: p.trunc_faithful(s))
        assert np.array_equal(ring.to_signed(out), big >> np.int64(F))
        # a shared square of a large residual (2^9 real) through square_fxp(exact=True)
        dx = np.full(128, 500.0)
        out = run_shared(h, enc(dx), lambda p, s: p.square_fxp(s, exact=True))
        assert np.max(np.abs(dec(out) - dx * dx)) <= 1.0 / 2 ** F
        st = [s for s in h.parties[0].all_stats if s["kind"] == "trunc_faithful"]
        assert all(s["rounds"] == 1 for s in st)
    finally:
        h.close()


def test_trunc_faithful_rejects_bad_params():
    h = TwoPartyHarness(seed=8)
    try:
        def fn(p: Party):
            with pytest.raises(ContractViolation):
                p.trunc_faithful(np.zeros(4, np.uint64), f=0)
            with pytest.raises(ContractViolation):
                p.trunc_faithful(np.zeros(4, np.uint64), f=40, L=40)
            return True
        assert h.run_same(fn) == (True, True)
    finally:
        h.close()


# ---------------------------------------------------------------------------
# tests: wmask install and versioning
# ---------------------------------------------------------------------------
def test_install_wmask_then_linear_wmask_and_versioning():
    rng = np.random.default_rng(9)
    W = rng.uniform(-1, 1, size=(12, 5))
    W2 = W + 0.25
    secrets = {"M.b": enc(np.linspace(-1, 1, 5))}
    h = TwoPartyHarness(seed=9, secrets=secrets, mask_shapes={"M": (12, 5), "M2": (12, 5)})
    try:
        X = rng.uniform(-3, 3, size=(4, 12))
        Xs = shares_of(enc(X), 4)

        def fn(p: Party):
            with pytest.raises(MaterialError):
                p.linear_wmask(Xs[p.party], "M")             # before install
            rec = p.install_wmask("M", enc(W) if p.party == 0 else None)
            assert rec["skipped"] is False and rec["rounds"] == 1 and rec["bytes"] == 2 * frame_size(np.zeros((12, 5), np.uint64))
            assert p.installed("M")
            again = p.install_wmask("M", enc(W) if p.party == 0 else None)
            assert again["skipped"] is True
            y = p.linear_wmask(Xs[p.party], "M")
            E1 = p._installed["M"]["E"].copy()
            # a new model version: the object is no longer installed and must be re-installed
            p.model_version = "model-v2"
            assert not p.installed("M")
            # ... but never under the mask that served model-v1: E2 - E1 would be W2 - W1 for the client.
            # The refusal comes before any message (byte counters unchanged) and drops the cached share.
            sent, recv = int(p.ch.stats.bytes_sent), int(p.ch.stats.bytes_recv)
            with pytest.raises(MaterialVersionMismatch) as ei:
                p.install_wmask("M", enc(W2) if p.party == 0 else None)
            assert ei.value.code == "WMASK_MASK_VERSION_MISMATCH"
            assert (int(p.ch.stats.bytes_sent), int(p.ch.stats.bytes_recv)) == (sent, recv)
            assert "M" not in p._wmask_b and np.array_equal(p._installed["M"]["E"], E1)
            with pytest.raises(MaterialVersionMismatch):          # still refused on a retry
                p.install_wmask("M", enc(W2) if p.party == 0 else None)
            with pytest.raises(MaterialVersionMismatch):          # the dealer client refuses the second version too
                p.dealer.client.wmask_mask_share("M", for_version=p._version())
            return y, E1

        (y0, E0), (y1, E1) = h.run_same(fn)
        assert np.array_equal(E0, E1)
        ref = X @ W + np.linspace(-1, 1, 5)
        assert np.max(np.abs(dec(ring.add(y0, y1)) - ref)) < 16.0 / 2 ** F
        assert h.ledgers[0].installed("wmask_install(M)", {"model_version": "model-v1", "client_key_domain": "default"})
        assert not h.ledgers[0].installed("wmask_install(M)", {"model_version": "model-v2", "client_key_domain": "default"})
        assert "M" in h.parties[0].installed_objects()
        # regression of mask reuse: a different matrix under a different mask gives E2 - E1 != W2 - W
        h.parties[0].model_version = h.parties[1].model_version = "model-v1"

        def fn2(p: Party):
            p.install_wmask("M2", enc(W2) if p.party == 0 else None)
            return p._installed["M2"]["E"]

        E2a, E2b = h.run_same(fn2)
        assert np.array_equal(E2a, E2b)
        diff = ring.to_signed(ring.sub(E2a, E0))
        assert not np.allclose(diff / 2 ** F, W2 - W, atol=1e-3)
    finally:
        h.close()


def test_install_wmask_owner_checks():
    h = TwoPartyHarness(seed=10, mask_shapes={"Q": (3, 3)})
    try:
        def fn(p: Party):
            with pytest.raises(ContractViolation):
                if p.party == 0:
                    p.install_wmask("Q", None)
                else:
                    p.install_wmask("Q", np.zeros((3, 3), np.uint64))
            return True
        assert h.run_same(fn) == (True, True)
    finally:
        h.close()


# ---------------------------------------------------------------------------
# tests: ring_to_field, ledger states, stats
# ---------------------------------------------------------------------------
def test_ring_to_field_signed_representative():
    h = TwoPartyHarness(seed=11)
    try:
        t = 68720050177
        rng = np.random.default_rng(6)
        vals = np.concatenate([rng.integers(-(2 ** 39), 2 ** 39, size=100), np.array([1, -1, 2 ** 39 - 1, -(2 ** 39)])])
        x = ring.from_signed(vals.astype(np.int64))
        # the wrapping pattern a0 = 2^64 - 1, a1 = 2 (u = 1)
        s0 = np.concatenate([np.array([2 ** 64 - 1], np.uint64), ring.rand_u64((x.size,), rng)])
        s1 = np.concatenate([np.array([2], np.uint64), ring.sub(x, s0[1:])])
        z0, z1 = h.run(lambda p: p.ring_to_field(s0, t), lambda p: p.ring_to_field(s1, t))
        got = sharing.reconstruct_mod(z0, z1, t)
        want = np.array([1] + [int(v) % t for v in vals.tolist()], np.uint64)
        assert np.array_equal(got, want)
        assert z0.max() < t and z1.max() < t
    finally:
        h.close()


def test_ledger_states_spent_and_stats_recorded():
    h = TwoPartyHarness(seed=12)
    try:
        x = enc(np.linspace(-2, 2, 32))
        run_shared(h, x, lambda p, s: p.drelu(s))
        for led in h.ledgers:
            c = led.counts()
            assert c["SPENT"] >= 1 and c["RESERVED"] == 0
        st = h.parties[0].stats
        assert st and st[-1]["kind"] == "drelu" and st[-1]["rounds"] == 1
        assert set(st[-1]) >= {"kind", "seconds", "bytes", "rounds", "material_bytes"}
        assert st[-1]["material_bytes"] > 0 and st[-1]["bytes"] == 2 * frame_size(np.zeros(32, np.uint64))
    finally:
        h.close()


def test_material_bytes_match_dealer_consumption():
    h = TwoPartyHarness(seed=13)
    try:
        x = enc(np.linspace(-1, 1, 16))
        run_shared(h, x, lambda p, s: p.mul_fxp(s, s))
        run_shared(h, x, lambda p, s: p.relu(s))
        for p in (0, 1):
            assert h.parties[p].material_bytes == h.clients[p].consumed_bytes
    finally:
        h.close()


def test_error_messages_carry_codes():
    with pytest.raises(ContractViolation) as ei:
        ReleasePolicy(1, "nope", 1)
    assert str(ei.value).startswith("RELEASE_KIND:")
    h = TwoPartyHarness(seed=14)
    try:
        def fn(p: Party):
            with pytest.raises(ContractViolation) as e2:
                p.max_rows(np.zeros((2, 4), np.uint64), k=3)
            assert str(e2.value).startswith("MAX_ROWS_K:")
            with pytest.raises(ContractViolation) as e3:
                p.max_rows(np.zeros((2, 4), np.uint64), mask=np.zeros((2, 4), bool))
            assert str(e3.value).startswith("MAX_ROWS_EMPTY:")
            return True
        assert h.run_same(fn) == (True, True)
    finally:
        h.close()
