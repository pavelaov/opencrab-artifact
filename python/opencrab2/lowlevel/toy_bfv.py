"""Toy power-of-two BFV reference scheme for the secure bridge (paper, Appendix B, toy scheme and A2H-toy).

Status: TEST_ONLY reference. The scheme lives in R_q = Z_q[X]/(X^N + 1) with
q = 2^64 (native numpy uint64 wrap), plaintext modulus t = 2^log_t with
log_t <= 32, N in {64, 128, 256, 512}, Delta = 2^(64 - log_t), a ternary
secret s, errors uniform in [-B_e, B_e], and public key pk = (-(a s + e), a).

    Enc(m; u, e0, e1) = (pk0 u + e0 + Delta m,  pk1 u + e1)
    Dec(c0, c1)       = floor((c0 + c1 s + Delta/2) / Delta) mod t

Because t divides q, Delta * t = 0 mod q, so plaintext arithmetic is exactly
arithmetic mod t and decryption is a faithful truncation by (64 - log_t) bits
of (c0 + c1 s + Delta/2). That is the property the secure bridge H2A_ref
(opencrab2.highlevel.bridge) relies on: it reproduces this decryption on
secret shares with one Beaver convolution and one faithful truncation gate,
and `decrypt_arith_steps` exposes the intermediate quantities so that a test
can check the two computations agree.

What is tested (tests/test_toy_bfv.py): 1000 encrypt/decrypt round trips at
every admitted N, homomorphic ct+ct, ct+pt, ct*pt against a pure-integer
reference, coefficient-packed matvec equals W x mod t at (N=128, d_in=d_out=8)
and (N=512, d_in=64, d_out=8), the noise bound refuses a violating parameter
set, and the A2H_toy local-reduction pattern.

Design limits (not security claims): q = 2^64 with N <= 512 offers no
computational security and the scheme is never used on real data. The noise
bound is a worst-case infinity-norm bound, so a parameter set it refuses may
still decrypt correctly most of the time; the module refuses anyway (fail
closed). Real-parameter BFV (SEAL/TenSEAL, N = 8192) is not emulated here.
Every error message starts with a stable upper-case code: TOY_BFV_PARAM,
TOY_BFV_SHAPE, TOY_BFV_PLAINTEXT_RANGE, TOY_BFV_NOISE_BUDGET, TOY_BFV_TYPE.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple

import numpy as np

from opencrab2.errors import BridgeError
from opencrab2.lowlevel.csprng import SecureRandom

RING_BITS = 64
ALLOWED_N = (64, 128, 256, 512)
MAX_LOG_T = 32

_U64 = np.uint64
_ZERO = np.uint64(0)
_ONE = np.uint64(1)


# ---------------------------------------------------------------------------
# uint64 helpers (local on purpose: this module must import nothing that is
# still being written; the arithmetic is the plain numpy wrap semantics)
# ---------------------------------------------------------------------------

def _as_poly(a, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype != np.uint64:
        raise BridgeError(f"TOY_BFV_TYPE: {name} must be a numpy uint64 array, got {arr.dtype}")
    if arr.ndim != 1:
        raise BridgeError(f"TOY_BFV_SHAPE: {name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _wrap_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (a + b).astype(np.uint64, copy=False)


def _wrap_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (a - b).astype(np.uint64, copy=False)


def _wrap_neg(a: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (_ZERO - a).astype(np.uint64, copy=False)


def _wrap_scalar_mul(a: np.ndarray, k: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (a * np.uint64(k % (1 << RING_BITS))).astype(np.uint64, copy=False)


def _signed_to_u64(values) -> np.ndarray:
    """Two's complement embedding of signed integers (int64 range) into uint64."""
    return np.asarray(values, dtype=np.int64).view(np.uint64)


def _u64_to_signed(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64).view(np.int64)


@lru_cache(maxsize=8)
def _negacyclic_index(N: int) -> Tuple[np.ndarray, np.ndarray]:
    k = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    idx = (k - j) % N
    neg = k < j
    return idx, neg


def negacyclic_mul(a, b) -> np.ndarray:
    """Product of two length-N uint64 polynomials modulo (X^N + 1) and 2^64.

    Implemented as the N x N negacyclic matrix of `a` applied to `b`; every
    multiply and the row sums wrap modulo 2^64 (exact ring arithmetic). N^2
    work is acceptable for N <= 512 (tested against a pure-integer reference).
    """
    a = _as_poly(a, "a")
    b = _as_poly(b, "b")
    if a.shape != b.shape:
        raise BridgeError(f"TOY_BFV_SHAPE: polynomial lengths differ ({a.shape[0]} vs {b.shape[0]})")
    N = a.shape[0]
    idx, neg = _negacyclic_index(N)
    with np.errstate(over="ignore"):
        M = a[idx]
        M = np.where(neg, _ZERO - M, M).astype(np.uint64, copy=False)
        prod = (M * b[None, :]).astype(np.uint64, copy=False)
        return prod.sum(axis=1, dtype=np.uint64)


def negacyclic_mul_reference(a, b) -> np.ndarray:
    """Pure-Python-integer reference of `negacyclic_mul` (slow; tests only)."""
    a = _as_poly(a, "a")
    b = _as_poly(b, "b")
    N = a.shape[0]
    al = [int(v) for v in a.tolist()]
    bl = [int(v) for v in b.tolist()]
    out = [0] * N
    mask = (1 << RING_BITS) - 1
    for i, ai in enumerate(al):
        if ai == 0:
            continue
        for j, bj in enumerate(bl):
            if bj == 0:
                continue
            k = i + j
            if k < N:
                out[k] = (out[k] + ai * bj) & mask
            else:
                out[k - N] = (out[k - N] - ai * bj) & mask
    return np.array(out, dtype=np.uint64)


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToyParams:
    """Toy BFV parameter set: ring degree N, plaintext bits log_t, error bound B_e."""

    N: int
    log_t: int
    B_e: int = 8

    def __post_init__(self) -> None:
        if self.N not in ALLOWED_N:
            raise BridgeError(f"TOY_BFV_PARAM: N must be one of {ALLOWED_N}, got {self.N}")
        if not (1 <= int(self.log_t) <= MAX_LOG_T):
            raise BridgeError(f"TOY_BFV_PARAM: log_t must satisfy 1 <= log_t <= {MAX_LOG_T}, got {self.log_t}")
        if int(self.B_e) < 1:
            raise BridgeError(f"TOY_BFV_PARAM: B_e must be >= 1, got {self.B_e}")

    @property
    def t(self) -> int:
        return 1 << self.log_t

    @property
    def delta(self) -> int:
        return 1 << (RING_BITS - self.log_t)

    @property
    def shift(self) -> int:
        """Number of low bits removed by decryption rounding: 64 - log_t."""
        return RING_BITS - self.log_t


# ---------------------------------------------------------------------------
# the scheme
# ---------------------------------------------------------------------------

class ToyBFV:
    """Power-of-two toy BFV over Z_{2^64}[X]/(X^N + 1) (see module docstring).

    Ciphertexts are tuples (c0, c1) of length-N uint64 arrays. Plaintexts are
    length-N uint64 arrays with coefficients in [0, t). The instance counts
    ciphertext-plaintext products in `ct_pt_products`.
    """

    def __init__(self, params: ToyParams):
        if not isinstance(params, ToyParams):
            raise BridgeError("TOY_BFV_TYPE: params must be a ToyParams instance")
        self.params = params
        self.ct_pt_products = 0

    # -- sampling -----------------------------------------------------------

    def _ternary(self, rng: Any) -> np.ndarray:
        return _signed_to_u64(rng.integers(-1, 2, size=self.params.N, dtype=np.int64))

    def _error(self, rng: Any) -> np.ndarray:
        B = self.params.B_e
        return _signed_to_u64(rng.integers(-B, B + 1, size=self.params.N, dtype=np.int64))

    def _uniform(self, rng: Any) -> np.ndarray:
        return rng.integers(0, 1 << RING_BITS, size=self.params.N, dtype=np.uint64)

    # -- validation ---------------------------------------------------------

    def _check_pt(self, m, name: str = "m") -> np.ndarray:
        m = _as_poly(m, name)
        if m.shape[0] != self.params.N:
            raise BridgeError(f"TOY_BFV_SHAPE: {name} must have length N={self.params.N}, got {m.shape[0]}")
        if m.size and int(m.max()) >= self.params.t:
            raise BridgeError(f"TOY_BFV_PLAINTEXT_RANGE: {name} has a coefficient >= t=2^{self.params.log_t}")
        return m

    def _check_ct(self, ct, name: str = "ct") -> Tuple[np.ndarray, np.ndarray]:
        if not (isinstance(ct, (tuple, list)) and len(ct) == 2):
            raise BridgeError(f"TOY_BFV_TYPE: {name} must be a pair (c0, c1)")
        c0 = _as_poly(ct[0], name + "[0]")
        c1 = _as_poly(ct[1], name + "[1]")
        if c0.shape[0] != self.params.N or c1.shape[0] != self.params.N:
            raise BridgeError(f"TOY_BFV_SHAPE: {name} components must have length N={self.params.N}")
        return c0, c1

    # -- key generation / encryption / decryption ---------------------------

    def keygen(self, rng: Any = None):
        """Returns (sk, pk): sk ternary in {-1, 0, 1} as uint64; pk = (-(a s + e), a).

        ``rng=None`` draws from a fresh OS-keyed ``csprng.SecureRandom``; a numpy Generator is for
        deterministic tests only (its state is recoverable from outputs).
        """
        if rng is None:
            rng = SecureRandom.from_os()
        s = self._ternary(rng)
        a = self._uniform(rng)
        e = self._error(rng)
        pk0 = _wrap_neg(_wrap_add(negacyclic_mul(a, s), e))
        return s, (pk0, a)

    def encrypt(self, pk, m, rng: Any = None):
        """Enc(m; u, e0, e1) = (pk0 u + e0 + Delta m, pk1 u + e1); m: (N,) uint64 with values < t.

        ``rng=None`` draws u, e0, e1 from a fresh OS-keyed ``csprng.SecureRandom``.
        """
        if rng is None:
            rng = SecureRandom.from_os()
        pk0, pk1 = self._check_ct(pk, "pk")
        m = self._check_pt(m)
        u = self._ternary(rng)
        e0 = self._error(rng)
        e1 = self._error(rng)
        c0 = _wrap_add(_wrap_add(negacyclic_mul(pk0, u), e0), _wrap_scalar_mul(m, self.params.delta))
        c1 = _wrap_add(negacyclic_mul(pk1, u), e1)
        return c0, c1

    def decrypt_arith_steps(self, sk, ct) -> dict:
        """Decryption broken into the steps the secure bridge reproduces on shares.

        Returns a dict with uint64 arrays: `c1_s` (= c1 * s), `phase`
        (= c0 + c1 s), `pre_round` (= phase + Delta/2 mod 2^64), `plaintext`
        (= pre_round >> shift, values in [0, t)), plus the integers `shift`
        (= 64 - log_t) and `half_delta`, and `noise_signed` (int64; the
        centred noise phase - Delta * plaintext, a test-only diagnostic of this
        toy scheme, never a protocol message). H2A_ref computes `c1_s` with
        Beaver triples, lets P0 add c0 and Delta/2 locally, then applies the
        faithful truncation gate by `shift` bits.
        """
        sk = _as_poly(sk, "sk")
        c0, c1 = self._check_ct(ct)
        if sk.shape[0] != self.params.N:
            raise BridgeError(f"TOY_BFV_SHAPE: sk must have length N={self.params.N}")
        shift = self.params.shift
        half_delta = self.params.delta // 2
        c1_s = negacyclic_mul(c1, sk)
        phase = _wrap_add(c0, c1_s)
        pre_round = _wrap_add(phase, np.full(self.params.N, half_delta, dtype=np.uint64))
        plaintext = (pre_round >> np.uint64(shift)).astype(np.uint64)
        noise = _wrap_sub(phase, _wrap_scalar_mul(plaintext, self.params.delta))
        return {
            "c1_s": c1_s,
            "phase": phase,
            "pre_round": pre_round,
            "plaintext": plaintext,
            "shift": shift,
            "half_delta": half_delta,
            "noise_signed": _u64_to_signed(noise),
        }

    def decrypt(self, sk, ct) -> np.ndarray:
        """m = round((c0 + c1 s) / Delta) mod t, as (N,) uint64 with values < t."""
        return self.decrypt_arith_steps(sk, ct)["plaintext"]

    # -- homomorphic operations --------------------------------------------

    def add_ct(self, a, b):
        a0, a1 = self._check_ct(a, "a")
        b0, b1 = self._check_ct(b, "b")
        return _wrap_add(a0, b0), _wrap_add(a1, b1)

    def add_pt(self, ct, m):
        """ct + Enc_0(m): adds Delta * m into c0 (no fresh noise)."""
        c0, c1 = self._check_ct(ct)
        m = self._check_pt(m)
        return _wrap_add(c0, _wrap_scalar_mul(m, self.params.delta)), c1

    def mul_pt(self, ct, p):
        """ct * p for a plaintext polynomial p (length N uint64, signed representatives recommended).

        Counts one ciphertext-plaintext product. The caller is responsible for
        the noise bound (`matvec_pt` checks it; a raw `mul_pt` does not know
        the operand magnitude semantics and only checks shapes).
        """
        c0, c1 = self._check_ct(ct)
        p = _as_poly(p, "p")
        if p.shape[0] != self.params.N:
            raise BridgeError(f"TOY_BFV_SHAPE: p must have length N={self.params.N}, got {p.shape[0]}")
        self.ct_pt_products += 1
        return negacyclic_mul(c0, p), negacyclic_mul(c1, p)

    # -- encoding -----------------------------------------------------------

    def encode_vector(self, x) -> np.ndarray:
        """Coefficient packing: x[j] mod t at coefficient j, zero elsewhere; len(x) <= N."""
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise BridgeError(f"TOY_BFV_SHAPE: encode_vector expects a vector, got shape {arr.shape}")
        if arr.shape[0] > self.params.N:
            raise BridgeError(f"TOY_BFV_SHAPE: vector length {arr.shape[0]} exceeds N={self.params.N}")
        t = self.params.t
        vals = [int(v) % t for v in arr.tolist()]
        poly = np.zeros(self.params.N, dtype=np.uint64)
        poly[: len(vals)] = np.array(vals, dtype=np.uint64)
        return poly

    def signed_mod_t(self, W) -> np.ndarray:
        """Entries of W reduced to the signed representative in [-t/2, t/2), as int64."""
        t = self.params.t
        arr = np.asarray(W)
        flat = [int(v) % t for v in arr.reshape(-1).tolist()]
        half = t // 2
        flat = [v - t if v >= half else v for v in flat]
        return np.array(flat, dtype=np.int64).reshape(arr.shape)

    def w_inf(self, W) -> int:
        """Infinity norm of the signed representatives of W modulo t."""
        s = self.signed_mod_t(W)
        return int(np.max(np.abs(s))) if s.size else 0

    def rows_per_ct(self, d_in: int) -> int:
        """Output rows that fit in one ciphertext with block width d_in (= N // d_in)."""
        if d_in < 1 or d_in > self.params.N:
            raise BridgeError(f"TOY_BFV_SHAPE: d_in must satisfy 1 <= d_in <= N, got {d_in}")
        return self.params.N // d_in

    def encode_weight_block(self, W_block) -> Tuple[np.ndarray, np.ndarray]:
        """Packed plaintext polynomial for rows k of W_block (r x d_in), r <= N // d_in.

        p = sum_k sum_j W[k, d_in-1-j] X^(j + k d_in). Multiplying the packed
        input polynomial sum_i x_i X^i by p places <W[k,:], x> at coefficient
        (k + 1) d_in - 1 and no cross term or negacyclic wrap reaches those
        positions (the coefficient packing of paper, Appendix B; the module tests check it).
        Entries are the signed representatives modulo t (keeps noise small).
        """
        Ws = self.signed_mod_t(W_block)
        if Ws.ndim != 2:
            raise BridgeError(f"TOY_BFV_SHAPE: weight block must be two-dimensional, got shape {Ws.shape}")
        r, d_in = Ws.shape
        if r > self.rows_per_ct(d_in):
            raise BridgeError(
                f"TOY_BFV_SHAPE: {r} rows exceed N // d_in = {self.params.N // d_in} rows per ciphertext"
            )
        coeffs = np.zeros(self.params.N, dtype=np.int64)
        for k in range(r):
            coeffs[k * d_in:(k + 1) * d_in] = Ws[k, ::-1]
        positions = np.arange(1, r + 1, dtype=np.int64) * d_in - 1
        return _signed_to_u64(coeffs), positions

    # -- noise bound --------------------------------------------------------

    def fresh_noise_bound(self) -> int:
        """||e0 - e u + e1 s||_inf <= B_e (2N + 1) for ternary u, s and errors <= B_e."""
        return self.params.B_e * (2 * self.params.N + 1)

    def noise_bound(self, d_in: int, w_inf: int) -> int:
        """Worst-case noise after one packed matvec: fresh * nnz(p) * ||W||_inf, nnz(p) <= (N // d_in) d_in.

        Conservative in two ways: it assumes every packable row is used and
        every weight attains the infinity norm.
        """
        nnz = self.rows_per_ct(d_in) * d_in
        return self.fresh_noise_bound() * nnz * int(w_inf)

    def noise_budget_ok(self, d_in: int, w_inf: int) -> bool:
        """True iff the worst-case matvec noise is below Delta / 2 (correct rounding)."""
        if int(w_inf) < 0:
            raise BridgeError("TOY_BFV_PARAM: w_inf must be non-negative")
        return self.noise_bound(d_in, w_inf) < self.params.delta // 2

    # -- matvec -------------------------------------------------------------

    def matvec_pt(self, ct_x, W) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Plaintext-weight matvec on an encrypted packed vector.

        ct_x encrypts `encode_vector(x)` (x of length d_in); W is (d_out, d_in)
        with integer entries (reduced mod t). Returns (ct_y, positions): the
        decryption of ct_y at `positions[k]` equals (W x)[k] mod t. Requires
        d_out <= N // d_in (one ciphertext); use `matvec_pt_blocks` otherwise.
        Refuses (BridgeError TOY_BFV_NOISE_BUDGET) when the worst-case noise
        bound would reach Delta / 2.
        """
        Warr = np.asarray(W)
        if Warr.ndim != 2:
            raise BridgeError(f"TOY_BFV_SHAPE: W must be two-dimensional, got shape {Warr.shape}")
        d_out, d_in = Warr.shape
        if d_out > self.rows_per_ct(d_in):
            raise BridgeError(
                f"TOY_BFV_SHAPE: d_out={d_out} exceeds N // d_in = {self.params.N // d_in}; use matvec_pt_blocks"
            )
        self._check_ct(ct_x, "ct_x")
        w_inf = self.w_inf(Warr)
        if not self.noise_budget_ok(d_in, w_inf):
            raise BridgeError(
                "TOY_BFV_NOISE_BUDGET: worst-case noise "
                f"{self.noise_bound(d_in, w_inf)} >= Delta/2 = {self.params.delta // 2} "
                f"for N={self.params.N}, log_t={self.params.log_t}, d_in={d_in}, w_inf={w_inf}"
            )
        p, positions = self.encode_weight_block(Warr)
        ct_y = self.mul_pt(ct_x, p)
        return ct_y, positions

    def matvec_pt_blocks(self, ct_x, W) -> list:
        """General d_out: splits W into row blocks of N // d_in rows; returns [(ct_y, positions, row_lo, row_hi)]."""
        Warr = np.asarray(W)
        if Warr.ndim != 2:
            raise BridgeError(f"TOY_BFV_SHAPE: W must be two-dimensional, got shape {Warr.shape}")
        d_out, d_in = Warr.shape
        rows = self.rows_per_ct(d_in)
        out = []
        for lo in range(0, d_out, rows):
            hi = min(d_out, lo + rows)
            ct_y, positions = self.matvec_pt(ct_x, Warr[lo:hi])
            out.append((ct_y, positions, lo, hi))
        return out

    # -- plaintext reference ------------------------------------------------

    def matvec_reference(self, W, x) -> np.ndarray:
        """(W x) mod t with exact Python integers (test oracle)."""
        Warr = np.asarray(W)
        xl = [int(v) for v in np.asarray(x).reshape(-1).tolist()]
        t = self.params.t
        rows = []
        for k in range(Warr.shape[0]):
            acc = 0
            for j, wv in enumerate(Warr[k].tolist()):
                acc += int(wv) * xl[j]
            rows.append(acc % t)
        return np.array(rows, dtype=np.uint64)


def a2h_toy_local(bfv: ToyBFV, pk, x1_share, x0_share, rng: Any):
    """Local arithmetic of A2H_toy (paper, Appendix B, toy scheme and A2H-toy), without the channel.

    P1 reduces its A64 share mod t (low log_t bits), encrypts under its own
    key; P0 adds Delta * (x0 mod t) into c0. The result encrypts x mod t
    because Delta * t = 0 mod 2^64. This helper is the arithmetic oracle used
    by the bridge test; the two-party version with one message P1 -> P0 lives
    in opencrab2.highlevel.bridge.
    """
    ct1 = bfv.encrypt(pk, bfv.encode_vector(np.asarray(x1_share, dtype=np.uint64)), rng)
    return bfv.add_pt(ct1, bfv.encode_vector(np.asarray(x0_share, dtype=np.uint64)))
