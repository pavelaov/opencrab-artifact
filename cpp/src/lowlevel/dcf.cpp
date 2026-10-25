// Batched GGM DCF. See include/opencrab2/lowlevel/dcf.hpp.
#include "opencrab2/lowlevel/dcf.hpp"

#include <atomic>
#include <cstring>

#include "opencrab2/lowlevel/sharing.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace dcf {

namespace {

// Payload group arithmetic: modulus 0 is Z_{2^64}, otherwise Z_t.
// fold(s) == s % t for every s: one subtraction, selected by a mask, covers s < 2t, which is
// every sum of two reduced operands; the division is only reached for unreduced inputs (a key
// read from the wire may carry any 64-bit word). The results are those of the plain % everywhere.
// The selection is a mask and not a branch on purpose: whether a sum of two pseudo-random group
// elements exceeds t is a coin flip, and a mispredicted branch costs more than the old division.
struct Group {
    u64 t;
    u64 fold(u64 s) const {
        const u64 d = s - t;
        const u64 keep = u64(0) - static_cast<u64>(s < t);   // all ones where s is already reduced
        const u64 r = d ^ ((d ^ s) & keep);
        return r < t ? r : (r % t);
    }
    u64 red(u64 w) const { return t ? (w % t) : w; }
    u64 add(u64 a, u64 b) const { return t ? fold(a + b) : (a + b); }             // a, b < t < 2^62: no wrap
    u64 sub(u64 a, u64 b) const { return t ? fold(a + t - b) : (a - b); }
    u64 neg(u64 a) const { return t ? fold(t - a) : (u64(0) - a); }
    // cond ? neg(a) : a, without a branch (cond is a control bit: a coin flip)
    u64 neg_if(u64 cond, u64 a) const {
        const u64 m = u64(0) - static_cast<u64>(cond != 0);
        return a ^ ((a ^ neg(a)) & m);
    }
};

// m all ones: b, m zero: a (the selections on key bits and control bits are masks, never
// branches: the bits are pseudo-random and a mispredicted branch costs more than an AES block).
inline u64 sel(u64 a, u64 b, u64 m) { return a ^ ((a ^ b) & m); }
inline Block128 sel(const Block128& a, const Block128& b, u64 m) { return Block128{sel(a.lo, b.lo, m), sel(a.hi, b.hi, m)}; }

inline u64 domain_mask(int n_bits) { return n_bits >= 64 ? ~u64(0) : ((u64(1) << n_bits) - 1); }

Block128 random_block(Rng& rng) {
    Block128 b;
    b.lo = rng.next_u64();
    b.hi = rng.next_u64();
    return b;
}

const uint8_t kPyMagic[8] = {'O', 'C', '2', 'D', 'C', 'F', '0', '1'};

// Component-specific Convert (prg.hpp; the comparison-key repair of paper, Section 5.3 and
// Appendix D, Disclosure). Component p of the payload
// reads its own pseudo-random words from the seed s:
//   Z_{2^64}: block 2 for p = 0 (so P = 1 keys are bit-identical to those of the pre-repair rule),
//             block 3 + p for p >= 1; vL = LE64 of bytes 0..7, vR = LE64 of bytes 8..15.
//   Z_t:      vL = LE128(block 4 + 2p) mod t, vR = LE128(block 5 + 2p) mod t (128-bit reduction,
//             statistical distance at most t / 2^128 per word).
// Words are group elements already (below t in Z_t mode).
//
// The loops below do not call the PRG block by block. They lay out every MMO block of one tree
// level (of one key in generation, of a group of EVAL_GROUP keys in evaluation) as one
// (seed, tweak) list and hand it to mmo_batch_repeat, so the cipher sees 8 and more independent blocks
// per call. The block lists are sized once per batch call; nothing is allocated per key.

// Keys evaluated together. One level of the group is EVAL_GROUP * (2 + P) blocks: 48 for P = 1.
constexpr std::size_t EVAL_GROUP = 16;

#if defined(__GNUC__) || defined(__clang__)
#define OPENCRAB2_DCF_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define OPENCRAB2_DCF_NOINLINE __declspec(noinline)
#else
#define OPENCRAB2_DCF_NOINLINE
#endif

// The three per-level steps of the evaluation of a group of w keys. They are separate functions
// with few live values each on purpose: written inline in one loop nest the compiler spilled
// most of them to the stack, and the bookkeeping cost more than the cipher. blk holds B = 2 + P
// blocks per key ([0] next seed, [1] control block, [2 + p] value block of component p); x_i is
// bit `shift` of xv[k]. The selections on x_i are masks: the bit is as good as random, a branch
// on it would be mispredicted half the time.

// acc[k * P + p] += vg + (cw_value & t), Z_{2^64}; cwv points at cw_values[(i * P) * M + m0].
// The component loop is the outer one: with the key loop outside, the mask of x_i is invariant
// in the inner loop and the compiler turns the selection back into a branch (loop unswitching).
OPENCRAB2_DCF_NOINLINE void eval_level_values_ring(u64* acc, const Block128* blk, std::size_t P, const u64* cwv, std::size_t M,
                                                   const u64* xv, unsigned shift, const u64* t, std::size_t w) {
    const std::size_t B = 2u + P;
    for (std::size_t p = 0; p < P; ++p) {
        const u64* cw = cwv + p * M;
        const Block128* y = blk + 2 + p;
        u64* a = acc + p;
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (std::size_t k = 0; k < w; ++k) {
            const u64 mx = u64(0) - ((xv[k] >> shift) & 1u);   // all ones where x_i == 1
            const u64 mt = u64(0) - t[k];                       // all ones where t == 1
            const u64 lo = y[k * B].lo;
            const u64 hi = y[k * B].hi;
            a[k * P] += (lo ^ ((lo ^ hi) & mx)) + (cw[k] & mt);   // the word of side x_i
        }
    }
}

// The same in Z_t: the value block (already the one of side x_i) reduced mod t.
OPENCRAB2_DCF_NOINLINE void eval_level_values_zt(u64* acc, const Block128* blk, std::size_t P, const u64* cwv, std::size_t M,
                                                 const u64* t, std::size_t w, const Group& g) {
    const std::size_t B = 2u + P;
    for (std::size_t k = 0; k < w; ++k) {
        const u64 mt = u64(0) - t[k];
        const Block128* y = blk + k * B + 2;
        for (std::size_t p = 0; p < P; ++p) {
            const u64 vg = block_mod(y[p], g.t);
            acc[k * P + p] = g.add(acc[k * P + p], g.add(vg, cwv[p * M + k] & mt));
        }
    }
}

// Seed and control bit of the next level: s = y0 xor (cw_seed & t), t = t_x xor (cw_t_x & t).
OPENCRAB2_DCF_NOINLINE void eval_level_step(Block128* s, u64* t, const Block128* blk, std::size_t B, const u64* cw_s_lo,
                                            const u64* cw_s_hi, const u64* cw_tl, const u64* cw_tr, const u64* xv,
                                            unsigned shift, std::size_t w) {
    for (std::size_t k = 0; k < w; ++k) {
        const u64 mx = u64(0) - ((xv[k] >> shift) & 1u);
        const u64 tk = t[k];
        const u64 mt = u64(0) - tk;
        const Block128* y = blk + k * B;
        const u64 tL = prg_tl_bit(y[1]);
        const u64 tR = prg_tr_bit(y[1]);
        const u64 t_x = tL ^ ((tL ^ tR) & mx);                      // x_i ? tR : tL
        const u64 cw_x = cw_tl[k] ^ ((cw_tl[k] ^ cw_tr[k]) & mx);   // x_i ? cw_tR : cw_tL
        s[k].lo = y[0].lo ^ (cw_s_lo[k] & mt);
        s[k].hi = y[0].hi ^ (cw_s_hi[k] & mt);
        t[k] = t_x ^ (cw_x & tk);
    }
}

}  // namespace

Status check_n_bits(int n_bits) {
    if (n_bits < 1 || n_bits > 64) return fail_status("DCF_N_BITS: n_bits must be in [1, 64]");
    return ok_status();
}

Status check_modulus(u64 modulus) {
    if (modulus == 0) return ok_status();
    if (modulus < 3 || modulus >= MAX_MODULUS || (modulus & 1u) == 0)
        return fail_status("DCF_MODULUS: modulus must be an odd prime with 3 <= t < 2^62");
    // Odd PRIME only. The last accepted modulus is remembered, so a session that evaluates
    // many batches over one field pays for the Miller-Rabin test once.
    static std::atomic<u64> last_prime{0};
    if (last_prime.load(std::memory_order_relaxed) == modulus) return ok_status();
    if (!sharing::is_prime_u64(modulus)) return fail_status("DCF_MODULUS: modulus must be an odd prime with 3 <= t < 2^62");
    last_prime.store(modulus, std::memory_order_relaxed);
    return ok_status();
}

std::size_t key_bytes_per_element(int n_bits, int payloads) {
    return 16u + static_cast<std::size_t>(n_bits) * (16u + 8u * static_cast<std::size_t>(payloads) + 1u) +
           8u * static_cast<std::size_t>(payloads);
}

// ---------------------------------------------------------------------------------------------
// DcfKeyBatch
// ---------------------------------------------------------------------------------------------
std::size_t DcfKeyBatch::nbytes() const { return M() * key_bytes_per_element(n_bits, P); }

std::size_t DcfKeyBatch::serialized_bytes(WireLayout layout) const {
    const std::size_t m = M();
    const std::size_t n = n_bits;
    const std::size_t p = P;
    if (layout == WireLayout::PythonV2) return PY_WIRE_HEADER_BYTES + m * key_bytes_per_element(n_bits, P);
    return WIRE_HEADER_BYTES + 8u * (2u * m + n * 2u * m + n * p * m + 2u * n * m + p * m);
}

Status DcfKeyBatch::validate() const {
    Status st = check_n_bits(n_bits);
    if (!st.ok) return st;
    st = check_modulus(modulus);
    if (!st.ok) return st;
    if (party > 1) return fail_status("DCF_PARTY: party must be 0 or 1");
    if (P < 1) return fail_status("DCF_SHAPE: P must be >= 1");
    const std::size_t m = M();
    const std::size_t n = n_bits;
    if (seed.size() != 2u * m) return fail_status("DCF_SHAPE: seed");
    if (cw_seeds.size() != n * 2u * m) return fail_status("DCF_SHAPE: cw_seeds");
    if (cw_values.size() != n * P * m) return fail_status("DCF_SHAPE: cw_values");
    if (cw_tL.size() != n * m) return fail_status("DCF_SHAPE: cw_tL");
    if (cw_tR.size() != n * m) return fail_status("DCF_SHAPE: cw_tR");
    if (cw_final.size() != static_cast<std::size_t>(P) * m) return fail_status("DCF_SHAPE: cw_final");
    return ok_status();
}

Expected<DcfKeyBatch> DcfKeyBatch::slice(std::size_t lo, std::size_t hi) const {
    using R = Expected<DcfKeyBatch>;
    const std::size_t m = M();
    if (lo > hi || hi > m) return R::failure("DCF_SLICE: range out of bounds");
    const std::size_t len = hi - lo;
    const std::size_t n = n_bits;
    DcfKeyBatch out;
    out.n_bits = n_bits;
    out.party = party;
    out.P = P;
    out.modulus = modulus;
    out.seed.assign(seed.begin() + 2 * lo, seed.begin() + 2 * hi);
    out.cw_seeds.resize(n * 2u * len);
    out.cw_values.resize(n * P * len);
    out.cw_tL.resize(n * len);
    out.cw_tR.resize(n * len);
    out.cw_final.resize(static_cast<std::size_t>(P) * len);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t w = 0; w < 2; ++w)
            std::memcpy(&out.cw_seeds[(i * 2 + w) * len], &cw_seeds[(i * 2 + w) * m + lo], len * sizeof(u64));
        for (std::size_t p = 0; p < P; ++p)
            std::memcpy(&out.cw_values[(i * P + p) * len], &cw_values[(i * P + p) * m + lo], len * sizeof(u64));
        std::memcpy(&out.cw_tL[i * len], &cw_tL[i * m + lo], len * sizeof(u64));
        std::memcpy(&out.cw_tR[i * len], &cw_tR[i * m + lo], len * sizeof(u64));
    }
    for (std::size_t p = 0; p < P; ++p) std::memcpy(&out.cw_final[p * len], &cw_final[p * m + lo], len * sizeof(u64));
    return R::success(std::move(out));
}

std::vector<uint8_t> DcfKeyBatch::to_bytes(WireLayout layout) const {
    std::vector<uint8_t> b;
    b.reserve(serialized_bytes(layout));
    const std::size_t m = M();
    if (layout == WireLayout::PythonV2) {
        // struct "<8sBBHQQI": magic, n_bits, party, P, M, modulus, flags
        b.insert(b.end(), kPyMagic, kPyMagic + 8);
        wire::put_u8(b, n_bits);
        wire::put_u8(b, party);
        wire::put_u16le(b, P);
        wire::put_u64le(b, static_cast<u64>(m));
        wire::put_u64le(b, modulus);
        wire::put_u32le(b, 0);
        wire::put_u64le_array(b, seed);
        wire::put_u64le_array(b, cw_seeds);
        wire::put_u64le_array(b, cw_values);
        for (std::size_t i = 0; i < cw_tL.size(); ++i)
            wire::put_u8(b, static_cast<uint8_t>((cw_tL[i] & 1u) | ((cw_tR[i] & 1u) << 1)));
        wire::put_u64le_array(b, cw_final);
        return b;
    }
    wire::put_u8(b, WIRE_VERSION);
    wire::put_u8(b, party);
    wire::put_u8(b, n_bits);
    wire::put_u8(b, P);
    wire::put_u64le(b, static_cast<u64>(m));
    wire::put_u64le(b, modulus);
    wire::put_u64le_array(b, seed);
    wire::put_u64le_array(b, cw_seeds);
    wire::put_u64le_array(b, cw_values);
    wire::put_u64le_array(b, cw_tL);
    wire::put_u64le_array(b, cw_tR);
    wire::put_u64le_array(b, cw_final);
    return b;
}

Expected<DcfKeyBatch> DcfKeyBatch::from_bytes(const uint8_t* data, std::size_t len) {
    using R = Expected<DcfKeyBatch>;
    if (len < WIRE_HEADER_BYTES) return R::failure("DCF_WIRE: buffer shorter than the header");
    DcfKeyBatch k;
    std::size_t m = 0;
    std::size_t off = 0;
    bool python_layout = false;
    if (len >= 8 && std::memcmp(data, kPyMagic, 8) == 0) {
        if (len < PY_WIRE_HEADER_BYTES) return R::failure("DCF_WIRE: buffer shorter than the header");
        python_layout = true;
        k.n_bits = data[8];
        k.party = data[9];
        const uint16_t p16 = wire::get_u16le(data + 10);
        if (p16 < 1 || p16 > 255) return R::failure("DCF_WIRE: P out of range");
        k.P = static_cast<uint8_t>(p16);
        const u64 m64 = wire::get_u64le(data + 12);
        k.modulus = wire::get_u64le(data + 20);
        const uint32_t flags = wire::get_u32le(data + 28);
        if (flags != 0) return R::failure("DCF_WIRE: unsupported flags");
        if (m64 > (u64(1) << 40)) return R::failure("DCF_WIRE: M out of range");
        m = static_cast<std::size_t>(m64);
        off = PY_WIRE_HEADER_BYTES;
    } else {
        if (data[0] != WIRE_VERSION) return R::failure("DCF_WIRE: bad version");
        k.party = data[1];
        k.n_bits = data[2];
        k.P = data[3];
        const u64 m64 = wire::get_u64le(data + 4);
        k.modulus = wire::get_u64le(data + 12);
        if (m64 > (u64(1) << 40)) return R::failure("DCF_WIRE: M out of range");
        m = static_cast<std::size_t>(m64);
        off = WIRE_HEADER_BYTES;
    }
    Status st = check_n_bits(k.n_bits);
    if (!st.ok) return R::failure(st.code);
    st = check_modulus(k.modulus);
    if (!st.ok) return R::failure(st.code);
    if (k.party > 1) return R::failure("DCF_WIRE: bad party");
    if (k.P < 1) return R::failure("DCF_WIRE: P must be >= 1");
    const std::size_t n = k.n_bits;
    const std::size_t p = k.P;
    const std::size_t expect = python_layout ? PY_WIRE_HEADER_BYTES + m * key_bytes_per_element(k.n_bits, k.P)
                                             : WIRE_HEADER_BYTES + 8u * (2u * m + n * 2u * m + n * p * m + 2u * n * m + p * m);
    if (len != expect) return R::failure("DCF_WIRE: length does not match the header");
    wire::get_u64le_array(data + off, 2u * m, k.seed);
    off += 8u * 2u * m;
    wire::get_u64le_array(data + off, n * 2u * m, k.cw_seeds);
    off += 8u * n * 2u * m;
    wire::get_u64le_array(data + off, n * p * m, k.cw_values);
    off += 8u * n * p * m;
    if (python_layout) {
        k.cw_tL.resize(n * m);
        k.cw_tR.resize(n * m);
        for (std::size_t i = 0; i < n * m; ++i) {
            k.cw_tL[i] = data[off + i] & 1u;
            k.cw_tR[i] = (data[off + i] >> 1) & 1u;
        }
        off += n * m;
    } else {
        wire::get_u64le_array(data + off, n * m, k.cw_tL);
        off += 8u * n * m;
        wire::get_u64le_array(data + off, n * m, k.cw_tR);
        off += 8u * n * m;
        for (std::size_t i = 0; i < n * m; ++i)
            if (k.cw_tL[i] > 1 || k.cw_tR[i] > 1) return R::failure("DCF_WIRE: control bit word not in {0,1}");
    }
    wire::get_u64le_array(data + off, p * m, k.cw_final);
    st = k.validate();
    if (!st.ok) return R::failure(st.code);
    return R::success(std::move(k));
}

// ---------------------------------------------------------------------------------------------
// Key generation (dealer side)
// ---------------------------------------------------------------------------------------------
Expected<std::pair<DcfKeyBatch, DcfKeyBatch>> gen_dcf_batch(const std::vector<u64>& alpha, const std::vector<u64>& beta,
                                                            int n_bits, Rng& rng, u64 modulus) {
    using R = Expected<std::pair<DcfKeyBatch, DcfKeyBatch>>;
    Status st = check_n_bits(n_bits);
    if (!st.ok) return R::failure(st.code);
    st = check_modulus(modulus);
    if (!st.ok) return R::failure(st.code);
    const std::size_t M = alpha.size();
    if (M == 0) return R::failure("DCF_SHAPE: empty batch");
    if (beta.size() % M != 0) return R::failure("DCF_SHAPE: beta must have M*P entries");
    const std::size_t P = beta.size() / M;
    if (P < 1 || P > 255) return R::failure("DCF_SHAPE: P must be in [1, 255]");
    const std::size_t n = static_cast<std::size_t>(n_bits);
    const Group g{modulus};
    const u64 dmask = domain_mask(n_bits);

    DcfKeyBatch k0;
    k0.n_bits = static_cast<uint8_t>(n_bits);
    k0.party = 0;
    k0.P = static_cast<uint8_t>(P);
    k0.modulus = modulus;
    k0.seed.resize(2 * M);
    k0.cw_seeds.resize(n * 2 * M);
    k0.cw_values.resize(n * P * M);
    k0.cw_tL.resize(n * M);
    k0.cw_tR.resize(n * M);
    k0.cw_final.resize(P * M);
    std::vector<u64> seed1(2 * M);

    // Block list of one level of one key, both parties' seeds b = 0, 1 (B blocks each):
    //   [b*B + 0] seed L (tweak 0), [b*B + 1] seed R (tweak 1), [b*B + 2] control (tweak 3),
    //   Z_{2^64}: [b*B + 3 + p]            the value block of component p (vL = lo, vR = hi)
    //   Z_t:      [b*B + 3 + 2p + side]    the value block of component p and side, reduced mod t
    // and of the leaf: [b*P + p] the vL block of component p. Sized once for the whole batch.
    const std::size_t V = g.t ? 2u : 1u;          // value blocks per component
    const std::size_t B = 3u + V * P;
    std::vector<Block128> blk(2 * B);
    std::vector<u64> tw(2 * B), tw_leaf(2 * P);   // tweak words (prg_tweak_word)
    for (std::size_t b = 0; b < 2; ++b) {
        tw[b * B + 0] = prg_tweak_word(PRG_TWEAK_SEED_L);
        tw[b * B + 1] = prg_tweak_word(PRG_TWEAK_SEED_R);
        tw[b * B + 2] = prg_tweak_word(PRG_TWEAK_CONTROL);
        for (std::size_t p = 0; p < P; ++p) {
            const uint32_t c = static_cast<uint32_t>(p);
            if (g.t) {
                tw[b * B + 3 + 2 * p] = prg_tweak_word(prg_zt_value_tweak(c, 0));
                tw[b * B + 3 + 2 * p + 1] = prg_tweak_word(prg_zt_value_tweak(c, 1));
                tw_leaf[b * P + p] = prg_tweak_word(prg_zt_value_tweak(c, 0));
            } else {
                tw[b * B + 3 + p] = prg_tweak_word(prg_ring_value_tweak(c));
                tw_leaf[b * P + p] = prg_tweak_word(prg_ring_value_tweak(c));
            }
        }
    }
    std::vector<u64> v_alpha(P), v_cw(P), beta_red(P);
    for (std::size_t m = 0; m < M; ++m) {
        Block128 s[2] = {random_block(rng), random_block(rng)};
        k0.seed[2 * m] = s[0].lo;
        k0.seed[2 * m + 1] = s[0].hi;
        seed1[2 * m] = s[1].lo;
        seed1[2 * m + 1] = s[1].hi;
        u64 t[2] = {0, 1};
        for (std::size_t p = 0; p < P; ++p) {
            v_alpha[p] = 0;
            beta_red[p] = g.red(beta[m * P + p]);
        }
        const u64 a = alpha[m] & dmask;
        for (std::size_t i = 0; i < n; ++i) {
            const u64 a_i = (a >> (n - 1 - i)) & 1u;
            const u64 mR = u64(0) - a_i;   // all ones where Keep == Right (a_i == 1), zero where Keep == Left
            // every block of this level, of both parties' current seeds, in one cipher call
            mmo_batch_repeat(s, 2, B, tw.data(), blk.data());
            const Block128 sL[2] = {blk[0], blk[B]};
            const Block128 sR[2] = {blk[1], blk[B + 1]};
            const u64 tL[2] = {prg_tl_bit(blk[2]), prg_tl_bit(blk[B + 2])};
            const u64 tR[2] = {prg_tr_bit(blk[2]), prg_tr_bit(blk[B + 2])};
            // the Lose seeds: R where Keep == Left, L where Keep == Right
            const Block128 s_cw = sel(sR[0], sL[0], mR) ^ sel(sR[1], sL[1], mR);
            const u64 sign1 = t[1];
            for (std::size_t p = 0; p < P; ++p) {
                // Convert words of THIS payload component (the Convert rule above): a word
                // shared between components would expose alpha and beta_p - beta_0 in the public
                // difference of the value correction words.
                ValueWords w0, w1;
                if (g.t) {
                    w0.vL = block_mod(blk[3 + 2 * p], g.t);
                    w0.vR = block_mod(blk[3 + 2 * p + 1], g.t);
                    w1.vL = block_mod(blk[B + 3 + 2 * p], g.t);
                    w1.vR = block_mod(blk[B + 3 + 2 * p + 1], g.t);
                } else {
                    w0.vL = blk[3 + p].lo;
                    w0.vR = blk[3 + p].hi;
                    w1.vL = blk[B + 3 + p].lo;
                    w1.vR = blk[B + 3 + p].hi;
                }
                const u64 v_lose0 = sel(w0.vR, w0.vL, mR);
                const u64 v_lose1 = sel(w1.vR, w1.vL, mR);
                const u64 v_keep0 = sel(w0.vL, w0.vR, mR);
                const u64 v_keep1 = sel(w1.vL, w1.vR, mR);
                const u64 diff = g.sub(v_lose1, v_lose0);
                // the payload enters on the Lose branch exactly when Keep == Right (a_i == 1)
                const u64 add_beta = beta_red[p] & mR;
                const u64 vc = g.neg_if(sign1, g.add(g.sub(diff, v_alpha[p]), add_beta));
                v_cw[p] = vc;
                v_alpha[p] = g.add(g.add(g.sub(v_alpha[p], v_keep1), v_keep0), g.neg_if(sign1, vc));
            }
            const u64 t_cw_L = tL[0] ^ tL[1] ^ a_i ^ 1u;
            const u64 t_cw_R = tR[0] ^ tR[1] ^ a_i;
            k0.cw_seeds[(i * 2 + 0) * M + m] = s_cw.lo;
            k0.cw_seeds[(i * 2 + 1) * M + m] = s_cw.hi;
            for (std::size_t p = 0; p < P; ++p) k0.cw_values[(i * P + p) * M + m] = v_cw[p];
            k0.cw_tL[i * M + m] = t_cw_L;
            k0.cw_tR[i * M + m] = t_cw_R;
            const u64 t_cw_keep = sel(t_cw_L, t_cw_R, mR);
            for (int b = 0; b < 2; ++b) {
                const u64 mt = u64(0) - static_cast<u64>(t[b] != 0);   // apply the correction where t == 1
                const Block128 s_keep = sel(sL[b], sR[b], mR);
                const u64 t_keep = sel(tL[b], tR[b], mR);
                s[b].lo = s_keep.lo ^ (s_cw.lo & mt);
                s[b].hi = s_keep.hi ^ (s_cw.hi & mt);
                t[b] = t_keep ^ (t_cw_keep & mt);
            }
        }
        // leaf convert: the final correction word of component p uses vL_p of the leaf seeds
        mmo_batch_repeat(s, 2, P, tw_leaf.data(), blk.data());
        for (std::size_t p = 0; p < P; ++p) {
            const u64 vfin0 = g.t ? block_mod(blk[p], g.t) : blk[p].lo;
            const u64 vfin1 = g.t ? block_mod(blk[P + p], g.t) : blk[P + p].lo;
            k0.cw_final[p * M + m] = g.neg_if(t[1], g.sub(g.sub(vfin1, vfin0), v_alpha[p]));
        }
    }
    DcfKeyBatch k1 = k0;
    k1.party = 1;
    k1.seed = std::move(seed1);
    return R::success(std::make_pair(std::move(k0), std::move(k1)));
}

// ---------------------------------------------------------------------------------------------
// Evaluation (party side)
// ---------------------------------------------------------------------------------------------
Expected<std::vector<u64>> eval_dcf_batch(const DcfKeyBatch& key, const std::vector<u64>& x) {
    using R = Expected<std::vector<u64>>;
    const Status st = key.validate();
    if (!st.ok) return R::failure(st.code);
    const std::size_t M = key.M();
    if (x.size() != M) return R::failure("DCF_SHAPE: x must have M entries");
    const std::size_t n = key.n_bits;
    const std::size_t P = key.P;
    const Group g{key.modulus};
    const u64 dmask = domain_mask(key.n_bits);
    // out[m * P + p] doubles as the running sum of component p of key m.
    std::vector<u64> out(M * P, 0);
    // Keys are walked EVAL_GROUP at a time, level by level: all the MMO blocks of one level of the
    // group go to the cipher in one call (B blocks per key: [0] the seed of branch x_i, [1] the
    // control block, [2 + p] the value block of component p on side x_i). The correction words of
    // a group are adjacent in the key arrays (index (i, m)), so a level reads them in order.
    // Everything below is sized here, once per call; nothing is allocated per key.
    const std::size_t B = 2u + P;
    const std::size_t G = M < EVAL_GROUP ? M : EVAL_GROUP;
    const bool zt = (g.t != 0);
    std::vector<Block128> blk(G * B);
    std::vector<u64> tw(G * B), tw_leaf(G * P);   // tweak words (prg_tweak_word)
    for (std::size_t k = 0; k < G; ++k) {
        tw[k * B + 0] = prg_tweak_word(PRG_TWEAK_SEED_L);      // the branch x_i is set per level
        tw[k * B + 1] = prg_tweak_word(PRG_TWEAK_CONTROL);
        for (std::size_t p = 0; p < P; ++p) {
            const uint32_t c = static_cast<uint32_t>(p);
            // one component-specific value block per payload component (the Convert rule above); only
            // the Z_t tweak depends on the side (block 4 + 2p + x_i, set per level)
            const u64 word = prg_tweak_word(zt ? prg_zt_value_tweak(c, 0) : prg_ring_value_tweak(c));
            tw[k * B + 2 + p] = word;
            tw_leaf[k * P + p] = word;   // leaf: vL_p
        }
    }
    Block128 s[EVAL_GROUP];
    u64 t[EVAL_GROUP];
    u64 xv[EVAL_GROUP];
    for (std::size_t m0 = 0; m0 < M; m0 += G) {
        const std::size_t w = (M - m0 < G) ? (M - m0) : G;
        for (std::size_t k = 0; k < w; ++k) {
            s[k] = key.seed_block(m0 + k);
            t[k] = key.party;
            xv[k] = x[m0 + k] & dmask;
        }
        u64* acc = &out[m0 * P];
        for (std::size_t i = 0; i < n; ++i) {
            const unsigned shift = static_cast<unsigned>(n - 1 - i);
            for (std::size_t k = 0; k < w; ++k) {
                const uint32_t side = static_cast<uint32_t>((xv[k] >> shift) & 1u);   // public input bit x_i
                u64* tk = &tw[k * B];
                tk[0] = prg_tweak_word(side);   // seed branch x_i (tweak 0 or 1)
                if (zt)
                    for (std::size_t p = 0; p < P; ++p)
                        tk[2 + p] = prg_tweak_word(prg_zt_value_tweak(static_cast<uint32_t>(p), side));
            }
            mmo_batch_repeat(s, w, B, tw.data(), blk.data());
            const u64* cwv = &key.cw_values[(i * P) * M + m0];
            if (zt)
                eval_level_values_zt(acc, blk.data(), P, cwv, M, t, w, g);
            else
                eval_level_values_ring(acc, blk.data(), P, cwv, M, xv, shift, t, w);
            eval_level_step(s, t, blk.data(), B, &key.cw_seeds[(i * 2 + 0) * M + m0], &key.cw_seeds[(i * 2 + 1) * M + m0],
                            &key.cw_tL[i * M + m0], &key.cw_tR[i * M + m0], xv, shift, w);
        }
        // leaf convert: vL_p of the leaf seed, P blocks per key
        mmo_batch_repeat(s, w, P, tw_leaf.data(), blk.data());
        for (std::size_t k = 0; k < w; ++k) {
            const u64 mt = u64(0) - t[k];
            for (std::size_t p = 0; p < P; ++p) {
                const Block128& y = blk[k * P + p];
                const u64 vfin = zt ? block_mod(y, g.t) : y.lo;
                u64 o = g.add(acc[k * P + p], g.add(vfin, key.cw_final[p * M + m0 + k] & mt));
                if (key.party == 1) o = g.neg(o);
                acc[k * P + p] = o;
            }
        }
    }
    return R::success(std::move(out));
}

Expected<DcfKeyBatch> concat_keys(const DcfKeyBatch& a, const DcfKeyBatch& b) {
    using R = Expected<DcfKeyBatch>;
    if (a.n_bits != b.n_bits || a.party != b.party || a.P != b.P || a.modulus != b.modulus)
        return R::failure("DCF_CONCAT: key batches differ in n_bits, party, P or modulus");
    Status st = a.validate();
    if (!st.ok) return R::failure(st.code);
    st = b.validate();
    if (!st.ok) return R::failure(st.code);
    const std::size_t ma = a.M(), mb = b.M(), m = ma + mb;
    const std::size_t n = a.n_bits, P = a.P;
    DcfKeyBatch out;
    out.n_bits = a.n_bits;
    out.party = a.party;
    out.P = a.P;
    out.modulus = a.modulus;
    out.seed = a.seed;
    out.seed.insert(out.seed.end(), b.seed.begin(), b.seed.end());
    out.cw_seeds.resize(n * 2 * m);
    out.cw_values.resize(n * P * m);
    out.cw_tL.resize(n * m);
    out.cw_tR.resize(n * m);
    out.cw_final.resize(P * m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t w = 0; w < 2; ++w) {
            std::memcpy(&out.cw_seeds[(i * 2 + w) * m], &a.cw_seeds[(i * 2 + w) * ma], ma * sizeof(u64));
            std::memcpy(&out.cw_seeds[(i * 2 + w) * m + ma], &b.cw_seeds[(i * 2 + w) * mb], mb * sizeof(u64));
        }
        for (std::size_t p = 0; p < P; ++p) {
            std::memcpy(&out.cw_values[(i * P + p) * m], &a.cw_values[(i * P + p) * ma], ma * sizeof(u64));
            std::memcpy(&out.cw_values[(i * P + p) * m + ma], &b.cw_values[(i * P + p) * mb], mb * sizeof(u64));
        }
        std::memcpy(&out.cw_tL[i * m], &a.cw_tL[i * ma], ma * sizeof(u64));
        std::memcpy(&out.cw_tL[i * m + ma], &b.cw_tL[i * mb], mb * sizeof(u64));
        std::memcpy(&out.cw_tR[i * m], &a.cw_tR[i * ma], ma * sizeof(u64));
        std::memcpy(&out.cw_tR[i * m + ma], &b.cw_tR[i * mb], mb * sizeof(u64));
    }
    for (std::size_t p = 0; p < P; ++p) {
        std::memcpy(&out.cw_final[p * m], &a.cw_final[p * ma], ma * sizeof(u64));
        std::memcpy(&out.cw_final[p * m + ma], &b.cw_final[p * mb], mb * sizeof(u64));
    }
    return R::success(std::move(out));
}

Expected<std::vector<u64>> reconstruct_dcf(const std::vector<u64>& out0, const std::vector<u64>& out1, u64 modulus) {
    using R = Expected<std::vector<u64>>;
    if (out0.size() != out1.size()) return R::failure("DCF_SHAPE: output share vectors differ in size");
    const Group g{modulus};
    std::vector<u64> r(out0.size());
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = g.add(g.red(out0[i]), g.red(out1[i]));
    return R::success(std::move(r));
}

}  // namespace dcf
}  // namespace lowlevel
}  // namespace opencrab2
