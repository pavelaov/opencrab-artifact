// Faithful truncation gate. See include/opencrab2/lowlevel/trunc.hpp.
#include "opencrab2/lowlevel/trunc.hpp"

#include <cstring>

#include "opencrab2/lowlevel/ring.hpp"
#include "opencrab2/lowlevel/sharing.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace trunc {

namespace {
const uint8_t kMagic[8] = {'O', 'C', '2', 'T', 'R', 'N', '0', '1'};
using Pair = std::pair<TruncMaterial, TruncMaterial>;
}  // namespace

Status check_trunc_params(int f, int L, int k) {
    const Status st = ring::check_k(k);
    if (!st.ok) return st;
    if (!(1 <= f && f <= L - 1 && L - 1 <= k - 1)) return fail_status("TRUNC_PARAMS: need 1 <= f <= L-1 <= k-1");
    return ok_status();
}

std::size_t trunc_bytes_per_element(int f) {
    return dcf::key_bytes_per_element(64, 1) + dcf::key_bytes_per_element(f, 1) + 16u;
}

std::size_t TruncMaterial::nbytes() const {
    return M() * (dcf::key_bytes_per_element(k(), 1) + dcf::key_bytes_per_element(f, 1) + 16u);
}

Status TruncMaterial::validate() const {
    Status st = key_wrap.validate();
    if (!st.ok) return st;
    st = key_borrow.validate();
    if (!st.ok) return st;
    st = check_trunc_params(f, L, k());
    if (!st.ok) return st;
    if (party > 1) return fail_status("TRUNC_PARTY: party must be 0 or 1");
    const std::size_t m = M();
    if (r_high_share.size() != m) return fail_status("TRUNC_SHAPE: share arrays differ in size");
    if (key_wrap.M() != m || key_borrow.M() != m) return fail_status("TRUNC_SHAPE: key batches differ in size");
    if (key_borrow.n_bits != f) return fail_status("TRUNC_SHAPE: borrow key domain differs from f");
    if (key_wrap.party != party || key_borrow.party != party) return fail_status("TRUNC_PARTY: keys belong to another party");
    if (key_wrap.P != 1 || key_borrow.P != 1) return fail_status("TRUNC_SHAPE: keys must carry one payload word");
    if (key_wrap.modulus != 0 || key_borrow.modulus != 0) return fail_status("TRUNC_SHAPE: keys must use the Z_{2^64} payload group");
    return ok_status();
}

Expected<TruncMaterial> TruncMaterial::slice(std::size_t lo, std::size_t hi) const {
    using R = Expected<TruncMaterial>;
    if (lo > hi || hi > M()) return R::failure("TRUNC_SLICE: range out of bounds");
    TruncMaterial out;
    out.f = f;
    out.L = L;
    out.party = party;
    out.r_share.assign(r_share.begin() + lo, r_share.begin() + hi);
    out.r_high_share.assign(r_high_share.begin() + lo, r_high_share.begin() + hi);
    Expected<dcf::DcfKeyBatch> kw = key_wrap.slice(lo, hi);
    if (!kw.ok) return R::failure(kw.code);
    Expected<dcf::DcfKeyBatch> kb = key_borrow.slice(lo, hi);
    if (!kb.ok) return R::failure(kb.code);
    out.key_wrap = std::move(kw.value);
    out.key_borrow = std::move(kb.value);
    return R::success(std::move(out));
}

std::vector<uint8_t> TruncMaterial::to_bytes(dcf::WireLayout key_layout) const {
    const std::vector<uint8_t> kw = key_wrap.to_bytes(key_layout);
    const std::vector<uint8_t> kb = key_borrow.to_bytes(key_layout);
    std::vector<uint8_t> b;
    b.reserve(WIRE_HEADER_BYTES + 16u * M() + kw.size() + kb.size());
    b.insert(b.end(), kMagic, kMagic + 8);
    dcf::wire::put_u8(b, f);
    dcf::wire::put_u8(b, L);
    dcf::wire::put_u8(b, static_cast<uint8_t>(k()));
    dcf::wire::put_u8(b, party);
    dcf::wire::put_u32le(b, 0);
    dcf::wire::put_u64le(b, static_cast<u64>(M()));
    dcf::wire::put_u64le(b, static_cast<u64>(kw.size()));
    dcf::wire::put_u64le(b, static_cast<u64>(kb.size()));
    dcf::wire::put_u64le_array(b, r_share);
    dcf::wire::put_u64le_array(b, r_high_share);
    b.insert(b.end(), kw.begin(), kw.end());
    b.insert(b.end(), kb.begin(), kb.end());
    return b;
}

Expected<TruncMaterial> TruncMaterial::from_bytes(const uint8_t* data, std::size_t len) {
    using R = Expected<TruncMaterial>;
    if (len < WIRE_HEADER_BYTES) return R::failure("TRUNC_WIRE: buffer shorter than the header");
    if (std::memcmp(data, kMagic, 8) != 0) return R::failure("TRUNC_WIRE: bad magic");
    TruncMaterial m;
    m.f = data[8];
    m.L = data[9];
    const uint8_t k = data[10];
    m.party = data[11];
    const uint32_t flags = dcf::wire::get_u32le(data + 12);
    const u64 M64 = dcf::wire::get_u64le(data + 16);
    const u64 lkw = dcf::wire::get_u64le(data + 24);
    const u64 lkb = dcf::wire::get_u64le(data + 32);
    if (flags != 0) return R::failure("TRUNC_WIRE: unsupported flags");
    Status st = check_trunc_params(m.f, m.L, k);
    if (!st.ok) return R::failure(st.code);
    if (M64 > (u64(1) << 40) || lkw > len || lkb > len) return R::failure("TRUNC_WIRE: sizes out of range");
    const std::size_t M = static_cast<std::size_t>(M64);
    const std::size_t expect = WIRE_HEADER_BYTES + 16u * M + static_cast<std::size_t>(lkw) + static_cast<std::size_t>(lkb);
    if (len != expect) return R::failure("TRUNC_WIRE: length does not match the header");
    std::size_t off = WIRE_HEADER_BYTES;
    dcf::wire::get_u64le_array(data + off, M, m.r_share);
    off += 8u * M;
    dcf::wire::get_u64le_array(data + off, M, m.r_high_share);
    off += 8u * M;
    Expected<dcf::DcfKeyBatch> kw = dcf::DcfKeyBatch::from_bytes(data + off, static_cast<std::size_t>(lkw));
    if (!kw.ok) return R::failure(kw.code);
    off += static_cast<std::size_t>(lkw);
    Expected<dcf::DcfKeyBatch> kb = dcf::DcfKeyBatch::from_bytes(data + off, static_cast<std::size_t>(lkb));
    if (!kb.ok) return R::failure(kb.code);
    m.key_wrap = std::move(kw.value);
    m.key_borrow = std::move(kb.value);
    if (m.key_wrap.n_bits != k) return R::failure("TRUNC_WIRE: embedded wrap key disagrees with the header");
    st = m.validate();
    if (!st.ok) return R::failure(st.code);
    return R::success(std::move(m));
}

// ---------------------------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------------------------
Expected<Pair> trunc_material_from_mask(const std::vector<u64>& r_in, int f, int L, Rng& rng, int k) {
    using R = Expected<Pair>;
    const Status st = check_trunc_params(f, L, k);
    if (!st.ok) return R::failure(st.code);
    const std::size_t M = r_in.size();
    if (M == 0) return R::failure("TRUNC_SHAPE: empty batch");
    std::vector<u64> r(M), r_hi(M), r_lo(M);
    const std::vector<u64> one(M, 1);
    for (std::size_t m = 0; m < M; ++m) {
        r[m] = ring::reduce_k(r_in[m], k);
        r_hi[m] = r[m] >> f;
        r_lo[m] = ring::reduce_k(r[m], f);
    }
    Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kw = dcf::gen_dcf_batch(r, one, k, rng, 0);
    if (!kw.ok) return R::failure(kw.code);
    Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kb = dcf::gen_dcf_batch(r_lo, one, f, rng, 0);
    if (!kb.ok) return R::failure(kb.code);
    auto rs = sharing::share_k(r, k, rng);
    auto hs = sharing::share_k(r_hi, k, rng);
    TruncMaterial m0, m1;
    m0.f = m1.f = static_cast<uint8_t>(f);
    m0.L = m1.L = static_cast<uint8_t>(L);
    m0.party = 0;
    m1.party = 1;
    m0.r_share = std::move(rs.first);
    m1.r_share = std::move(rs.second);
    m0.r_high_share = std::move(hs.first);
    m1.r_high_share = std::move(hs.second);
    m0.key_wrap = std::move(kw.value.first);
    m1.key_wrap = std::move(kw.value.second);
    m0.key_borrow = std::move(kb.value.first);
    m1.key_borrow = std::move(kb.value.second);
    return R::success(std::make_pair(std::move(m0), std::move(m1)));
}

Expected<Pair> trunc_material_k(std::size_t M, int f, int L, Rng& rng, int k) {
    using R = Expected<Pair>;
    const Status st = check_trunc_params(f, L, k);
    if (!st.ok) return R::failure(st.code);
    std::vector<u64> r(M);
    for (std::size_t m = 0; m < M; ++m) r[m] = ring::reduce_k(rng.next_u64(), k);
    return trunc_material_from_mask(r, f, L, rng, k);
}

Expected<Pair> trunc_material(std::size_t M, int f, int L, Rng& rng) { return trunc_material_k(M, f, L, rng, 64); }

// ---------------------------------------------------------------------------------------------
// Online
// ---------------------------------------------------------------------------------------------
Expected<std::vector<u64>> trunc_offset(const std::vector<u64>& x_share, int party, int L, int k) {
    using R = Expected<std::vector<u64>>;
    Status st = ring::check_k(k);
    if (!st.ok) return R::failure(st.code);
    if (!(2 <= L && L <= k)) return R::failure("TRUNC_PARAMS: need 2 <= L <= k");
    if (party != 0 && party != 1) return R::failure("TRUNC_PARTY: party must be 0 or 1");
    const u64 a = u64(1) << (L - 1);
    std::vector<u64> out(x_share.size());
    for (std::size_t m = 0; m < out.size(); ++m) out[m] = ring::reduce_k(party == 0 ? x_share[m] + a : x_share[m], k);
    return R::success(std::move(out));
}

Expected<std::vector<u64>> trunc_open_share(const std::vector<u64>& x_share, const TruncMaterial& mat, int party) {
    using R = Expected<std::vector<u64>>;
    if (party != mat.party) return R::failure("TRUNC_PARTY: material belongs to another party");
    if (x_share.size() != mat.M()) return R::failure("TRUNC_SHAPE: x_share must have M entries");
    Expected<std::vector<u64>> xp = trunc_offset(x_share, party, mat.L, mat.k());
    if (!xp.ok) return xp;
    for (std::size_t m = 0; m < xp.value.size(); ++m) xp.value[m] = ring::reduce_k(xp.value[m] + mat.r_share[m], mat.k());
    return xp;
}

Expected<std::vector<u64>> trunc_faithful_eval(const TruncMaterial& mat, const std::vector<u64>& y_in, int party) {
    using R = Expected<std::vector<u64>>;
    if (party != mat.party) return R::failure("TRUNC_PARTY: material belongs to another party");
    const Status st = mat.validate();
    if (!st.ok) return R::failure(st.code);
    const int k = mat.k();
    const int f = mat.f;
    const int L = mat.L;
    const std::size_t M = mat.M();
    if (y_in.size() != M) return R::failure("TRUNC_SHAPE: y must have M entries");
    std::vector<u64> y(M), y_lo(M);
    for (std::size_t m = 0; m < M; ++m) {
        y[m] = ring::reduce_k(y_in[m], k);
        y_lo[m] = ring::reduce_k(y[m], f);
    }
    Expected<std::vector<u64>> w = dcf::eval_dcf_batch(mat.key_wrap, y);
    if (!w.ok) return R::failure(w.code);
    Expected<std::vector<u64>> b = dcf::eval_dcf_batch(mat.key_borrow, y_lo);
    if (!b.ok) return R::failure(b.code);
    const u64 wrap_unit = u64(1) << (k - f);            // 2^(k-f), k - f >= 1
    const u64 offset_term = u64(1) << (L - 1 - f);      // 2^(L-1-f)
    std::vector<u64> out(M);
    for (std::size_t m = 0; m < M; ++m) {
        u64 z = w.value[m] * wrap_unit - mat.r_high_share[m] - b.value[m];
        if (party == 0) z = z + (y[m] >> f) - offset_term;
        out[m] = ring::reduce_k(z, k);
    }
    return R::success(std::move(out));
}

Expected<u64> trunc_local(u64 share, int f, int party) {
    using R = Expected<u64>;
    if (f < 0 || f > 63) return R::failure("TRUNC_PARAMS: trunc_local needs 0 <= f <= 63");
    if (party != 0 && party != 1) return R::failure("TRUNC_PARTY: party must be 0 or 1");
    if (f == 0) return R::success(share);
    if (party == 0) return R::success(ring::arith_shift_right(share, f));
    return R::success(ring::neg(ring::arith_shift_right(ring::neg(share), f)));
}

Expected<std::vector<u64>> trunc_local(const std::vector<u64>& share, int f, int party) {
    using R = Expected<std::vector<u64>>;
    if (f < 0 || f > 63) return R::failure("TRUNC_PARAMS: trunc_local needs 0 <= f <= 63");
    if (party != 0 && party != 1) return R::failure("TRUNC_PARTY: party must be 0 or 1");
    std::vector<u64> out(share.size());
    for (std::size_t m = 0; m < out.size(); ++m) out[m] = trunc_local(share[m], f, party).value;
    return R::success(std::move(out));
}

// ---------------------------------------------------------------------------------------------
// Pure-integer reference
// ---------------------------------------------------------------------------------------------
namespace {
Expected<i64> reference_impl(int k, int L, int f, i64 x, u64 r, bool with_offset_term) {
    using R = Expected<i64>;
    const Status st = check_trunc_params(f, L, k);
    if (!st.ok) return R::failure(st.code);
    const u64 a = u64(1) << (L - 1);
    // x in [-a, a)  <=>  x + a in [0, 2^L) as an unsigned value (for L = 64 every int64 qualifies)
    const u64 xp = ring::from_signed(x) + a;
    if (L < 64 && xp >= (u64(1) << L)) return R::failure("TRUNC_RANGE: x outside [-2^(L-1), 2^(L-1))");
    r = ring::reduce_k(r, k);
    const u64 y = ring::reduce_k(xp + r, k);
    const u64 w = (y < r) ? 1 : 0;
    const u64 b = (ring::reduce_k(y, f) < ring::reduce_k(r, f)) ? 1 : 0;
    u64 z = (y >> f) - (r >> f) - b + (u64(1) << (k - f)) * w;
    if (with_offset_term) z -= u64(1) << (L - 1 - f);
    return R::success(ring::to_signed_k(z, k));
}
}  // namespace

Expected<i64> reference_trunc_small(int k, int L, int f, i64 x, u64 r) { return reference_impl(k, L, f, x, r, true); }

Expected<i64> reference_trunc_small_no_offset_term(int k, int L, int f, i64 x, u64 r) {
    return reference_impl(k, L, f, x, r, false);
}

Expected<TruncMaterial> concat_trunc(const TruncMaterial& a, const TruncMaterial& b) {
    using R = Expected<TruncMaterial>;
    if (a.f != b.f || a.L != b.L || a.party != b.party || a.k() != b.k())
        return R::failure("TRUNC_CONCAT: materials differ in f, L, k or party");
    TruncMaterial out;
    out.f = a.f;
    out.L = a.L;
    out.party = a.party;
    out.r_share = a.r_share;
    out.r_share.insert(out.r_share.end(), b.r_share.begin(), b.r_share.end());
    out.r_high_share = a.r_high_share;
    out.r_high_share.insert(out.r_high_share.end(), b.r_high_share.begin(), b.r_high_share.end());
    Expected<dcf::DcfKeyBatch> kw = dcf::concat_keys(a.key_wrap, b.key_wrap);
    if (!kw.ok) return R::failure(kw.code);
    Expected<dcf::DcfKeyBatch> kb = dcf::concat_keys(a.key_borrow, b.key_borrow);
    if (!kb.ok) return R::failure(kb.code);
    out.key_wrap = std::move(kw.value);
    out.key_borrow = std::move(kb.value);
    return R::success(std::move(out));
}

}  // namespace trunc
}  // namespace lowlevel
}  // namespace opencrab2
