// A64 -> Zt resharing. See include/opencrab2/lowlevel/ring_to_field.hpp.
#include "opencrab2/lowlevel/ring_to_field.hpp"

#include <cstring>

#include "opencrab2/lowlevel/ring.hpp"
#include "opencrab2/lowlevel/sharing.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace r2f {

namespace {
const uint8_t kMagic[8] = {'O', 'C', '2', 'R', '2', 'F', '0', '1'};
using Pair = std::pair<R2FMaterial, R2FMaterial>;
}  // namespace

Status check_r2f_params(u64 t, int k) {
    const Status st = ring::check_k(k);
    if (!st.ok) return st;
    if (t < 3 || t >= dcf::MAX_MODULUS || (t & 1u) == 0 || !sharing::is_prime_u64(t))
        return fail_status("R2F_MODULUS: t must be an odd prime with 3 <= t < 2^62");
    return ok_status();
}

u64 mulmod(u64 a, u64 b, u64 t) {
#if defined(__SIZEOF_INT128__)
    return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % t);
#else
    // double-and-add; a, b < t < 2^62 so every intermediate stays below 2^63
    u64 r = 0;
    a %= t;
    b %= t;
    while (b) {
        if (b & 1u) {
            r += a;
            if (r >= t) r -= t;
        }
        a <<= 1;
        if (a >= t) a -= t;
        b >>= 1;
    }
    return r;
#endif
}

u64 pow2_mod(int k, u64 t) {
    u64 v = 1 % t;
    for (int i = 0; i < k; ++i) {
        v <<= 1;   // v < t < 2^62, so 2v < 2^63
        if (v >= t) v -= t;
    }
    return v;
}

std::size_t r2f_bytes_per_element(u64 /*t*/) { return 24u + 2u * dcf::key_bytes_per_element(64, 1); }

std::size_t R2FMaterial::nbytes() const { return M() * (24u + 2u * dcf::key_bytes_per_element(k(), 1)); }

Status R2FMaterial::validate() const {
    Status st = key_A.validate();
    if (!st.ok) return st;
    st = key_C.validate();
    if (!st.ok) return st;
    st = check_r2f_params(t, k());
    if (!st.ok) return st;
    if (party > 1) return fail_status("R2F_PARTY: party must be 0 or 1");
    const std::size_t m = M();
    if (r_mod_t_share.size() != m || c_share.size() != m) return fail_status("R2F_SHAPE: share arrays differ in size");
    if (key_A.M() != m || key_C.M() != m) return fail_status("R2F_SHAPE: key batches differ in size");
    if (key_C.n_bits != key_A.n_bits) return fail_status("R2F_SHAPE: key domains differ");
    if (key_A.party != party || key_C.party != party) return fail_status("R2F_PARTY: keys belong to another party");
    if (key_A.P != 1 || key_C.P != 1) return fail_status("R2F_SHAPE: keys must carry one payload word");
    if (key_A.modulus != t || key_C.modulus != t) return fail_status("R2F_SHAPE: keys must use the Z_t payload group");
    return ok_status();
}

Expected<R2FMaterial> R2FMaterial::slice(std::size_t lo, std::size_t hi) const {
    using R = Expected<R2FMaterial>;
    if (lo > hi || hi > M()) return R::failure("R2F_SLICE: range out of bounds");
    R2FMaterial out;
    out.t = t;
    out.party = party;
    out.r_share.assign(r_share.begin() + lo, r_share.begin() + hi);
    out.r_mod_t_share.assign(r_mod_t_share.begin() + lo, r_mod_t_share.begin() + hi);
    out.c_share.assign(c_share.begin() + lo, c_share.begin() + hi);
    Expected<dcf::DcfKeyBatch> ka = key_A.slice(lo, hi);
    if (!ka.ok) return R::failure(ka.code);
    Expected<dcf::DcfKeyBatch> kc = key_C.slice(lo, hi);
    if (!kc.ok) return R::failure(kc.code);
    out.key_A = std::move(ka.value);
    out.key_C = std::move(kc.value);
    return R::success(std::move(out));
}

std::vector<uint8_t> R2FMaterial::to_bytes(dcf::WireLayout key_layout) const {
    const std::vector<uint8_t> ka = key_A.to_bytes(key_layout);
    const std::vector<uint8_t> kc = key_C.to_bytes(key_layout);
    std::vector<uint8_t> b;
    b.reserve(WIRE_HEADER_BYTES + 24u * M() + ka.size() + kc.size());
    b.insert(b.end(), kMagic, kMagic + 8);
    dcf::wire::put_u8(b, static_cast<uint8_t>(k()));
    dcf::wire::put_u8(b, party);
    dcf::wire::put_u16le(b, 0);
    dcf::wire::put_u32le(b, 0);
    dcf::wire::put_u64le(b, static_cast<u64>(M()));
    dcf::wire::put_u64le(b, t);
    dcf::wire::put_u64le(b, static_cast<u64>(ka.size()));
    dcf::wire::put_u64le(b, static_cast<u64>(kc.size()));
    dcf::wire::put_u64le_array(b, r_share);
    dcf::wire::put_u64le_array(b, r_mod_t_share);
    dcf::wire::put_u64le_array(b, c_share);
    b.insert(b.end(), ka.begin(), ka.end());
    b.insert(b.end(), kc.begin(), kc.end());
    return b;
}

Expected<R2FMaterial> R2FMaterial::from_bytes(const uint8_t* data, std::size_t len) {
    using R = Expected<R2FMaterial>;
    if (len < WIRE_HEADER_BYTES) return R::failure("R2F_WIRE: buffer shorter than the header");
    if (std::memcmp(data, kMagic, 8) != 0) return R::failure("R2F_WIRE: bad magic");
    R2FMaterial m;
    const uint8_t k = data[8];
    m.party = data[9];
    const uint16_t reserved = dcf::wire::get_u16le(data + 10);
    const uint32_t flags = dcf::wire::get_u32le(data + 12);
    const u64 M64 = dcf::wire::get_u64le(data + 16);
    m.t = dcf::wire::get_u64le(data + 24);
    const u64 lka = dcf::wire::get_u64le(data + 32);
    const u64 lkc = dcf::wire::get_u64le(data + 40);
    if (reserved != 0 || flags != 0) return R::failure("R2F_WIRE: unsupported flags");
    Status st = check_r2f_params(m.t, k);
    if (!st.ok) return R::failure(st.code);
    if (M64 > (u64(1) << 40) || lka > len || lkc > len) return R::failure("R2F_WIRE: sizes out of range");
    const std::size_t M = static_cast<std::size_t>(M64);
    const std::size_t expect = WIRE_HEADER_BYTES + 24u * M + static_cast<std::size_t>(lka) + static_cast<std::size_t>(lkc);
    if (len != expect) return R::failure("R2F_WIRE: length does not match the header");
    std::size_t off = WIRE_HEADER_BYTES;
    dcf::wire::get_u64le_array(data + off, M, m.r_share);
    off += 8u * M;
    dcf::wire::get_u64le_array(data + off, M, m.r_mod_t_share);
    off += 8u * M;
    dcf::wire::get_u64le_array(data + off, M, m.c_share);
    off += 8u * M;
    Expected<dcf::DcfKeyBatch> ka = dcf::DcfKeyBatch::from_bytes(data + off, static_cast<std::size_t>(lka));
    if (!ka.ok) return R::failure(ka.code);
    off += static_cast<std::size_t>(lka);
    Expected<dcf::DcfKeyBatch> kc = dcf::DcfKeyBatch::from_bytes(data + off, static_cast<std::size_t>(lkc));
    if (!kc.ok) return R::failure(kc.code);
    m.key_A = std::move(ka.value);
    m.key_C = std::move(kc.value);
    if (m.key_A.n_bits != k) return R::failure("R2F_WIRE: embedded key A disagrees with the header");
    st = m.validate();
    if (!st.ok) return R::failure(st.code);
    return R::success(std::move(m));
}

// ---------------------------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------------------------
Expected<Pair> r2f_material_from_mask(const std::vector<u64>& r_in, u64 t, Rng& rng, int k) {
    using R = Expected<Pair>;
    const Status st = check_r2f_params(t, k);
    if (!st.ok) return R::failure(st.code);
    const std::size_t M = r_in.size();
    if (M == 0) return R::failure("R2F_SHAPE: empty batch");
    const u64 half = u64(1) << (k - 1);
    std::vector<u64> r(M), alpha_c(M), r_mod_t(M), c(M);
    const std::vector<u64> one(M, 1);
    for (std::size_t m = 0; m < M; ++m) {
        r[m] = ring::reduce_k(r_in[m], k);
        alpha_c[m] = ring::reduce_k(r[m] + half, k);
        r_mod_t[m] = r[m] % t;
        c[m] = (r[m] > alpha_c[m]) ? 1 : 0;
    }
    Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> ka = dcf::gen_dcf_batch(r, one, k, rng, t);
    if (!ka.ok) return R::failure(ka.code);
    Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kc = dcf::gen_dcf_batch(alpha_c, one, k, rng, t);
    if (!kc.ok) return R::failure(kc.code);
    auto rs = sharing::share_k(r, k, rng);
    Expected<std::pair<std::vector<u64>, std::vector<u64>>> rt = sharing::share_mod(r_mod_t, t, rng);
    if (!rt.ok) return R::failure(rt.code);
    Expected<std::pair<std::vector<u64>, std::vector<u64>>> cs = sharing::share_mod(c, t, rng);
    if (!cs.ok) return R::failure(cs.code);
    R2FMaterial m0, m1;
    m0.t = m1.t = t;
    m0.party = 0;
    m1.party = 1;
    m0.r_share = std::move(rs.first);
    m1.r_share = std::move(rs.second);
    m0.r_mod_t_share = std::move(rt.value.first);
    m1.r_mod_t_share = std::move(rt.value.second);
    m0.c_share = std::move(cs.value.first);
    m1.c_share = std::move(cs.value.second);
    m0.key_A = std::move(ka.value.first);
    m1.key_A = std::move(ka.value.second);
    m0.key_C = std::move(kc.value.first);
    m1.key_C = std::move(kc.value.second);
    return R::success(std::make_pair(std::move(m0), std::move(m1)));
}

Expected<Pair> r2f_material_k(std::size_t M, u64 t, Rng& rng, int k) {
    using R = Expected<Pair>;
    const Status st = check_r2f_params(t, k);
    if (!st.ok) return R::failure(st.code);
    std::vector<u64> r(M);
    for (std::size_t m = 0; m < M; ++m) r[m] = ring::reduce_k(rng.next_u64(), k);
    return r2f_material_from_mask(r, t, rng, k);
}

Expected<Pair> r2f_material(std::size_t M, u64 t, Rng& rng) { return r2f_material_k(M, t, rng, 64); }

// ---------------------------------------------------------------------------------------------
// Online
// ---------------------------------------------------------------------------------------------
Expected<std::vector<u64>> r2f_open_share(const std::vector<u64>& u_share, const R2FMaterial& mat, int party) {
    using R = Expected<std::vector<u64>>;
    if (party != mat.party) return R::failure("R2F_PARTY: material belongs to another party");
    if (u_share.size() != mat.M()) return R::failure("R2F_SHAPE: u_share must have M entries");
    std::vector<u64> out(u_share.size());
    for (std::size_t m = 0; m < out.size(); ++m) out[m] = ring::reduce_k(u_share[m] + mat.r_share[m], mat.k());
    return R::success(std::move(out));
}

Expected<std::vector<u64>> r2f_eval(const R2FMaterial& mat, const std::vector<u64>& y_in, int party) {
    using R = Expected<std::vector<u64>>;
    if (party != mat.party) return R::failure("R2F_PARTY: material belongs to another party");
    const Status st = mat.validate();
    if (!st.ok) return R::failure(st.code);
    const std::size_t M = mat.M();
    if (y_in.size() != M) return R::failure("R2F_SHAPE: y must have M entries");
    const int k = mat.k();
    const u64 t = mat.t;
    std::vector<u64> y(M);
    for (std::size_t m = 0; m < M; ++m) y[m] = ring::reduce_k(y_in[m], k);
    Expected<std::vector<u64>> wA = dcf::eval_dcf_batch(mat.key_A, y);
    if (!wA.ok) return R::failure(wA.code);
    Expected<std::vector<u64>> wC = dcf::eval_dcf_batch(mat.key_C, y);
    if (!wC.ok) return R::failure(wC.code);
    const u64 qmt = pow2_mod(k, t);
    std::vector<u64> out(M);
    for (std::size_t m = 0; m < M; ++m) {
        const u64 w = wA.value[m];                                            // [1{y < r}]_t
        const u64 nn = ((wC.value[m] + t - wA.value[m]) % t + mat.c_share[m]) % t;   // [1{u >= 0 signed}]_t
        const u64 b = ((party == 0 ? 1 : 0) + t - nn) % t;                    // [msb(u)]_t
        const u64 yt = (party == 0) ? (y[m] % t) : 0;
        const u64 u_mod_t = ((yt + t - mat.r_mod_t_share[m]) % t + mulmod(qmt, w, t)) % t;
        out[m] = (u_mod_t + t - mulmod(qmt, b, t)) % t;
    }
    return R::success(std::move(out));
}

u64 signed_mod(u64 u, u64 t, int k) {
    const u64 ur = ring::reduce_k(u, k);
    const u64 half = u64(1) << (k - 1);
    const u64 um = ur % t;
    if (ur >= half) return (um + t - pow2_mod(k, t)) % t;
    return um;
}

std::vector<u64> signed_mod(const std::vector<u64>& u, u64 t, int k) {
    std::vector<u64> out(u.size());
    for (std::size_t m = 0; m < u.size(); ++m) out[m] = signed_mod(u[m], t, k);
    return out;
}

Expected<R2FMaterial> concat_r2f(const R2FMaterial& a, const R2FMaterial& b) {
    using R = Expected<R2FMaterial>;
    if (a.t != b.t || a.party != b.party || a.k() != b.k()) return R::failure("R2F_CONCAT: materials differ in t, k or party");
    R2FMaterial out;
    out.t = a.t;
    out.party = a.party;
    out.r_share = a.r_share;
    out.r_share.insert(out.r_share.end(), b.r_share.begin(), b.r_share.end());
    out.r_mod_t_share = a.r_mod_t_share;
    out.r_mod_t_share.insert(out.r_mod_t_share.end(), b.r_mod_t_share.begin(), b.r_mod_t_share.end());
    out.c_share = a.c_share;
    out.c_share.insert(out.c_share.end(), b.c_share.begin(), b.c_share.end());
    Expected<dcf::DcfKeyBatch> ka = dcf::concat_keys(a.key_A, b.key_A);
    if (!ka.ok) return R::failure(ka.code);
    Expected<dcf::DcfKeyBatch> kc = dcf::concat_keys(a.key_C, b.key_C);
    if (!kc.ok) return R::failure(kc.code);
    out.key_A = std::move(ka.value);
    out.key_C = std::move(kc.value);
    return R::success(std::move(out));
}

}  // namespace r2f
}  // namespace lowlevel
}  // namespace opencrab2
