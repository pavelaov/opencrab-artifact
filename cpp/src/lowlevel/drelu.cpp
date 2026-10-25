// One-round DReLU / ReLU. See include/opencrab2/lowlevel/drelu.hpp.
#include "opencrab2/lowlevel/drelu.hpp"

#include <cstring>

#include "opencrab2/lowlevel/ring.hpp"
#include "opencrab2/lowlevel/sharing.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace drelu {

namespace {
const uint8_t kMagic[8] = {'O', 'C', '2', 'D', 'R', 'L', '0', '1'};
using Pair = std::pair<DreluMaterial, DreluMaterial>;
}  // namespace

std::size_t drelu_bytes_per_element(int n_bits, bool with_product) {
    return 2u * dcf::key_bytes_per_element(n_bits, with_product ? 2 : 1) + 24u;
}

std::size_t DreluMaterial::nbytes() const { return M() * drelu_bytes_per_element(n_bits, with_product()); }

Status DreluMaterial::validate() const {
    if (n_bits < 2 || n_bits > 64) return fail_status("DRELU_N_BITS: n_bits must be in [2, 64]");
    if (party > 1) return fail_status("DRELU_PARTY: party must be 0 or 1");
    Status st = key_p.validate();
    if (!st.ok) return st;
    st = key_q.validate();
    if (!st.ok) return st;
    const std::size_t m = M();
    if (c_b.size() != m || c_rb.size() != m) return fail_status("DRELU_SHAPE: share arrays differ in size");
    if (key_p.M() != m || key_q.M() != m) return fail_status("DRELU_SHAPE: key batches differ in size");
    if (key_p.n_bits != n_bits || key_q.n_bits != n_bits) return fail_status("DRELU_SHAPE: key domain differs from n_bits");
    if (key_p.party != party || key_q.party != party) return fail_status("DRELU_PARTY: keys belong to another party");
    if (key_p.P != key_q.P || key_p.P > 2) return fail_status("DRELU_SHAPE: payload width must be 1 or 2 on both keys");
    if (key_p.modulus != 0 || key_q.modulus != 0) return fail_status("DRELU_SHAPE: keys must use the Z_{2^64} payload group");
    return ok_status();
}

Expected<DreluMaterial> DreluMaterial::slice(std::size_t lo, std::size_t hi) const {
    using R = Expected<DreluMaterial>;
    if (lo > hi || hi > M()) return R::failure("DRELU_SLICE: range out of bounds");
    DreluMaterial out;
    out.n_bits = n_bits;
    out.party = party;
    out.r_share.assign(r_share.begin() + lo, r_share.begin() + hi);
    out.c_b.assign(c_b.begin() + lo, c_b.begin() + hi);
    out.c_rb.assign(c_rb.begin() + lo, c_rb.begin() + hi);
    Expected<dcf::DcfKeyBatch> kp = key_p.slice(lo, hi);
    if (!kp.ok) return R::failure(kp.code);
    Expected<dcf::DcfKeyBatch> kq = key_q.slice(lo, hi);
    if (!kq.ok) return R::failure(kq.code);
    out.key_p = std::move(kp.value);
    out.key_q = std::move(kq.value);
    return R::success(std::move(out));
}

std::vector<uint8_t> DreluMaterial::to_bytes(dcf::WireLayout key_layout) const {
    const std::vector<uint8_t> kp = key_p.to_bytes(key_layout);
    const std::vector<uint8_t> kq = key_q.to_bytes(key_layout);
    std::vector<uint8_t> b;
    b.reserve(WIRE_HEADER_BYTES + 24u * M() + kp.size() + kq.size());
    b.insert(b.end(), kMagic, kMagic + 8);
    dcf::wire::put_u8(b, n_bits);
    dcf::wire::put_u8(b, party);
    dcf::wire::put_u16le(b, with_product() ? 1 : 0);
    dcf::wire::put_u32le(b, 0);
    dcf::wire::put_u64le(b, static_cast<u64>(M()));
    dcf::wire::put_u64le(b, static_cast<u64>(kp.size()));
    dcf::wire::put_u64le(b, static_cast<u64>(kq.size()));
    dcf::wire::put_u64le_array(b, r_share);
    dcf::wire::put_u64le_array(b, c_b);
    dcf::wire::put_u64le_array(b, c_rb);
    b.insert(b.end(), kp.begin(), kp.end());
    b.insert(b.end(), kq.begin(), kq.end());
    return b;
}

Expected<DreluMaterial> DreluMaterial::from_bytes(const uint8_t* data, std::size_t len) {
    using R = Expected<DreluMaterial>;
    if (len < WIRE_HEADER_BYTES) return R::failure("DRELU_WIRE: buffer shorter than the header");
    if (std::memcmp(data, kMagic, 8) != 0) return R::failure("DRELU_WIRE: bad magic");
    DreluMaterial m;
    m.n_bits = data[8];
    m.party = data[9];
    const uint16_t wp = dcf::wire::get_u16le(data + 10);
    const uint32_t flags = dcf::wire::get_u32le(data + 12);
    const u64 M64 = dcf::wire::get_u64le(data + 16);
    const u64 lkp = dcf::wire::get_u64le(data + 24);
    const u64 lkq = dcf::wire::get_u64le(data + 32);
    if (flags != 0) return R::failure("DRELU_WIRE: unsupported flags");
    if (wp > 1) return R::failure("DRELU_WIRE: with_product must be 0 or 1");
    if (M64 > (u64(1) << 40) || lkp > len || lkq > len) return R::failure("DRELU_WIRE: sizes out of range");
    const std::size_t M = static_cast<std::size_t>(M64);
    const std::size_t expect = WIRE_HEADER_BYTES + 24u * M + static_cast<std::size_t>(lkp) + static_cast<std::size_t>(lkq);
    if (len != expect) return R::failure("DRELU_WIRE: length does not match the header");
    std::size_t off = WIRE_HEADER_BYTES;
    dcf::wire::get_u64le_array(data + off, M, m.r_share);
    off += 8u * M;
    dcf::wire::get_u64le_array(data + off, M, m.c_b);
    off += 8u * M;
    dcf::wire::get_u64le_array(data + off, M, m.c_rb);
    off += 8u * M;
    Expected<dcf::DcfKeyBatch> kp = dcf::DcfKeyBatch::from_bytes(data + off, static_cast<std::size_t>(lkp));
    if (!kp.ok) return R::failure(kp.code);
    off += static_cast<std::size_t>(lkp);
    Expected<dcf::DcfKeyBatch> kq = dcf::DcfKeyBatch::from_bytes(data + off, static_cast<std::size_t>(lkq));
    if (!kq.ok) return R::failure(kq.code);
    m.key_p = std::move(kp.value);
    m.key_q = std::move(kq.value);
    if ((m.key_p.P >= 2) != (wp == 1)) return R::failure("DRELU_WIRE: embedded key payload disagrees with the header");
    const Status st = m.validate();
    if (!st.ok) return R::failure(st.code);
    return R::success(std::move(m));
}

Expected<Pair> drelu_material_from_mask(const std::vector<u64>& r, int n_bits, Rng& rng, bool with_product) {
    using R = Expected<Pair>;
    if (n_bits < 2 || n_bits > 64) return R::failure("DRELU_N_BITS: n_bits must be in [2, 64]");
    const std::size_t M = r.size();
    if (M == 0) return R::failure("DRELU_SHAPE: empty batch");
    const std::size_t P = with_product ? 2 : 1;
    const u64 half = u64(1) << (n_bits - 1);
    std::vector<u64> p(M), q(M), beta(M * P), wrap(M), wrap_r(M);
    for (std::size_t m = 0; m < M; ++m) {
        p[m] = ring::reduce_k(r[m], n_bits);
        q[m] = ring::reduce_k(p[m] + half, n_bits);
        beta[m * P] = 1;
        if (with_product) beta[m * P + 1] = r[m];
        wrap[m] = (p[m] > q[m]) ? 1 : 0;
        wrap_r[m] = wrap[m] * r[m];
    }
    Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kp = dcf::gen_dcf_batch(p, beta, n_bits, rng, 0);
    if (!kp.ok) return R::failure(kp.code);
    Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kq = dcf::gen_dcf_batch(q, beta, n_bits, rng, 0);
    if (!kq.ok) return R::failure(kq.code);
    auto rs = sharing::share(r, rng);
    auto cb = sharing::share(wrap, rng);
    DreluMaterial m0, m1;
    m0.n_bits = m1.n_bits = static_cast<uint8_t>(n_bits);
    m0.party = 0;
    m1.party = 1;
    m0.r_share = std::move(rs.first);
    m1.r_share = std::move(rs.second);
    m0.c_b = std::move(cb.first);
    m1.c_b = std::move(cb.second);
    if (with_product) {
        auto crb = sharing::share(wrap_r, rng);
        m0.c_rb = std::move(crb.first);
        m1.c_rb = std::move(crb.second);
    } else {
        m0.c_rb.assign(M, 0);
        m1.c_rb.assign(M, 0);
    }
    m0.key_p = std::move(kp.value.first);
    m1.key_p = std::move(kp.value.second);
    m0.key_q = std::move(kq.value.first);
    m1.key_q = std::move(kq.value.second);
    return R::success(std::make_pair(std::move(m0), std::move(m1)));
}

Expected<Pair> drelu_material(std::size_t M, int n_bits, Rng& rng, bool with_product) {
    std::vector<u64> r(M);
    for (std::size_t m = 0; m < M; ++m) r[m] = rng.next_u64();
    return drelu_material_from_mask(r, n_bits, rng, with_product);
}

Expected<DreluOutput> drelu_eval(const DreluMaterial& mat, const std::vector<u64>& y) {
    using R = Expected<DreluOutput>;
    const Status st = mat.validate();
    if (!st.ok) return R::failure(st.code);
    const std::size_t M = mat.M();
    if (y.size() != M) return R::failure("DRELU_SHAPE: y must have M entries");
    std::vector<u64> yn(M);
    for (std::size_t m = 0; m < M; ++m) yn[m] = ring::reduce_k(y[m], mat.n_bits);
    Expected<std::vector<u64>> oq = dcf::eval_dcf_batch(mat.key_q, yn);
    if (!oq.ok) return R::failure(oq.code);
    Expected<std::vector<u64>> op = dcf::eval_dcf_batch(mat.key_p, yn);
    if (!op.ok) return R::failure(op.code);
    const std::size_t P = mat.key_p.P;
    DreluOutput out;
    out.b.resize(M);
    if (P >= 2) out.rb.resize(M);
    for (std::size_t m = 0; m < M; ++m) {
        out.b[m] = (oq.value[m * P] - op.value[m * P]) + mat.c_b[m];
        if (P >= 2) out.rb[m] = (oq.value[m * P + 1] - op.value[m * P + 1]) + mat.c_rb[m];
    }
    return R::success(std::move(out));
}

Expected<std::vector<u64>> relu_from_drelu(const std::vector<u64>& y, const std::vector<u64>& b_share,
                                           const std::vector<u64>& rb_share) {
    using R = Expected<std::vector<u64>>;
    if (y.size() != b_share.size() || y.size() != rb_share.size()) return R::failure("DRELU_SHAPE: operand sizes differ");
    std::vector<u64> out(y.size());
    for (std::size_t m = 0; m < y.size(); ++m) out[m] = y[m] * b_share[m] - rb_share[m];
    return R::success(std::move(out));
}

Expected<DreluMaterial> concat_drelu(const DreluMaterial& a, const DreluMaterial& b) {
    using R = Expected<DreluMaterial>;
    if (a.n_bits != b.n_bits || a.party != b.party || a.with_product() != b.with_product())
        return R::failure("DRELU_CONCAT: materials differ in n_bits, party or payload");
    DreluMaterial out;
    out.n_bits = a.n_bits;
    out.party = a.party;
    out.r_share = a.r_share;
    out.r_share.insert(out.r_share.end(), b.r_share.begin(), b.r_share.end());
    out.c_b = a.c_b;
    out.c_b.insert(out.c_b.end(), b.c_b.begin(), b.c_b.end());
    out.c_rb = a.c_rb;
    out.c_rb.insert(out.c_rb.end(), b.c_rb.begin(), b.c_rb.end());
    Expected<dcf::DcfKeyBatch> kp = dcf::concat_keys(a.key_p, b.key_p);
    if (!kp.ok) return R::failure(kp.code);
    Expected<dcf::DcfKeyBatch> kq = dcf::concat_keys(a.key_q, b.key_q);
    if (!kq.ok) return R::failure(kq.code);
    out.key_p = std::move(kp.value);
    out.key_q = std::move(kq.value);
    return R::success(std::move(out));
}

}  // namespace drelu
}  // namespace lowlevel
}  // namespace opencrab2
