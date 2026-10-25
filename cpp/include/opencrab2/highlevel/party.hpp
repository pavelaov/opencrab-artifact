// OpenCrab v2 C++17 mirror: Party (mirror of python/opencrab2/highlevel/party.py).
//
// One semi-honest party of the two-party computation. Both parties run the
// same code; `party` (0 or 1) selects the asymmetric steps. Every online
// message goes through the Channel (exact byte and round accounting); every
// unit of correlated randomness comes from the DealerClient, which journals
// RESERVED / SPENT in the MaterialLedger before the material is used.
//
// Shares are held as Mat (row-major uint64 matrices over Z_{2^64}); fixed point
// uses the NumericContract (f fractional bits, logical width l, DCF domain n).
// Message sizes depend only on public shapes (no secret-dependent branching
// that changes a message).
//
// Realizations (paper, Section 5):
//   share_input / reveal / reveal_final (ReleasePolicy: only the declared
//   recipient, kinds, max count), add / sub / public ops, beaver_mul (1 round),
//   mul_fxp (Beaver + SecureML local truncation), trunc_faithful (one round,
//   faithful gate), matmul_shared (matrix triple), linear_generic (shared W),
//   linear_wmask (installed E = W - B, opens X - A only), install_wmask (real
//   one-round protocol: P1 sends B1, P0 returns E = W - B0 - B1; versioned by
//   (model_version, client_key_domain) in the ledger), drelu / relu (one round,
//   two DCF keys), compare_public, select, k-ary max_rows / argmax_row (k in
//   {2, 4, 8}, lowest index on ties, identical output for every k), exp_neg,
//   recip, rsqrt, softmax_rows, layernorm_rows (faithful square), gelu / silu /
//   tanh (clipped polynomial), poly_horner, ring_to_field (A64 -> Zt).
// Counters per gadget: wall seconds, bytes, rounds, material bytes (stats).
//
// k-ary tournament (max_rows, argmax_row): the FUSED two-round node of
// python/opencrab2/highlevel/party.py (Party._tournament), ported line by line
// (paper, Section 5.3 and Appendix B). A level groups the w surviving candidates
// into nodes of k (the last node may be smaller). Per level, exactly two
// openings (2 ceil(log_k n) rounds in total):
//   round 1  every pair (i < j) inside a node is compared once with a DReLU
//            WITHOUT payload (P = 1) on x_i - x_j: g (g - 1) / 2 comparisons per
//            node of g members. A pair with a publicly masked member is decided
//            publicly (no comparison, no message bytes). Locally every member
//            gets its rank r_i = #{j : i beats j}; ties go to the lower index,
//            so the ranks of a node are a permutation of 0 .. g-1.
//   round 2  a node of 2 members spends one Beaver select (material type
//            kary_select) for the value, plus one for the index once indices
//            are secret (every level after the first); a node of g >= 3 members
//            spends, per non-leading member i >= 1, one DReLU WITH product
//            payload (P = 2) on
//                u_i = (r_i - (g - 1)) M + (x_i - x_0) + M / 2,
//                M = 2^(n - 1 - ceil(log2 g)),  n = contract.dcf_domain_bits,
//            so that 1{u_i >= 0} = 1{r_i = g - 1} = w_i and
//            ReLU(u_i) - (M / 2) w_i = w_i (x_i - x_0). The node maximum is
//            x_0 + sum_i w_i (x_i - x_0); the node argmax is
//            idx_0 + sum_i w_i (idx_i - idx_0) (a second payload DReLU batch on
//            the index differences when the indices are shared, a local sum at
//            the first level where they are public). All round-2 openings of a
//            level travel in ONE message per direction.
// Opened elements per level (8 bytes each, per direction): round 1 one element
// per live pair; round 2 two elements per Beaver select and one per payload
// DReLU. Material per level: P = 1 DReLU gates, P = 2 DReLU gates and
// kary_select triples in exactly these counts (costmodel.hpp tournament_counts,
// dealer.hpp tournament_material).
//
// MAGNITUDE BOUND (paper, Section 5.3 and Appendix B; a precondition, not a
// run-time check, because a check on secret values would leak): inside a
// fused node of g >= 3 members every compared difference must satisfy
//     |x_i - x_j| < M / 2 = 2^(n - 2 - ceil(log2 g)) ring units,
// that is 2^12 real units for g in {3, 4} and 2^11 real units for g in {5 .. 8}
// at n = 32, f = 16; binary nodes keep the full DCF bound 2^(n - 1). The same
// bound applies to the index differences of the argmax (a public quantity:
// the tournament refuses a row longer than the bound with KARY_MAGNITUDE).
// The bound is enforced at admission from the numeric contract and the
// declared public operand bound (candidates.hpp, code KARY_MAGNITUDE). Inside
// the bound the outputs are exact integers and therefore bit-identical for
// every k, with the lowest index winning ties.
#pragma once

#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/contract.hpp>
#include <opencrab2/highlevel/dealer.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace opencrab2 {
namespace highlevel {

// Row-major uint64 matrix (shares or public ring values).
struct Mat {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<uint64_t> a;

    Mat() = default;
    Mat(size_t r, size_t c, uint64_t fill = 0) : rows(r), cols(c), a(r * c, fill) {}
    static Mat from_vector(std::vector<uint64_t> v, size_t rows, size_t cols);
    static Mat row_vector(std::vector<uint64_t> v) {
        size_t n = v.size();
        return from_vector(std::move(v), 1, n);
    }
    size_t size() const { return a.size(); }
    bool empty() const { return a.empty(); }
    uint64_t& at(size_t i, size_t j) { return a[i * cols + j]; }
    uint64_t at(size_t i, size_t j) const { return a[i * cols + j]; }
    bool same_shape(const Mat& o) const { return rows == o.rows && cols == o.cols; }
    Mat row(size_t i) const;
    Mat rows_range(size_t lo, size_t hi) const;
    Mat cols_range(size_t lo, size_t hi) const;
    Mat transpose() const;
    Mat reshape(size_t r, size_t c) const;
    static Mat vstack(const Mat& a, const Mat& b);
    static Mat hstack(const Mat& a, const Mat& b);
    // Flattened concatenation of several matrices (1 x total).
    static Mat concat_flat(const std::vector<const Mat*>& ms);
};

Mat ring_add(const Mat& a, const Mat& b);
Mat ring_sub(const Mat& a, const Mat& b);
Mat ring_neg(const Mat& a);
Mat ring_mul(const Mat& a, const Mat& b);           // elementwise
Mat ring_mul_const(const Mat& a, uint64_t c);
Mat ring_matmul(const Mat& A, const Mat& B);         // exact mod 2^64

class ReleasePolicy {
public:
    ReleasePolicy() = default;
    ReleasePolicy(int recipient, std::vector<std::string> kinds, int max_count);
    // Fails with UNAUTHORIZED_RELEASE when kind is not declared or the count is exhausted.
    Expected<Unit> authorize(const std::string& kind);
    int recipient() const { return recipient_; }
    int count() const { return count_; }
    int max_count() const { return max_count_; }
    int remaining() const { return max_count_ - count_; }
    bool declared() const { return declared_; }
    Value to_value() const;

private:
    bool declared_ = false;
    int recipient_ = 1;
    std::vector<std::string> kinds_;
    int max_count_ = 0;
    int count_ = 0;
};

struct GadgetStat {
    std::string kind;
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t rounds = 0;
    uint64_t material_bytes = 0;
    std::vector<size_t> shape;
    Value to_value() const;
};

class Party {
public:
    Party(int party, Channel& channel, DealerClient& dealer, const NumericContract& contract, ReleasePolicy release,
          ll::Rng rng, std::string model_version = "model-0", std::string client_key_domain = "client-0");

    int party() const { return party_; }
    int f() const { return contract_.frac_bits; }
    const NumericContract& contract() const { return contract_; }
    Channel& channel() { return ch_; }
    DealerClient& dealer() { return dealer_; }
    ReleasePolicy& release() { return release_; }
    void set_gate(const std::string& gate_id, int64_t epoch);

    // --- fixed point ---------------------------------------------------------------
    uint64_t enc(double x) const;                    // rint(x 2^f) mod 2^64 (no range check)
    double dec(uint64_t v) const;                    // signed / 2^f
    Expected<Mat> encode(const std::vector<double>& x, size_t rows, size_t cols) const;  // CONTRACT_VIOLATION on saturation
    std::vector<double> decode(const Mat& m) const;
    Mat const_mat(double value, size_t rows, size_t cols) const;    // P0 holds the encoded constant, P1 zeros
    Mat const_ring(uint64_t value, size_t rows, size_t cols) const; // P0 holds the ring constant, P1 zeros

    // --- sharing / opening --------------------------------------------------------
    // Owner passes its encoded value; the peer passes nullptr. One message owner -> peer.
    Expected<Mat> share_input(const Mat* value_enc, int owner, size_t rows, size_t cols);
    Expected<Mat> reveal(const Mat& share);          // both learn the value (one exchange)
    // Release to the declared recipient through the ReleasePolicy; the recipient
    // receives the value, the other party an empty Mat.
    // `flat` sends the share as a 1-D array of share.size() elements (the wire shape of the Python
    // party for an argmax index vector, one element per row); otherwise as a (rows, cols) array.
    Expected<Mat> reveal_final(const Mat& share, const std::string& kind, bool flat = false);

    // --- local linear algebra ---------------------------------------------------------
    Mat add(const Mat& a, const Mat& b) const { return ring_add(a, b); }
    Mat sub(const Mat& a, const Mat& b) const { return ring_sub(a, b); }
    Mat add_public(const Mat& share, const Mat& pub) const;        // only P0 folds it in
    Mat add_public_real(const Mat& share, double value) const;
    Mat mul_public(const Mat& share, uint64_t pub) const { return ring_mul_const(share, pub); }
    Mat mul_public_real(const Mat& share, double value) const;     // frexp mantissa, pre-shift, truncation
    Mat matmul_public(const Mat& share, const Mat& pub, bool pub_on_left = false) const;

    // --- truncation -------------------------------------------------------------------
    // SecureML local truncation, 0 rounds: P0 floor(s0 / 2^f), P1 -floor(-s1 / 2^f), with the
    // arithmetic shift of lowlevel::ring::arith_shift_right (no signed right shift).
    Mat trunc_local(const Mat& share, int f = -1) const;
    Expected<Mat> trunc_faithful(const Mat& share, int f = -1, int L = -1);  // one round, exact

    // --- Beaver products ------------------------------------------------------------
    Expected<Mat> beaver_mul(const Mat& sx, const Mat& sy);
    // Several products in ONE round (all openings batched).
    Expected<std::vector<Mat>> beaver_mul_batch(const std::vector<Mat>& xs, const std::vector<Mat>& ys);
    Expected<Mat> mul_fxp(const Mat& sx, const Mat& sy, int trunc_bits = -1);
    Expected<Mat> square_fxp(const Mat& sx, bool exact = false);

    // --- matrix products ------------------------------------------------------------
    Expected<Mat> matmul_shared(const Mat& sX, const Mat& sY, bool trunc = true);
    Expected<std::vector<Mat>> matmul_shared_batch(const std::vector<Mat>& Xs, const std::vector<Mat>& Ys, bool trunc = true);

    // --- linear layers -----------------------------------------------------------------
    // Server-private vectors / matrices held as shares (registered by share_input or the dealer).
    void set_vector_share(const std::string& name, Mat share);
    bool has_vector_share(const std::string& name) const;
    Expected<Mat> vector_share(const std::string& name);
    Expected<Mat> linear_generic(const Mat& sX, const std::string& mat_id, bool trunc = true);
    Expected<Mat> linear_wmask(const Mat& sX, const std::string& mat_id, bool trunc = true);
    // Real one-round install; P0 passes W_enc (d x m), P1 nullptr. Returns
    // {mat_id, bytes, seconds, rounds, version, skipped, shape}.
    Expected<Value> install_wmask(const std::string& mat_id, const Mat* W_enc_or_null);
    bool installed(const std::string& mat_id) const;
    Value installed_objects() const;
    Value version() const;
    static std::string install_object_id(const std::string& mat_id) { return "wmask_install(" + mat_id + ")"; }

    // --- comparison layer (one round) ------------------------------------------------
    Expected<Mat> drelu(const Mat& x);                       // shares of 1{x >= 0}
    Expected<Mat> relu(const Mat& x);
    Expected<Mat> compare_public(const Mat& x, double c, const std::string& op = "ge");  // ge | lt
    Expected<Mat> select(const Mat& bit, const Mat& a, const Mat& b);   // b + bit (a - b)

    // --- tournaments -------------------------------------------------------------------
    // mask (rows x cols, 1 = valid) is public; every row needs one valid entry.
    Expected<Mat> max_rows(const Mat& x, const std::vector<uint8_t>* mask = nullptr, int k = 2);
    // Lowest-index argmax per row. release = true: the index is released to the
    // ReleasePolicy recipient (P1 gets a 1 x rows Mat, P0 an empty Mat);
    // release = false: both get the index SHARES (nothing revealed).
    Expected<Mat> argmax_row(const Mat& x, int k = 2, bool release = true, const std::string& kind = "token",
                             const std::vector<uint8_t>* mask = nullptr);

    // --- polynomials and nonlinear gadgets (gadgets.cpp) ---------------------------------
    Expected<Mat> poly_horner(const Mat& u, const std::vector<double>& coeffs);
    Expected<Mat> poly_contract(const Mat& x, const std::string& name);
    Expected<Mat> clip_poly_act(const Mat& x, const std::string& name);
    Expected<Mat> gelu(const Mat& x);
    Expected<Mat> silu(const Mat& x);
    Expected<Mat> tanh(const Mat& x);
    Expected<Mat> exp_neg(const Mat& z);
    Expected<Mat> recip(const Mat& S, int extra_bits = 0);
    Expected<Mat> rsqrt(const Mat& V, int extra_bits = 0);
    Expected<Mat> softmax_rows(const Mat& x, const std::vector<uint8_t>* mask = nullptr, int k_max = 2);
    Expected<Mat> layernorm_rows(const Mat& x, const Mat* gamma = nullptr, const Mat* beta = nullptr, double eps = 1e-5,
                                 bool rms = false);

    // --- conversions ----------------------------------------------------------------
    Expected<Mat> ring_to_field(const Mat& share, uint64_t t);   // Z_t shares of x_signed mod t

    // --- counters -------------------------------------------------------------------
    std::vector<GadgetStat> stats;      // top-level gadget calls
    std::vector<GadgetStat> all_stats;  // including nested calls
    std::vector<Value> install_log;
    std::vector<Value> release_log;
    Value stats_value() const;
    uint64_t material_bytes() const { return dealer_.bytes_consumed(); }

    // RAII tracker used by every gadget (public so that gadgets.cpp can use it).
    struct Track {
        Track(Party& p, std::string kind, std::vector<size_t> shape);
        ~Track();
        Party& p;
        std::string kind;
        std::vector<size_t> shape;
        double t0;
        uint64_t b0, r0, m0;
    };

    // Shared helpers for gadgets.cpp.
    Expected<Mat> open(const Mat& share);  // one exchange round
    // One exchange round of a flat (1-D) vector, the wire shape of the Python
    // tournament openings (numpy 1-D array): 8 + 11 + 8 N bytes per direction.
    Expected<std::vector<uint64_t>> open_flat(const std::vector<uint64_t>& share);
    void count_round() { ch_.count_round(); }

    // Fused k-ary node geometry (public quantities only).
    static int kary_ceil_log2(size_t g);                          // ceil(log2 g), 0 for g <= 1
    // M = 2^(n_bits - 1 - ceil(log2 g)); 0 when the exponent would be negative.
    static uint64_t kary_node_modulus(int n_bits, size_t g);
    // Exclusive bound on |x_i - x_j| inside a node of g members, in ring units:
    // 2^(n_bits - 1) for g <= 2, M / 2 = 2^(n_bits - 2 - ceil(log2 g)) for g >= 3.
    static uint64_t kary_difference_bound(int n_bits, size_t g);
    Expected<std::vector<Mat>> threshold_bits(const Mat& S, const std::vector<double>& thresholds, std::vector<bool>& live);

private:
    struct BeaverPrep {
        Mat a, b, c;
    };
    Expected<std::pair<Mat, Mat>> drelu_raw(const Mat& x_flat, bool with_product, Mat* rb_out);
    Expected<std::pair<Mat, Mat>> tournament(const Mat& x, const std::vector<uint8_t>* mask, int k, bool with_index);
    // The mask shape is public: P0 takes it from W, P1 from the registered weight share.
    Expected<Mat> wmask_mask(const std::string& mat_id, size_t d, size_t m);

    int party_;
    Channel& ch_;
    DealerClient& dealer_;
    NumericContract contract_;
    ReleasePolicy release_;
    ll::Rng rng_;
    std::string model_version_;
    std::string client_key_domain_;
    int depth_ = 0;
    std::map<std::string, Mat> vectors_;
    std::map<std::string, Mat> wmask_b_;
    struct Installed {
        Mat E;
        Value version;
    };
    std::map<std::string, Installed> installed_;
};

// Extra scale bits e2 of the rsqrt selector s2 (held at scale 2^(f + e2)): max(0, 2 (j0+1+m) - f),
// the smallest scale at which every 4^-(j0+1+k), k <= m, is an integer. The normalisation product
// V * s2 is then below 2^rsqrt_product_bits ring units (34 under NC-v2), inside the declared bound
// of the local truncation (paper, Appendix D, Disclosure).
int rsqrt_scale_bits(const NumericContract& c);
int rsqrt_product_bits(const NumericContract& c);

}  // namespace highlevel
}  // namespace opencrab2
