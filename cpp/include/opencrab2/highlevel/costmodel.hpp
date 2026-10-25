// OpenCrab v2: calibrated per-component cost model (paper, Section 4.2 and Section 7.3; mirror of
// python/opencrab2/highlevel/costmodel.py).
//
// Affine per-component model
//     t = (a * MACs + c * bytes + b) / cpu_share + bytes * 8 / BW + rounds * RTT
// with (a, c, b) = (compute_per_mac_s, compute_per_byte_s, compute_fixed_s) fitted by NNLS on
// calibration rows, and bytes / rounds / material given by the deterministic protocol formulas
// below. Uncertainty: sigma = sqrt((sigma_compute / cpu_share)^2 + (comm * rel_bw)^2 +
// (rounds * RTT * rel_rtt)^2).
//
// Coefficient file (results/calibration/fitted_model.json, written by the Python calibration):
//     {"label": "MEASURED_NATIVE", "rel_bw": 0.1, "rel_rtt": 0.1,
//      "components": {name: {"compute_per_mac_s": a, "compute_per_byte_s": c, "compute_fixed_s": b,
//                            "sigma_compute_s": s, "n_train": k, "status": ..., "source": ...}}}
// The coefficients were fitted with the MAC and byte conventions of the Python model, so this file
// uses exactly the same conventions (shapes, MACs, framing). default_model() loads the file when it
// is present and falls back to the HISTORICAL prior, reporting which one it returned. A component
// missing from a model cannot be priced (predict fails with UNKNOWN_COMPONENT or
// COSTMODEL_NOT_FITTED); nothing is approximated silently.
//
// Shared formula file (results/calibration/formulas.json, schema opencrab2-formulas-v1, written
// by python costmodel.write_formulas_json).
// It carries the framing bytes per message (framing.frame_bytes), the MEASURED BFV ciphertext size
// of a2h_exact (bfv.ct_bytes) and, for a declared list of shapes, the payload bytes, messages,
// bytes, rounds, gate counts and material of every component plus the gate counts of the fused
// tournament. load_formulas() reads the shared constants (frame bytes, ciphertext size, slot count)
// into this model (the built-in values below equal the file of this tree, so the two languages
// agree even when the file is not found);
// check_formulas() compares EVERY vector of the file with this implementation, field by field, so
// a divergence between the two languages shows up as a named mismatch instead of a silent one
// (tests/test_decision.cpp requires zero mismatches whenever the file exists).
//
// =============================================================================================
// FORMULA TABLE (the C++ twin of python/opencrab2/highlevel/costmodel.py gate_counts)
//
// Shape conventions (n, d_in, d_out), every used dimension >= 1:
//   linear_wmask, linear_generic, matmul_shared      X (n x d_in) times W (d_in x d_out)
//   embedding_wmask, embedding_generic               n one-hot rows, d_in = vocabulary, d_out = d_model
//   install_wmask                                    one matrix d_in x d_out (n is ignored)
//   gelu_clip12, trunc_faithful, ring_to_field,
//   a2h_exact                                        elementwise on N = n * d_in elements (d_out ignored)
//   softmax_rows_k2 / k4                             n rows of length d_in (keys)
//   layernorm_rows                                   n rows of width d_in, with the affine gamma product
//   argmax_k2 / k4 / k8                              n rows of d_in logits
// MACs (work unit of the compute term): n * d_in * d_out for the five linear components,
// d_in * d_out for install_wmask, the element count n * d_in for everything else.
//
// Every formula derives from gadget_counts, which counts exactly the gates Party issues:
//   payload_bytes = 16 (drelu + drelu_p2 + trunc_faithful + ring_to_field + open_elems)
//                 + 32 (beaver + kary_select) + 8 oneway_elems + bfv_ciphertext_bytes * bfv_ct
//   bytes_formula = payload_bytes + frame_bytes * messages        (frame_bytes = 19)
// One opened ring element is 16 bytes (8 from each party); a Beaver product opens two masked
// elements (32 bytes); a one-way element is 8 bytes; every message is one frame whose framing is
// charged at the 1-D uint64 array figure of the wire codec (8-byte prefix, tag, dtype, ndim, one u64
// dimension = 19 bytes). messages = 2 per exchange round; a2h_exact has 3 (one exchange plus the
// ciphertext frame). Message sizes depend on public shapes only.
//
// Material bytes per party (the serializations of the lowlevel headers):
//   DCF key                     key(nb, P) = 16 + nb * (16 + 8 P + 1) + 8 P
//   DReLU, comparison (P = 1)   2 key(n_dcf, 1) + 24   (= 1672 at n_dcf = 32)
//   DReLU, payload    (P = 2)   2 key(n_dcf, 2) + 24   (= 2200 at n_dcf = 32)
//   Beaver product / select     24
//   trunc_faithful(f)           16 + key(64, 1) + key(f, 1)   (= 2064 at f = 16)
//   ring_to_field(t)            24 + 2 key(64, 1)             (= 3272)
//   matrix triple element 8 ; weight-mask element 8 (per-call A, C and the reusable mask B)
// material_by_type uses the cost-model vocabulary shared with Python: beaver, kary_select, drelu
// (P = 1 and P = 2 gates summed with their own sizes), trunc_faithful, ring_to_field,
// matrix_triple, wmask_mask (the per-call shares of a weight-masked linear AND the mask an install
// consumes; the dealer delivers them as the bundle types wmask_call and wmask_mask).
//
// Gadget counts from the contract (default: exp_r = 7, exp_order = 2, recip_K = 11, recip_degree
// = 5, recip_newton = 2, rsqrt_m = 15, rsqrt_j0 = -7, rsqrt_degree = 5, rsqrt_newton = 2, gelu
// degree 12). A threshold comparison is LIVE when the threshold lies below the DCF bound
// 2^(n-1-f) real units; the others are answered with a public 0 and cost nothing:
//   exp_neg per element    comparison 1, Beaver 1 + [order >= 2] + r (= 9), rounds 1 + Beaver (= 10)
//   recip per row          comparison live(2^1 .. 2^K) (= 11), Beaver 1 + deg + 2 newton + 1 (= 11),
//                          rounds [live > 0] + Beaver (= 12)
//   rsqrt per row          comparison live(4^(j0+1) .. 4^(j0+m)) (= 14 of 15), Beaver 1 + deg +
//                          3 newton + 1 (= 13), rounds [live > 0] + Beaver (= 14)
//   gelu per element       comparison 2, Beaver deg + 2 (= 14), rounds 1 + deg + 1 (= 14)
//
// FUSED k-ary tournament over one row of L candidates (tournament_counts; exactly what
// Party::tournament and python Party._tournament issue; dealer.hpp tournament_material and python
// costmodel.tournament_gates give the same counts). Per level the w survivors form nodes of k
// (last node smaller); per node of g members:
//     round 1   g (g - 1) / 2 pairwise comparisons (DReLU, P = 1)
//     round 2   g = 2 : 1 Beaver select for the value, + 1 for the index when it is secret
//               g >= 3: g - 1 payload DReLUs (P = 2) for the value, + g - 1 for a secret index
// The index is public at the first level and secret afterwards. levels = ceil(log_k L),
// rounds = 2 levels, payload = 16 (p1 + p2) + 32 selects per row. The formulas price the unmasked
// row; a public mask only removes pairwise comparisons.
//
// Components (n, d_in, d_out) -> gates ; rounds:
//   linear_wmask        wmask_call n d_in + n d_out, open n d_in ; 1
//   linear_generic      matrix_triple n d_in + d_in d_out + n d_out, open n d_in + d_in d_out ; 1
//   embedding_wmask     = linear_wmask with d_in = vocabulary
//   embedding_generic   = linear_generic with d_in = vocabulary (both directions counted)
//   matmul_shared       = linear_generic
//   install_wmask       wmask_mask d_in d_out, one-way 2 d_in d_out (B1 up, E down) ; 1
//   gelu_clip12         comparison 2 N, Beaver 14 N (payload 480 N) ; 14
//   layernorm_rows      N = n d_in: Beaver 3 N + 13 n (square, dx * rs, gamma; rsqrt), comparison
//                       14 n, faithful truncation N (payload 112 N + 640 n) ; 1 + 1 + 14 + 1 + 1 = 18
//   softmax_rows_k2/k4  N = n d_in: tournament(d_in, k) per row + exp per element + recip per row
//                       + 1 product per element ; 2 levels + 10 + 12 + 1
//   argmax_k2/k4/k8     tournament(d_in, k, index) per row ; 2 levels. The RELEASE_FINAL message of
//                       the index share (8 bytes per row, one message, one round) is a separate
//                       step and is not part of the component.
//   trunc_faithful      N gates ; 1          ring_to_field  N gates ; 1
//   a2h_exact           ring_to_field N + ceil(N / bfv_slots) ciphertexts in ONE message P1 -> P0 ;
//                       2 rounds, 3 messages; the ciphertext size is MEASURED (432404 bytes at
//                       N = 8192, t = 68720050177; the TenSEAL serializer compresses, so the size
//                       varies by about 0.05 percent around the declared value)
// =============================================================================================
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/highlevel/contract.hpp"
#include "opencrab2/highlevel/json.hpp"

namespace opencrab2 {
namespace highlevel {

struct Env {
    double bandwidth_bps = 0.0;        // 0 = unlimited (loopback)
    double one_way_latency_s = 0.0;
    double cpu_share = 1.0;            // fraction of a core available (S4 contention traces)
    double rtt_s() const { return 2.0 * one_way_latency_s; }
    Json to_json() const;
    static Expected<Env> from_json(const Json& j);
};

// Built-in values of the shared constants: the values of results/calibration/formulas.json of
// this tree (python costmodel.FRAME_BYTES, BFV_N, BFV_CT_BYTES), used when the file is not found.
constexpr std::int64_t kFrameBytes = 19;               // framing of one message (1-D uint64 array frame)
constexpr std::int64_t kOpenBytes = 16;                // one opened ring element: 8 bytes from each party
constexpr std::int64_t kBeaverOpenBytes = 32;          // one Beaver product opens two masked elements
constexpr std::int64_t kOnewayBytes = 8;               // one ring element sent in one direction only (install)
constexpr std::int64_t kBfvSlots = 8192;
constexpr std::int64_t kBfvCiphertextBytes = 432404;   // measured with TenSEAL 0.3.17 on 2026-10-10

// ---------------------------------------------------------------------------------------------
// Shared constants (results/calibration/formulas.json)
// ---------------------------------------------------------------------------------------------
struct FormulaConstants {
    std::int64_t frame_bytes = kFrameBytes;
    std::int64_t bfv_slots = kBfvSlots;
    std::int64_t bfv_ciphertext_bytes = kBfvCiphertextBytes;
    std::string source = "builtin";            // "builtin" or the path of the loaded formulas.json
    std::string a2h_status = "MODELED";        // evidence label of the ciphertext size: bfv.status of the file once it is
                                               // loaded (MEASURED_NATIVE); the built-in copy alone stays MODELED
    Json to_json() const;
};

FormulaConstants formula_constants();                       // snapshot of the process-wide table
Status set_formula_constants(const FormulaConstants& c);    // FORMULAS_PARAMS on a non-positive value
void reset_formula_constants();                             // back to the built-in values

// Result of reading formulas.json. found = false means the file does not exist (not an error:
// the built-in constants stay in force and nothing is compared).
struct FormulasReport {
    std::string path;
    bool found = false;
    std::vector<std::string> adopted;          // constants taken from the file, "name=value"
    std::int64_t vectors_compared = 0;         // component vectors with at least one recognized field
    std::int64_t tournament_compared = 0;      // tournament count vectors
    std::int64_t fields_compared = 0;
    std::vector<std::string> mismatches;       // "argmax_k4 (1, 50257, 1) rounds: file 16, cpp 18"
    std::vector<std::string> notes;            // unrecognized but harmless content
    Json to_json() const;
};

// Search order for a calibration file: dir_hint/name when dir_hint is not empty, then
// $OPENCRAB2_CALIBRATION_DIR/name, then results/calibration/name relative to the working
// directory, its parent and its grandparent. Returns "" when no candidate exists.
std::string find_calibration_file(const std::string& name, const std::string& dir_hint = "");

// Reads the shared constants of formulas.json into the process-wide table. path = "" searches as
// above. Recognized constants (in the top-level object or up to three objects below it, names
// compared case-insensitively and without punctuation): framing.frame_bytes; bfv.ct_bytes (also
// bfv_ct_bytes, ciphertext_bytes, a2h_exact_ciphertext_bytes); constants.BFV_N /
// bfv.slots_per_ciphertext. The protocol facts OPEN_BYTES (16), BEAVER_OPEN_BYTES / SELECT_BYTES (32)
// and ONEWAY_BYTES (8), when present, must equal the values of this implementation. Codes:
// FORMULAS_IO (unreadable), FORMULAS_JSON (not a JSON object, or a schema other than
// opencrab2-formulas-v1), FORMULAS_PARAMS (a non-positive or non-integer constant),
// FORMULAS_MISMATCH (a protocol fact differs). On any failure the table is left unchanged.
Expected<FormulasReport> load_formulas(const std::string& path = "");

// load_formulas, then compares every vector of the file with this implementation: the material
// sizes of "constants", the derived counts and the hash of "contract", and every entry of the
// vector lists (arrays anywhere down to depth 4 whose elements are objects with the keys below):
//   component vectors   {component, n, d_in [, d_out]} with any of macs, bytes (framed total),
//                       payload_bytes, rounds, messages, material_bytes, material ({type: bytes}
//                       or a number), gates ({drelu_p1, drelu_p2, beaver, kary_select, trunc, r2f,
//                       matrix_triple, wmask_call, wmask_mask, open_elems, oneway_elems, bfv_ct,
//                       rounds, messages})                                   ("shapes" of the file)
//   tournament vectors  {k, n | L | V [, with_index]} with any of drelu_p1, drelu_p2,
//                       beaver_select, levels, rounds, payload_bytes         ("tournament" of the file)
// Every difference is listed in mismatches (the call itself still succeeds, so that a caller can
// report all of them); a missing file gives found = false.
Expected<FormulasReport> check_formulas(const std::string& path = "", const NumericContract& contract = default_contract());

// The C++ side of the shared document in the layout of the Python file (framing, bfv, constants,
// contract, shapes, tournament) on a fixed grid of shapes. Used by the cross-language parity pass.
Json formulas_document(const NumericContract& contract = default_contract());

// ---------------------------------------------------------------------------------------------
// Deterministic protocol formulas
// ---------------------------------------------------------------------------------------------
// Exact component list of the Python costmodel.py (COMPONENTS).
const std::vector<std::string>& components();
bool is_component(const std::string& name);

// Gates, openings, rounds and messages of ONE call (the twin of python costmodel.GateCounts).
struct GadgetCounts {
    std::int64_t drelu = 0;            // comparisons, DReLU without payload (P = 1; python drelu_p1)
    std::int64_t drelu_p2 = 0;         // DReLU with product payload (P = 2): fused tournament nodes
    std::int64_t beaver = 0;           // elementwise products from the general pool
    std::int64_t kary_select = 0;      // Beaver selects of binary tournament nodes
    std::int64_t trunc_faithful = 0;   // faithful truncation gates (python trunc)
    std::int64_t ring_to_field = 0;    // A64 -> Zt gates (python r2f)
    std::int64_t matrix_triple = 0;    // matrix triple elements (A, B, C)
    std::int64_t wmask_call = 0;       // per-call weight-mask elements (A, C)
    std::int64_t wmask_mask = 0;       // reusable mask elements (B) consumed by an install
    std::int64_t open_elems = 0;       // other ring elements opened by both parties (linear openings)
    std::int64_t oneway_elems = 0;     // ring elements sent in one direction only (install: B1 up, E down)
    std::int64_t bfv_ct = 0;           // BFV ciphertexts sent P1 -> P0
    std::int64_t rounds = 0;
    std::int64_t messages = 0;         // frames over both directions
};

// Gates of the fused k-ary tournament over ONE row of length L; with_index adds the index path of
// argmax. k must be 2, 4 or 8 and L >= 1 (zero counts otherwise).
struct TournamentCounts {
    std::int64_t comparisons = 0;      // pairwise DReLUs without payload (round 1)
    std::int64_t payload_drelu = 0;    // DReLUs with product payload of the fused nodes (round 2)
    std::int64_t selects = 0;          // Beaver selects of the binary nodes (round 2)
    std::int64_t rounds = 0;           // 2 * levels
    std::int64_t levels = 0;           // ceil(log_k L)
};
TournamentCounts tournament_counts(std::int64_t L, int k, bool with_index);

// Fused node geometry and the magnitude condition (paper, Section 5.3; public quantities only).
int kary_ceil_log2(std::int64_t g);                                       // ceil(log2 g), 0 for g <= 1
// Exclusive bound on a compared difference inside a node of g members, in ring units:
// 2^(n - 1) for g <= 2, 2^(n - 2 - ceil(log2 g)) for g >= 3 (0 when the exponent is negative).
std::int64_t kary_difference_bound_ring(int dcf_domain_bits, std::int64_t g);
// Largest declared operand magnitude (real units) for which a k-ary tournament over rows of
// length L is admitted: every |x| < B implies |x_i - x_j| < 2 B, so B_max = bound(g_max) / (2 * 2^f)
// with g_max = min(k, L).
double kary_operand_bound_real(const NumericContract& contract, int k, std::int64_t L);
// ok when |x| <= declared_bound_real keeps every compared difference of the tournament inside the
// node bound (and, with_index, the index differences up to L - 1 as well); otherwise the stable
// code KARY_MAGNITUDE with the numbers in the detail.
Status kary_magnitude_check(const NumericContract& contract, int k, std::int64_t L, double declared_bound_real, bool with_index);

std::int64_t dcf_key_bytes(int n_bits, int payloads);                   // 16 + n*(16 + 8P + 1) + 8P
std::int64_t drelu_material_bytes(const NumericContract& c, int payloads = 1);   // 2*key(n_dcf, P) + 24
std::int64_t trunc_material_bytes(const NumericContract& c);            // 16 + key(64,1) + key(f,1)
std::int64_t r2f_material_bytes();                                      // 24 + 2*key(64,1)

// Codes: UNKNOWN_COMPONENT, COSTMODEL_SHAPE (a used dimension below 1).
Expected<GadgetCounts> gadget_counts(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out,
                                     const NumericContract& contract);

Expected<std::int64_t> macs(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out);
// Opened payload bytes, both directions, without framing.
Expected<std::int64_t> payload_bytes(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out);
// payload_bytes + frame_bytes * messages: the byte figure the predictor uses.
Expected<std::int64_t> bytes_formula(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out);
Expected<std::int64_t> rounds_formula(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out);
// Frames of one call over both directions (two per exchange round; a2h_exact: two plus the ciphertext frame).
Expected<std::int64_t> messages_formula(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out);
Expected<std::int64_t> material_formula(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out,
                                        const NumericContract& contract);
// Material by type in the cost-model vocabulary shared with Python (keys: beaver, kary_select,
// drelu, trunc_faithful, ring_to_field, matrix_triple, wmask_mask; wmask_mask covers the per-call
// weight-mask shares and the mask of an install). Types with zero need are omitted.
Expected<std::map<std::string, std::int64_t>> material_by_type(const std::string& component, std::int64_t n, std::int64_t d_in,
                                                               std::int64_t d_out, const NumericContract& contract);

// ---------------------------------------------------------------------------------------------
// Fitted model
// ---------------------------------------------------------------------------------------------
struct ComponentFit {
    double compute_per_mac_s = 0.0;
    double compute_per_byte_s = 0.0;
    double compute_fixed_s = 0.0;
    double sigma_compute_s = 0.0;
    std::int64_t n_train = 0;
    std::string status = "MEASURED_NATIVE";   // MEASURED_NATIVE | HISTORICAL | MODELED (evidence label)
    std::string source;
    Json to_json() const;
    static Expected<ComponentFit> from_json(const Json& j);
};

struct CalibrationRow {
    std::string component;
    std::int64_t n = 0, d_in = 0, d_out = 0;
    double bandwidth_bps = 0.0;
    double one_way_latency_s = 0.0;
    double cpu_share = 1.0;
    double measured_s = 0.0;
    std::int64_t bytes = 0;    // measured channel counter
    std::int64_t rounds = 0;
};

struct Prediction {
    double mean_s = 0.0;
    double sigma_s = 0.0;
    double compute_s = 0.0;
    double comm_s = 0.0;
    double round_s = 0.0;
    std::int64_t bytes = 0;    // bytes_formula (payload + framing)
    std::int64_t rounds = 0;
};

class FittedCostModel {
public:
    double rel_bw = 0.10;
    double rel_rtt = 0.10;
    std::string label = "MEASURED_NATIVE";
    std::map<std::string, ComponentFit> fits;

    // Fit (a, c, b) per component by non-negative least squares on measured_s minus the transport
    // term; sigma_compute_s is the RMS of per-configuration repeat standard deviations.
    static Expected<FittedCostModel> fit(const std::vector<CalibrationRow>& rows, double rel_bw = 0.10, double rel_rtt = 0.10);

    bool has(const std::string& component) const { return fits.count(component) != 0; }
    Expected<Prediction> predict(const std::string& component, std::int64_t n, std::int64_t d_in, std::int64_t d_out,
                                 const Env& env) const;
    std::string status_of(const std::string& component) const;   // the fit's status, "MISSING" when absent

    Json to_json() const;
    static Expected<FittedCostModel> from_json(const Json& j);
    static Expected<FittedCostModel> load(const std::string& path);
    Status save(const std::string& path) const;
};

// Hand-written prior coefficients (status HISTORICAL: fixed constants, not fitted on the shipped
// calibration rows) for a smoke prediction when no fitted file exists. Never used for evidence records.
FittedCostModel historical_prior_model();

// The cost model a session should use: results/calibration/fitted_model.json when the file
// exists (searched as find_calibration_file does; status = the file's label, source = its path),
// else the HISTORICAL prior (status "HISTORICAL", source "historical_prior_model"). A file that
// exists but cannot be read or parsed fails closed (JSON_IO, JSON_PARSE, COSTMODEL_JSON,
// UNKNOWN_COMPONENT): a broken calibration is never replaced silently by the prior.
struct DefaultModel {
    FittedCostModel model;
    std::string status;
    std::string source;
};
Expected<DefaultModel> default_model(const std::string& calibration_dir = "");

// Error metrics between measured and predicted vectors: mape, median_rel_err, p95_rel_err, spearman, n.
Expected<Json> error_metrics(const std::vector<double>& measured, const std::vector<double>& predicted);

// Non-negative least squares (Lawson-Hanson) for small dense systems; exposed for tests.
Expected<std::vector<double>> nnls(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

}  // namespace highlevel
}  // namespace opencrab2
