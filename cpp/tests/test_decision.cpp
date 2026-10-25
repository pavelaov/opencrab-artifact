// OpenCrab v2: tests of the decision stack (json, contract, costmodel, candidates, controller, planner, statistics).
//
// Build (Windows, ziglang):
//   python -m ziglang c++ -std=c++17 -O2 -I include src/highlevel/json.cpp src/highlevel/contract.cpp
//       src/highlevel/costmodel.cpp src/highlevel/candidates.cpp src/highlevel/controller.cpp src/highlevel/planner.cpp
//       src/highlevel/statistics.cpp src/lowlevel/sha256.cpp tests/test_decision.cpp -o test_decision
// Run:  test_decision                          (all checks)
//       test_decision --vectors <path>         (also writes the cross-language controller vectors)
//       test_decision --check-vectors <path>   (also re-runs the vectors of an existing file)
//
// Covered regressions: the install example (keep 50 / A 105 / B 21 -> B), the stationary cold start
// (installs, then no further switch), the typed planner state (the domain-only DP misses the
// level-feasible prefix), the assumption set of a decision, exhaustive brute-force agreement of the stationary no-uncertainty no-hysteresis rule
// over 3 candidates x 5 epochs x 50 random tables, ExactAssign == ILPAdapter == brute_force on
// random chains with n <= 6, the contract hash against the Python hashlib value, the formula table
// with the fused k-ary tournament, the shared
// results/calibration/formulas.json (compared vector by vector when the Python side has
// written it, skipped with a note otherwise) and fitted_model.json, the KARY_MAGNITUDE feasibility
// filter, the planner x controller factorial, the per-step horizon (candidates with
// exec_s_by_epoch, horizon_schedule, q_hat, horizon_exec and the baselines on them), the paired
// bootstrap and the zero-failure bound.
//       test_decision --formulas-out <path>    (also writes the C++ formulas document for the parity pass)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "opencrab2/highlevel/candidates.hpp"
#include "opencrab2/highlevel/contract.hpp"
#include "opencrab2/highlevel/controller.hpp"
#include "opencrab2/highlevel/costmodel.hpp"
#include "opencrab2/highlevel/json.hpp"
#include "opencrab2/highlevel/planner.hpp"
#include "opencrab2/highlevel/statistics.hpp"

using namespace opencrab2;
using namespace opencrab2::highlevel;

static int g_checks = 0;

static void check(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) {
        std::printf("FAIL check %d: %s\n", g_checks, what.c_str());
        std::exit(1);
    }
}

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b))); }
static bool prefix(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
static void test_json() {
    Expected<Json> j = Json::parse("{\"b\":[1, 2.5, \"x\", true, null, {\"z\":1, \"a\":[]}], \"a\":\"\\u00e9\\ud83d\\ude00\"}");
    check(j.ok, "parse nested document: " + j.code);
    check(j.value.dump() == "{\"a\":\"\\u00e9\\ud83d\\ude00\",\"b\":[1,2.5,\"x\",true,null,{\"a\":[],\"z\":1}]}", "canonical dump: " + j.value.dump());
    Expected<Json> again = Json::parse(j.value.dump());
    check(again.ok && again.value == j.value, "round trip equality");
    check(j.value.get("b")[0].is_int() && j.value.get("b")[1].is_double(), "int and double kinds");
    check(j.value.get("a").as_string() == "\xC3\xA9\xF0\x9F\x98\x80", "unicode escapes decode to UTF-8");

    struct Sample { double v; const char* s; };
    const Sample samples[] = {{1e-05, "1e-05"}, {1e16, "1e+16"}, {1e15, "1000000000000000.0"}, {0.1, "0.1"}, {1.5e-7, "1.5e-07"},
                              {123456789012345680.0, "1.2345678901234568e+17"}, {2.5e-9, "2.5e-09"}, {7.246562834215189e-09, "7.246562834215189e-09"},
                              {100.0, "100.0"}, {1e22, "1e+22"}, {0.0001, "0.0001"}, {-0.0, "-0.0"}, {0.0, "0.0"}, {3.0, "3.0"},
                              {123456.789, "123456.789"}, {-2.5, "-2.5"}, {0.5, "0.5"}, {1234567890123456.0, "1234567890123456.0"}};
    for (const Sample& s : samples) check(format_double_repr(s.v) == s.s, std::string("float repr ") + s.s + " got " + format_double_repr(s.v));

    check(!Json::parse("{").ok && prefix(Json::parse("{").code, "JSON_PARSE"), "unterminated object fails");
    check(!Json::parse("[1,]").ok, "trailing comma fails");
    check(!Json::parse("[1] x").ok, "trailing characters fail");
    std::string deep(600, '[');
    deep += std::string(600, ']');
    check(prefix(Json::parse(deep).code, "JSON_DEPTH"), "depth limit");
    Expected<Json> big = Json::parse("[9223372036854775807, 9223372036854775808, -5, -9223372036854775808]");
    check(big.ok && big.value[0].is_int() && big.value[1].is_double() && big.value[2].as_int() == -5 && big.value[3].is_int(), "int64 boundary parsing");
    check(big.value[0].as_int() == 9223372036854775807LL, "max int64 value");
    Json o = Json::object();
    o.set("k", 3);
    o.set("s", "v");
    check(o.get_int("k").ok && o.get_int("k").value == 3, "get_int");
    check(prefix(o.get_int("s").code, "JSON_TYPE"), "get_int type code");
    check(prefix(o.get_int("missing").code, "JSON_MISSING"), "get_int missing code");
    check(o.get_string("s").value == "v", "get_string");
    o["new"] = Json(true);
    check(o.dump() == "{\"k\":3,\"new\":true,\"s\":\"v\"}", "operator[] keeps sorted order: " + o.dump());
    check(o.dump_pretty(2).find("\n  \"k\": 3") != std::string::npos, "pretty print");
    Json s = Json("line\n\"q\"\\ tab\t \x01");
    check(s.dump() == "\"line\\n\\\"q\\\"\\\\ tab\\t \\u0001\"", "string escaping: " + s.dump());
    check(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    check(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    Json u(18446744073709551615ULL);
    check(u.is_double(), "uint64 above int64 becomes double");
    std::printf("  json ok\n");
}

// ---------------------------------------------------------------------------
// Contract
// ---------------------------------------------------------------------------
static void test_contract() {
    NumericContract c = default_contract();
    check(c.validate().ok, "default contract valid");
    check(c.to_json().size() == 23, "23 contract fields");
    check(c.contract_hash() == "9c357c649b5264717a658c4edd57fcb149ef354c8759809088f8ee20896f4b88", "default hash equals Python: " + c.contract_hash());
    NumericContract s = smollm_contract();
    check(s.contract_hash() == "91e1988b7bd81214b77b25e39a5a50880663c93d3a03f042172118dfeeeb769f", "smollm hash equals Python: " + s.contract_hash());
    check(s.logical_bits == 46 && s.rsqrt_m == 16 && s.name == "NC-v2-f16-l46-smollm", "smollm fields");
    Expected<NumericContract> rt = NumericContract::from_json(c.to_json());
    check(rt.ok && rt.value.contract_hash() == c.contract_hash(), "from_json round trip");
    Json broken = c.to_json();
    broken.erase("gelu");
    check(prefix(NumericContract::from_json(broken).code, "CONTRACT_JSON"), "missing field refused");
    Json extra = c.to_json();
    extra.set("extra", 1);
    check(prefix(NumericContract::from_json(extra).code, "CONTRACT_JSON"), "extra field refused");
    NumericContract bad = c;
    bad.frac_bits = 0;
    check(prefix(bad.validate().code, "CONTRACT_PARAMS"), "frac_bits 0 refused");
    bad = c;
    bad.exp_r = 20;
    check(!bad.validate().ok, "exp_r > frac_bits refused");
    check(prefix(contract_by_name("nope").code, "CONTRACT_UNKNOWN"), "unknown contract name");
    check(contract_by_name("NC-v2-f16-l40").ok, "default by name");
    check(c.gelu_degree() == 12, "gelu degree 12");
    check(c.softmax_extra_bits(1024).value == 10 && c.softmax_extra_bits(50).value == 6 && c.softmax_extra_bits(1).value == 0, "softmax extra bits");
    check(c.scale() == 65536ULL && c.logical_bound() == (1ULL << 39), "scale and logical bound");
    std::printf("  contract ok\n");
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------
static FittedCostModel synthetic_model() {
    // every component gets a = 1e-9 per MAC, c = 2e-9 per byte, b = 1e-3 fixed, sigma 1e-3 (test values)
    Json comps = Json::object();
    for (const std::string& name : components()) {
        Json f = Json::object();
        f.set("compute_per_mac_s", 1e-9);
        f.set("compute_per_byte_s", 2e-9);
        f.set("compute_fixed_s", 1e-3);
        f.set("sigma_compute_s", 1e-3);
        f.set("n_train", 10);
        comps.set(name, f);
    }
    Json j = Json::object();
    j.set("rel_bw", 0.1);
    j.set("rel_rtt", 0.1);
    j.set("components", comps);
    Expected<FittedCostModel> m = FittedCostModel::from_json(j);
    check(m.ok, "synthetic model from_json: " + m.code);
    return m.value;
}

// A scratch directory unique to this process (concurrent test runs never share a file).
static std::string temp_dir() {
    static std::string dir;
    if (dir.empty()) {
        const char* base = std::getenv("TEMP");
        if (!base) base = std::getenv("TMPDIR");
        if (!base) base = "/tmp";
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        dir = std::string(base) + "/oc2_decision_" + std::to_string(static_cast<long long>(stamp));
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
    }
    return dir;
}

static std::string temp_file(const std::string& name) { return temp_dir() + "/" + name; }

static void test_costmodel() {
    reset_formula_constants();   // the checks below pin the built-in constants (frame 19, measured ciphertext 432404)
    const std::int64_t F = 19;
    check(formula_constants().frame_bytes == F && formula_constants().source == "builtin" && kBfvCiphertextBytes == 432404 && kBfvSlots == 8192,
          "built-in shared constants equal results/calibration/formulas.json of this tree");
    check(components().size() == 16, "16 components");
    check(dcf_key_bytes(32, 1) == 824 && dcf_key_bytes(64, 1) == 1624 && dcf_key_bytes(16, 1) == 424 && dcf_key_bytes(32, 2) == 1088, "DCF key sizes");
    const NumericContract c = default_contract();
    check(drelu_material_bytes(c) == 1672 && drelu_material_bytes(c, 2) == 2200 && trunc_material_bytes(c) == 2064 && r2f_material_bytes() == 3272,
          "gadget material sizes (DReLU 1672 without and 2200 with the product payload)");
    // payload (opened elements, both directions) and the framed total: payload + frame_bytes * messages
    check(payload_bytes("gelu_clip12", 4, 100, 100).value == 480 * 400, "gelu payload 480 N");
    check(rounds_formula("gelu_clip12", 4, 100, 100).value == 14 && messages_formula("gelu_clip12", 4, 100, 100).value == 28, "gelu rounds 14, two messages each");
    check(bytes_formula("gelu_clip12", 4, 100, 100).value == 480 * 400 + 28 * F, "gelu bytes = payload + framing");
    // layernorm: 3 Beaver products and one faithful truncation per element, rsqrt per row with its 14 LIVE
    // thresholds (4^8 is at or above the DCF bound 2^15 and costs nothing) and 13 Beaver products
    check(payload_bytes("layernorm_rows", 2, 768, 768).value == 112 * 1536 + 640 * 2, "layernorm payload 112 N + 640 R");
    check(rounds_formula("layernorm_rows", 2, 768, 768).value == 18, "layernorm rounds 18");
    Expected<GadgetCounts> ln = gadget_counts("layernorm_rows", 2, 768, 768, c);
    check(ln.ok && ln.value.drelu == 14 * 2 && ln.value.beaver == 3 * 1536 + 13 * 2 && ln.value.trunc_faithful == 1536 && ln.value.messages == 36, "layernorm gates");
    NumericContract no_faith = c;
    no_faith.trunc_square_ln = "local_prob";
    Expected<GadgetCounts> ln2 = gadget_counts("layernorm_rows", 2, 768, 768, no_faith);
    check(ln2.ok && ln2.value.trunc_faithful == 0 && ln2.value.rounds == 17, "without the faithful square: no gate, one round less");
    NumericContract wide_dcf = c;
    wide_dcf.dcf_domain_bits = 40;   // DCF bound 2^23: every rsqrt threshold is live
    check(gadget_counts("layernorm_rows", 1, 8, 8, wide_dcf).value.drelu == 15, "live thresholds follow the DCF bound of the contract");
    check(payload_bytes("linear_wmask", 3, 768, 2304).value == 16 * 3 * 768, "linear_wmask payload");
    check(bytes_formula("linear_wmask", 3, 768, 2304).value == 16 * 3 * 768 + 2 * F && messages_formula("linear_wmask", 3, 768, 2304).value == 2,
          "linear_wmask bytes: one round, two frames");
    check(payload_bytes("linear_generic", 3, 768, 2304).value == 16 * (3 * 768 + 768 * 2304), "linear_generic payload");
    check(payload_bytes("matmul_shared", 3, 768, 2304).value == payload_bytes("linear_generic", 3, 768, 2304).value, "matmul_shared = linear_generic");
    // the generic embedding counts both directions like every other opening
    check(payload_bytes("embedding_generic", 1, 50257, 768).value == 16LL * (50257 + 50257LL * 768), "embedding_generic payload 16 (V + V d)");
    check(bytes_formula("embedding_generic", 1, 50257, 768).value == 16LL * (50257 + 50257LL * 768) + 2 * F, "embedding_generic bytes with framing");
    check(payload_bytes("embedding_wmask", 4, 1000, 64).value == 16 * 4 * 1000, "embedding_wmask opens the one-hot rows only");
    check(material_formula("linear_wmask", 3, 768, 2304, c).value == 8 * (3 * 768 + 3 * 2304), "linear_wmask material");
    check(material_formula("linear_generic", 3, 768, 2304, c).value == 8 * (3 * 768 + 768 * 2304 + 3 * 2304), "linear_generic material");
    check(payload_bytes("install_wmask", 1, 768, 2304).value == 16 * 768 * 2304 && rounds_formula("install_wmask", 1, 768, 2304).value == 1, "install_wmask");
    check(bytes_formula("install_wmask", 1, 768, 2304).value == 16 * 768 * 2304 + 2 * F, "install: B1 up and E down, two frames");
    Expected<GadgetCounts> ig = gadget_counts("install_wmask", 1, 768, 2304, c);
    check(ig.ok && ig.value.oneway_elems == 2 * 768 * 2304 && ig.value.open_elems == 0 && ig.value.wmask_mask == 768 * 2304, "install gates: one-way elements");
    Expected<std::map<std::string, std::int64_t>> inst = material_by_type("install_wmask", 1, 768, 2304, c);
    check(inst.ok && inst.value.size() == 1 && inst.value["wmask_mask"] == 8 * 768 * 2304, "install consumes the mask share (wmask_mask)");
    Expected<std::map<std::string, std::int64_t>> lwm = material_by_type("linear_wmask", 3, 768, 2304, c);
    check(lwm.ok && lwm.value.size() == 1 && lwm.value["wmask_mask"] == 8 * (3 * 768 + 3 * 2304),
          "cost-model vocabulary: the per-call weight-mask shares are reported as wmask_mask");
    check(macs("linear_wmask", 3, 768, 2304).value == 3 * 768 * 2304 && macs("gelu_clip12", 4, 100, 100).value == 400, "macs");
    check(macs("softmax_rows_k2", 12, 32, 32).value == 12 * 32 && macs("argmax_k4", 1, 50257, 1).value == 50257 &&
              macs("install_wmask", 7, 768, 2304).value == 768 * 2304 && macs("layernorm_rows", 2, 768, 768).value == 1536,
          "macs of row components are element counts; install ignores n");
    check(payload_bytes("trunc_faithful", 10, 10, 1).value == 1600 && material_formula("trunc_faithful", 10, 10, 1, c).value == 206400, "trunc formulas");
    check(payload_bytes("ring_to_field", 10, 10, 1).value == 1600 && material_formula("ring_to_field", 10, 10, 1, c).value == 327200, "r2f formulas");
    check(rounds_formula("a2h_exact", 1, 100, 1).value == 2 && payload_bytes("a2h_exact", 1, 100, 1).value == 1600 + kBfvCiphertextBytes, "a2h formulas");
    check(payload_bytes("a2h_exact", 1, 8193, 1).value == 16 * 8193 + 2 * kBfvCiphertextBytes, "a2h two ciphertexts");
    check(messages_formula("a2h_exact", 1, 100, 1).value == 3 && bytes_formula("a2h_exact", 1, 100, 1).value == 1600 + 432404 + 3 * F,
          "a2h: one exchange plus ONE ciphertext message (measured size 432404)");

    // fused k-ary tournament (paper, Section 5.3): two rounds per level, per node of g
    // members g (g - 1) / 2 comparisons, then one Beaver select (g = 2) or g - 1 payload DReLUs (g >= 3);
    // a secret index (every level after the first) doubles the round-2 work
    TournamentCounts t2 = tournament_counts(50257, 2, true);
    check(t2.levels == 16 && t2.rounds == 32 && t2.comparisons == 50256 && t2.payload_drelu == 0, "binary tournament counts");
    check(t2.selects == 50256 + (50256 - 25128), "binary argmax: a value select per node, an index select after the first level");
    TournamentCounts t2m = tournament_counts(50257, 2, false);
    check(t2m.selects == 50256 && t2m.comparisons == 50256, "binary max: n - 1 comparisons and selects");
    TournamentCounts t4 = tournament_counts(50257, 4, true);
    check(t4.levels == 8 && t4.rounds == 16, "4-ary tournament: 8 levels, 16 rounds");
    TournamentCounts t8 = tournament_counts(50257, 8, true);
    // levels of 50257 under k = 8: 6283, 786, 99, 13, 2, 1
    check(t8.levels == 6 && t8.rounds == 12 && t8.comparisons > t2.comparisons && t8.payload_drelu > 0, "8-ary tournament: fewer rounds, more comparisons");
    check(rounds_formula("argmax_k2", 1, 50257, 1).value == 32 && rounds_formula("argmax_k4", 1, 50257, 1).value == 16 &&
              rounds_formula("argmax_k8", 1, 50257, 1).value == 12,
          "argmax rounds = 2 ceil(log_k n)");
    check(payload_bytes("argmax_k2", 1, 50257, 1).value == 16 * 50256 + 32 * 75384, "argmax_k2 payload");
    check(payload_bytes("argmax_k2", 3, 50257, 1).value == 3 * (16 * 50256 + 32 * 75384), "argmax payload scales with the rows");
    TournamentCounts t5 = tournament_counts(5, 4, false);
    check(t5.levels == 2 && t5.comparisons == 6 + 1 && t5.payload_drelu == 3 && t5.selects == 1 && t5.rounds == 4, "node of 4, pass-through, then a pair");
    TournamentCounts t5i = tournament_counts(5, 4, true);
    check(t5i.comparisons == 7 && t5i.payload_drelu == 3 && t5i.selects == 2, "the index of the second level costs one more select");
    TournamentCounts t17 = tournament_counts(17, 8, true);   // level 1: 8, 8, 1; level 2: one node of 3 with a secret index
    check(t17.comparisons == 28 + 28 + 3 && t17.payload_drelu == 7 + 7 + 2 * 2 && t17.selects == 0 && t17.levels == 2, "17 candidates under k = 8");
    check(tournament_counts(1, 4, true).rounds == 0 && tournament_counts(64, 3, false).levels == 0, "degenerate tournaments count nothing");
    Expected<GadgetCounts> a8 = gadget_counts("argmax_k8", 1, 8, 1, c);
    check(a8.ok && a8.value.drelu == 28 && a8.value.drelu_p2 == 7 && a8.value.kary_select == 0 && a8.value.rounds == 2, "argmax_k8 over one node of 8");
    check(material_formula("argmax_k8", 1, 8, 1, c).value == 28 * 1672 + 7 * 2200, "payload DReLUs are priced with their larger keys");
    check(payload_bytes("argmax_k8", 1, 8, 1).value == 16 * 35, "one opened element per DReLU");
    check(rounds_formula("softmax_rows_k2", 12, 32, 32).value == 2 * 5 + 10 + 12 + 1, "softmax k2 rounds for 32 keys");
    check(messages_formula("argmax_k8", 1, 50257, 1).value == 24 && bytes_formula("argmax_k2", 1, 2, 1).value == 16 + 32 + 4 * F,
          "argmax messages: two per round; a single pair is one comparison and one select");
    check(rounds_formula("argmax_k4", 3, 1, 1).value == 0 && payload_bytes("argmax_k4", 3, 1, 1).value == 0, "argmax over one item costs nothing");
    // 32 keys under k = 4: 8 nodes of 4, 2 nodes of 4, one pair: 3 levels
    check(rounds_formula("softmax_rows_k4", 12, 32, 32).value == 2 * 3 + 10 + 12 + 1, "softmax k4 rounds for 32 keys");
    // softmax over 2 rows of 5 keys, k = 2: max 4 comparisons + 4 selects per row, exp 16 + 9 * 32 per
    // element, recip 11 * 16 + 11 * 32 per row, one product per element
    check(payload_bytes("softmax_rows_k2", 2, 5, 5).value == 2 * 4 * 48 + 10 * 304 + 2 * 528 + 10 * 32, "softmax k2 payload");
    Expected<std::map<std::string, std::int64_t>> sm = material_by_type("softmax_rows_k4", 1, 5, 5, c);
    check(sm.ok && sm.value["drelu"] == 1672 * (7 + 5 + 11) + 2200 * 3 && sm.value["kary_select"] == 24 && sm.value["beaver"] == 24 * (45 + 11 + 5),
          "softmax k4 material by dealer type");
    Expected<std::map<std::string, std::int64_t>> mt = material_by_type("layernorm_rows", 2, 768, 768, c);
    check(mt.ok && mt.value.count("beaver") && mt.value.count("drelu") && mt.value.count("trunc_faithful"), "layernorm material types");
    check(mt.value["trunc_faithful"] == 2064 * 1536 && mt.value["drelu"] == 1672 * 14 * 2 && mt.value["beaver"] == 24 * (3 * 1536 + 13 * 2), "layernorm material values");
    check(prefix(bytes_formula("nope", 1, 1, 1).code, "UNKNOWN_COMPONENT"), "unknown component code");
    check(prefix(gadget_counts("gelu_clip12", -1, 1, 1, c).code, "COSTMODEL_SHAPE"), "negative shape refused");
    check(prefix(gadget_counts("gelu_clip12", 0, 1, 1, c).code, "COSTMODEL_SHAPE") && prefix(payload_bytes("linear_wmask", 1, 8, 0).code, "COSTMODEL_SHAPE") &&
              prefix(macs("linear_generic", 1, 0, 4).code, "COSTMODEL_SHAPE"),
          "a used dimension below 1 is refused");
    check(payload_bytes("gelu_clip12", 1, 8, 0).ok, "d_out is ignored by elementwise components");

    // magnitude bound of the fused nodes
    check(kary_ceil_log2(1) == 0 && kary_ceil_log2(2) == 1 && kary_ceil_log2(3) == 2 && kary_ceil_log2(4) == 2 && kary_ceil_log2(5) == 3 && kary_ceil_log2(8) == 3,
          "ceil(log2 g)");
    check(kary_difference_bound_ring(32, 2) == (1LL << 31) && kary_difference_bound_ring(32, 3) == (1LL << 28) &&
              kary_difference_bound_ring(32, 4) == (1LL << 28) && kary_difference_bound_ring(32, 5) == (1LL << 27) &&
              kary_difference_bound_ring(32, 8) == (1LL << 27),
          "difference bound 2^(n-1) for binary nodes, 2^(n-2-ceil(log2 g)) for fused nodes");
    check(kary_difference_bound_ring(32, 8) / 65536 == 2048 && kary_difference_bound_ring(32, 4) / 65536 == 4096, "2^11 and 2^12 real units at n = 32, f = 16");
    check(kary_difference_bound_ring(3, 8) == 0 && kary_difference_bound_ring(64, 2) > 0, "degenerate domains");
    check(kary_operand_bound_real(c, 8, 50257) == 1024.0 && kary_operand_bound_real(c, 4, 50257) == 2048.0 && kary_operand_bound_real(c, 2, 50257) == 16384.0,
          "admitted operand magnitude: half the difference bound");
    check(kary_operand_bound_real(c, 8, 3) == 2048.0 && kary_operand_bound_real(c, 8, 2) == 16384.0, "short rows only form small nodes");
    check(kary_magnitude_check(c, 8, 50257, 512.0, true).ok && kary_magnitude_check(c, 4, 50257, 1024.0, true).ok && kary_magnitude_check(c, 2, 50257, 2048.0, true).ok,
          "bounds inside the node bound are admitted");
    Status k8 = kary_magnitude_check(c, 8, 50257, 1024.0, true);
    check(!k8.ok && k8.code_prefix() == "KARY_MAGNITUDE", "k = 8 refuses a declared bound of 2^10 (differences reach 2^11): " + k8.code);
    check(prefix(kary_magnitude_check(c, 4, 50257, 2048.0, false).code, "KARY_MAGNITUDE") && prefix(kary_magnitude_check(c, 2, 50257, 16384.0, false).code, "KARY_MAGNITUDE"),
          "k = 4 and k = 2 refuse at their own bounds");
    check(kary_magnitude_check(c, 8, 3, 1500.0, true).ok && kary_magnitude_check(c, 8, 1, 1e9, true).ok, "the bound follows the largest node that can occur");
    check(prefix(kary_magnitude_check(c, 8, 64, 0.0, false).code, "KARY_MAGNITUDE") && prefix(kary_magnitude_check(c, 8, 64, std::nan(""), false).code, "KARY_MAGNITUDE"),
          "a missing declared bound fails closed");
    check(prefix(kary_magnitude_check(c, 3, 64, 1.0, false).code, "KARY_PARAMS") && prefix(kary_magnitude_check(c, 4, 0, 1.0, false).code, "KARY_PARAMS"), "arity and length are validated");
    NumericContract narrow = c;
    narrow.dcf_domain_bits = 18;   // node bound 2^13: index differences up to L - 1 = 50256 do not fit
    check(prefix(kary_magnitude_check(narrow, 8, 50257, 0.001, true).code, "KARY_MAGNITUDE") && kary_magnitude_check(narrow, 8, 50257, 0.001, false).ok,
          "argmax also bounds the row length (index differences)");

    FittedCostModel m = synthetic_model();
    Env env;
    env.bandwidth_bps = 100e6;
    env.one_way_latency_s = 0.005;
    env.cpu_share = 0.5;
    Expected<Prediction> p = m.predict("gelu_clip12", 4, 100, 100, env);
    check(p.ok, "predict ok");
    const double framed = 480.0 * 400 + 28 * 19;
    const double expected_compute = (1e-9 * 400 + 2e-9 * framed + 1e-3) / 0.5;
    const double expected_comm = framed * 8 / 100e6;
    const double expected_round = 14 * 0.01;
    check(p.value.bytes == static_cast<std::int64_t>(framed) && p.value.rounds == 14, "prediction uses the framed byte figure");
    check(near(p.value.compute_s, expected_compute) && near(p.value.comm_s, expected_comm) && near(p.value.round_s, expected_round), "prediction terms");
    check(near(p.value.mean_s, expected_compute + expected_comm + expected_round), "prediction mean");
    const double sc = 1e-3 / 0.5;
    check(near(p.value.sigma_s, std::sqrt(sc * sc + std::pow(expected_comm * 0.1, 2) + std::pow(expected_round * 0.1, 2))), "prediction sigma");
    check(prefix(m.predict("nope", 1, 1, 1, env).code, "UNKNOWN_COMPONENT"), "predict unknown component");
    FittedCostModel partial;
    check(prefix(partial.predict("gelu_clip12", 1, 1, 1, env).code, "COSTMODEL_NOT_FITTED"), "unfitted component fails closed");
    Env bad_env = env;
    bad_env.cpu_share = 0.0;
    check(prefix(m.predict("gelu_clip12", 1, 1, 1, bad_env).code, "ENV_PARAMS"), "cpu_share 0 refused");
    Expected<FittedCostModel> rt = FittedCostModel::from_json(m.to_json());
    check(rt.ok && rt.value.to_json() == m.to_json(), "cost model json round trip");
    Json badc = m.to_json();
    badc["components"]["bogus"] = Json::object();
    check(prefix(FittedCostModel::from_json(badc).code, "UNKNOWN_COMPONENT"), "unknown component in file refused");
    Json negc = m.to_json();
    negc["components"]["gelu_clip12"]["compute_per_mac_s"] = Json(-1.0);
    check(prefix(FittedCostModel::from_json(negc).code, "COSTMODEL_JSON"), "negative coefficient refused");

    // fit recovers exact coefficients from noise-free rows (a = 3e-9, c = 5e-9, b = 0.02)
    std::vector<CalibrationRow> rows;
    const double a = 3e-9, cc = 5e-9, b = 0.02;
    for (std::int64_t n : {1, 4, 16, 32}) {
        for (std::int64_t d : {64, 256, 768}) {
            for (double bw : {50e6, 200e6}) {
                CalibrationRow r;
                r.component = "linear_wmask";
                r.n = n;
                r.d_in = d;
                r.d_out = 3 * d;
                r.bandwidth_bps = bw;
                r.one_way_latency_s = 0.004;
                r.cpu_share = 1.0;
                r.bytes = bytes_formula("linear_wmask", n, d, 3 * d).value;
                r.rounds = 1;
                r.measured_s = a * static_cast<double>(n * d * 3 * d) + cc * static_cast<double>(r.bytes) + b + static_cast<double>(r.bytes) * 8 / bw + 0.008;
                rows.push_back(r);
            }
        }
    }
    Expected<FittedCostModel> fit = FittedCostModel::fit(rows);
    check(fit.ok, "fit ok: " + fit.code);
    const ComponentFit& f = fit.value.fits["linear_wmask"];
    check(near(f.compute_per_mac_s, a, 1e-6) && near(f.compute_per_byte_s, cc, 1e-6) && near(f.compute_fixed_s, b, 1e-6), "fit recovers coefficients");
    check(f.n_train == static_cast<std::int64_t>(rows.size()) && f.sigma_compute_s == 0.0, "fit metadata, zero repeat spread");
    // repeats with spread give sigma > 0 and non-negative coefficients
    std::vector<CalibrationRow> noisy = rows;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        CalibrationRow r = rows[i];
        r.measured_s += (i % 2 == 0) ? 0.003 : -0.003;
        noisy.push_back(r);
    }
    Expected<FittedCostModel> fit2 = FittedCostModel::fit(noisy);
    check(fit2.ok && fit2.value.fits["linear_wmask"].sigma_compute_s > 0.0, "repeat spread gives sigma > 0");
    check(fit2.value.fits["linear_wmask"].compute_per_mac_s >= 0.0 && fit2.value.fits["linear_wmask"].compute_fixed_s >= 0.0, "NNLS non-negative");
    Expected<std::vector<double>> nn = nnls({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}, {1.0, -1.0, 0.0});
    check(nn.ok && near(nn.value[0], 0.5, 1e-9) && nn.value[1] == 0.0, "nnls clamps the negative coordinate");
    FittedCostModel prior = historical_prior_model();
    for (const std::string& name : components()) check(prior.has(name), "prior model has " + name);
    check(prior.label == "HISTORICAL" && prior.status_of("gelu_clip12") == "HISTORICAL" && prior.status_of("nope") == "MISSING", "the prior carries its evidence label");
    Expected<Json> em = error_metrics({1.0, 2.0, 3.0}, {1.0, 2.0, 3.0});
    check(em.ok && em.value.get("mape").as_double() == 0.0 && near(em.value.get("spearman").as_double(), 1.0), "error metrics identity");
    std::printf("  costmodel ok\n");
}

// ---------------------------------------------------------------------------
// Shared formulas (results/calibration/formulas.json) and the fitted model file
// ---------------------------------------------------------------------------
static void test_formulas_and_fitted_model() {
    reset_formula_constants();
    // --- constants table
    FormulaConstants fc = formula_constants();
    fc.frame_bytes = 100;
    fc.bfv_ciphertext_bytes = 655360;
    fc.source = "test";
    check(set_formula_constants(fc).ok, "set constants");
    check(bytes_formula("linear_wmask", 3, 768, 2304).value == 16 * 3 * 768 + 200, "frame bytes come from the shared table");
    check(payload_bytes("a2h_exact", 1, 100, 1).value == 1600 + 655360, "ciphertext bytes come from the shared table");
    FormulaConstants bad = fc;
    bad.bfv_slots = 0;
    check(prefix(set_formula_constants(bad).code, "FORMULAS_PARAMS") && formula_constants().bfv_slots == 8192, "invalid constants refused, table unchanged");
    reset_formula_constants();
    check(formula_constants().frame_bytes == 19 && formula_constants().bfv_ciphertext_bytes == kBfvCiphertextBytes, "reset restores the built-in values");

    // --- the C++ document checks against itself, with constants adopted from the file
    const std::string doc_path = temp_file("formulas_self.json");
    Json doc = formulas_document();
    check(doc.get("shapes").size() >= 30 && doc.get("tournament").size() == 60 && doc.get("schema").as_string() == "opencrab2-formulas-v1" &&
              doc.get("framing").get("frame_bytes").as_int() == 19 && doc.get("bfv").get("ct_bytes").as_int() == 432404,
          "formulas document in the layout of the shared file");
    check(Json::save_file(doc_path, doc, true).ok, "write the self document");
    Expected<FormulasReport> self = check_formulas(doc_path);
    check(self.ok && self.value.found, "self document found: " + self.code);
    for (const std::string& mm : self.value.mismatches) std::printf("    SELF MISMATCH %s\n", mm.c_str());
    check(self.value.mismatches.empty() && self.value.vectors_compared == static_cast<std::int64_t>(doc.get("shapes").size()) &&
              self.value.tournament_compared == 60 && self.value.fields_compared > 900,
          "every vector of the self document matches (" + std::to_string(self.value.fields_compared) + " fields)");
    check(self.value.adopted.size() == 3, "frame bytes, slot count and ciphertext size adopted from the file");

    // --- a file written with other shared constants: the constants are adopted, its vectors must then agree
    Json other = Json::object();
    Json oc = Json::object();
    oc.set("FRAME_BYTES", 96);
    oc.set("OPEN_BYTES", 16);
    oc.set("BEAVER_OPEN_BYTES", 32);
    oc.set("drelu_p2_bytes", 2200);
    oc.set("drelu_p1_bytes", 1700);   // wrong on purpose: the C++ size is 1672
    other.set("constants", oc);
    Json a2h = Json::object();
    a2h.set("ct_bytes", 700001);
    a2h.set("status", "MEASURED_NATIVE");
    a2h.set("N", 8192);
    other.set("bfv", a2h);
    other.set("schema", "opencrab2-formulas-v1");
    Json ov = Json::array();
    {
        Json e = Json::object();
        e.set("component", "linear_wmask");
        e.set("n", 3);
        e.set("d_in", 768);
        e.set("d_out", 2304);
        e.set("bytes", 16 * 3 * 768 + 2 * 96);
        e.set("rounds", 1);
        e.set("messages", 2);
        Json eg = Json::object();
        eg.set("open_elems", 3 * 768);
        eg.set("wmask_call", 3 * 768 + 3 * 2304);
        eg.set("beaver", 0);
        eg.set("hypothetical_gate", 0);   // an unknown gate kind with a zero count is harmless
        e.set("gates", eg);
        ov.push_back(e);
        Json e2 = Json::object();
        e2.set("component", "a2h_exact");
        e2.set("n", 1);
        e2.set("d_in", 100);
        e2.set("payload_bytes", 1600 + 700001);
        e2.set("bytes", 1600 + 700001 + 3 * 96);
        Json mat = Json::object();
        mat.set("ring_to_field", 327200);
        e2.set("material", mat);
        ov.push_back(e2);
        Json e3 = Json::object();   // a deliberately wrong vector: the three-round count of the retired realization
        e3.set("component", "argmax_k4");
        e3.set("n", 1);
        e3.set("d_in", 50257);
        e3.set("d_out", 1);
        e3.set("rounds", 25);
        ov.push_back(e3);
        Json e4 = Json::object();
        e4.set("component", "not_a_component");
        e4.set("n", 1);
        e4.set("d_in", 1);
        e4.set("rounds", 1);
        ov.push_back(e4);
    }
    other.set("shapes", ov);
    Json ot = Json::array();
    {
        Json t = Json::object();
        t.set("n", 5);
        t.set("k", 4);
        t.set("with_index", true);
        t.set("drelu_p1", 7);
        t.set("drelu_p2", 3);
        t.set("beaver_select", 2);
        t.set("payload_bytes", 16 * 10 + 32 * 2);
        ot.push_back(t);
        Json tb = Json::object();
        tb.set("L", 5);
        tb.set("k", 4);
        tb.set("comparisons", 10);   // wrong: 7
        ot.push_back(tb);
    }
    Json tsec = Json::object();
    tsec.set("cases", ot);
    other.set("tournament", tsec);
    const std::string other_path = temp_file("formulas_other.json");
    check(Json::save_file(other_path, other, true).ok, "write the foreign document");
    Expected<FormulasReport> rep = check_formulas(other_path);
    check(rep.ok && rep.value.found, "foreign document read: " + rep.code);
    check(formula_constants().frame_bytes == 96 && formula_constants().bfv_ciphertext_bytes == 700001 && formula_constants().a2h_status == "MEASURED_NATIVE" &&
              formula_constants().source == other_path,
          "shared constants adopted from the file (measured ciphertext size)");
    check(rep.value.vectors_compared == 3 && rep.value.tournament_compared == 2, "recognized vectors counted");
    bool has_rounds = false, has_unknown = false, has_cmp = false, has_const = false;
    for (const std::string& mm : rep.value.mismatches) {
        if (mm.find("argmax_k4 (1, 50257, 1) rounds: file 25, cpp 16") != std::string::npos) has_rounds = true;
        if (mm.find("not_a_component") != std::string::npos) has_unknown = true;
        if (mm.find("drelu_p1: file 10, cpp 7") != std::string::npos) has_cmp = true;
        if (mm.find("constant drelu_p1_bytes: file 1700, cpp 1672") != std::string::npos) has_const = true;
    }
    for (const std::string& mm : rep.value.mismatches) std::printf("    expected mismatch: %s\n", mm.c_str());
    check(rep.value.mismatches.size() == 4 && has_rounds && has_unknown && has_cmp && has_const, "every divergence is named, nothing else is flagged");
    check(rep.value.to_json().get("mismatches").size() == 4, "report json");
    reset_formula_constants();

    // --- failure codes: the table is left unchanged
    const std::string broken_path = temp_file("formulas_broken.json");
    {
        std::FILE* fb = std::fopen(broken_path.c_str(), "wb");
        check(fb != nullptr, "open broken file");
        std::fputs("{\"constants\": {\"frame_bytes\": 12", fb);
        std::fclose(fb);
    }
    check(prefix(load_formulas(broken_path).code, "FORMULAS_JSON"), "truncated file fails closed");
    Json wrong = Json::object();
    wrong.set("open_bytes", 8);
    wrong.set("frame_bytes", 10);
    check(Json::save_file(broken_path, wrong, true).ok, "write a contradicting file");
    check(prefix(load_formulas(broken_path).code, "FORMULAS_MISMATCH") && formula_constants().frame_bytes == 19, "a contradicted protocol fact is refused and nothing is adopted");
    Json other_schema = Json::object();
    other_schema.set("schema", "opencrab2-formulas-v9");
    other_schema.set("frame_bytes", 77);
    check(Json::save_file(broken_path, other_schema, true).ok, "write another schema version");
    check(prefix(load_formulas(broken_path).code, "FORMULAS_JSON") && formula_constants().frame_bytes == 19, "an unknown schema version is not interpreted");
    Json neg = Json::object();
    neg.set("bfv_ct_bytes", -5);
    check(Json::save_file(broken_path, neg, true).ok, "write a negative constant");
    check(prefix(load_formulas(broken_path).code, "FORMULAS_PARAMS"), "non-positive constant refused");
    Json arr = Json::array();
    check(Json::save_file(broken_path, arr, true).ok, "write a non-object");
    check(prefix(load_formulas(broken_path).code, "FORMULAS_JSON"), "top level must be an object");
    Expected<FormulasReport> missing = check_formulas(temp_file("formulas_missing_file.json"));
    check(missing.ok && !missing.value.found && missing.value.fields_compared == 0, "a missing file is not an error: nothing compared");
    std::remove(doc_path.c_str());
    std::remove(other_path.c_str());
    std::remove(broken_path.c_str());

    // --- the shared file of this tree, when the Python side has written it
    Expected<FormulasReport> tree = check_formulas("");
    check(tree.ok, "results/calibration/formulas.json is readable when present: " + tree.code);
    if (!tree.value.found) {
        std::printf("  formulas.json not found from this working directory: the built-in constants are in force, cross-language comparison skipped\n");
    } else {
        const FormulaConstants loaded = formula_constants();
        check(loaded.source == tree.value.path && loaded.a2h_status == "MEASURED_NATIVE" && loaded.bfv_ciphertext_bytes == kBfvCiphertextBytes &&
                  loaded.frame_bytes == kFrameBytes && loaded.bfv_slots == kBfvSlots,
              "the built-in constants equal the shared file of this tree (frame bytes, measured ciphertext size, slots)");
        check(tree.value.vectors_compared >= 100 && tree.value.tournament_compared >= 60, "the shared file carries the declared shape list");
        std::printf("  formulas.json %s: adopted %d constants, compared %lld component and %lld tournament vectors (%lld fields)\n", tree.value.path.c_str(),
                    static_cast<int>(tree.value.adopted.size()), static_cast<long long>(tree.value.vectors_compared),
                    static_cast<long long>(tree.value.tournament_compared), static_cast<long long>(tree.value.fields_compared));
        for (const std::string& mm : tree.value.mismatches) std::printf("    FORMULA MISMATCH %s\n", mm.c_str());
        check(tree.value.mismatches.empty(), "the C++ formulas agree with every vector of results/calibration/formulas.json (" +
                                                 std::to_string(tree.value.mismatches.size()) + " mismatches listed above)");
    }
    reset_formula_constants();

    // --- fitted_model.json: the calibrated model of this tree when present, the HISTORICAL prior otherwise
    Expected<DefaultModel> dm = default_model("");
    check(dm.ok, "default model: " + dm.code);
    const std::string fitted = find_calibration_file("fitted_model.json");
    if (fitted.empty()) {
        check(dm.value.status == "HISTORICAL" && dm.value.source == "historical_prior_model", "no calibration file: the prior is returned and labelled");
        std::printf("  fitted_model.json not present: default_model() returned the HISTORICAL prior\n");
    } else {
        check(dm.value.source == fitted && dm.value.status == dm.value.model.label && !dm.value.status.empty(), "calibration file loaded and labelled");
        Env e2;
        e2.bandwidth_bps = 100e6;
        e2.one_way_latency_s = 0.01;
        int priced = 0;
        for (const std::string& name : components()) {
            if (!dm.value.model.has(name)) continue;
            Expected<Prediction> pr = dm.value.model.predict(name, 2, 64, 64, e2);
            check(pr.ok && std::isfinite(pr.value.mean_s) && pr.value.mean_s > 0.0 && std::isfinite(pr.value.sigma_s) && pr.value.sigma_s >= 0.0,
                  "fitted model prices " + name);
            ++priced;
        }
        check(priced >= 12, "the calibration file covers the component list");
        std::printf("  fitted_model.json %s: label %s, %d components priced\n", fitted.c_str(), dm.value.status.c_str(), priced);
    }
    // an explicit directory takes precedence; a broken file there fails closed instead of falling back
    const std::string dir = temp_dir();
    const std::string local = dir + "/fitted_model.json";
    FittedCostModel syn = synthetic_model();
    syn.label = "MODELED";
    check(syn.save(local).ok, "write a model into the hint directory");
    Expected<DefaultModel> hinted = default_model(dir);
    check(hinted.ok && hinted.value.status == "MODELED" && hinted.value.source == local && hinted.value.model.fits.size() == 16, "hint directory model loaded");
    {
        std::FILE* fb = std::fopen(local.c_str(), "wb");
        check(fb != nullptr, "open the hint file");
        std::fputs("{\"components\": {\"gelu_clip12\": {\"compute_per_mac_s\": \"fast\"}}}", fb);
        std::fclose(fb);
    }
    Expected<DefaultModel> broken = default_model(dir);
    check(!broken.ok && prefix(broken.code, "COSTMODEL_JSON"), "a broken calibration file is an error, never a silent fallback: " + broken.code);
    std::remove(local.c_str());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::printf("  formulas and fitted model ok\n");
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------
static Shapes small_shapes() {
    Shapes s;
    s.n_layers = 2;
    s.d_model = 64;
    s.d_ff = 256;
    s.n_head = 4;
    s.vocab = 1000;
    s.n_rows = 1;
    s.kv_len = 16;
    return s;
}

static void test_candidates() {
    FittedCostModel cm = historical_prior_model();
    Env env;
    env.bandwidth_bps = 100e6;
    env.one_way_latency_s = 0.01;
    Shapes sh = small_shapes();
    SessionState st;
    st.incumbent = make_plan("generic", "generic", 2, 2);
    st.remaining_epochs = 8;
    st.kv_bucket = 16;
    st.model_version = "gpt2@rev1";
    st.contract_hash = default_contract().contract_hash();
    st.session_id = "s1";
    Expected<std::vector<Candidate>> cands = enumerate_candidates(st, Catalog::full(), cm, env, sh);
    check(cands.ok, "enumerate: " + cands.code);
    check(cands.value.size() == 24, "24 candidates in the full catalog");
    check(same_plan(cands.value[0].plan, st.incumbent), "keep is first");
    check(cands.value[0].install_objects.size() == 1 && cands.value[0].install_objects[0] == "lm_head", "generic incumbent needs only lm_head");
    const Candidate* ww = nullptr;
    for (const Candidate& c : cands.value)
        if (same_plan(c.plan, make_plan("wmask", "wmask", 8, 4))) ww = &c;
    check(ww != nullptr, "wmask/wmask/8/4 present");
    check(ww->install_objects.size() == 10 && ww->c_install_s > 0.0 && ww->c_install_sigma > 0.0, "wmask plan installs 10 objects");
    check(ww->install_cost_by_object.size() == 10 && ww->install_cost_by_object.count("wte") && ww->install_cost_by_object.count("h1.proj"), "per object install costs");
    check(ww->exec_s_per_epoch > 0.0 && ww->exec_sigma > 0.0 && ww->material_bytes_per_epoch > 0 && ww->mem_peak_bytes > 0, "positive predictions");
    check(ww->material_by_type.count("wmask_mask") && ww->material_by_type.count("drelu") && ww->material_by_type.count("beaver") &&
              ww->material_by_type.count("trunc_faithful") && ww->material_by_type.count("matrix_triple"),
          "material types of the wmask plan");
    check(cands.value[0].material_by_type.count("kary_select") && cands.value[0].material_by_type["kary_select"] > 0,
          "binary tournaments consume kary_select triples");
    check(cands.value[0].material_by_type.count("wmask_mask") && cands.value[0].material_by_type.count("matrix_triple"), "generic plan uses triples and the lm_head wmask call");
    check(ww->feasible && cands.value[0].feasible, "feasible without declared stock");
    check(ww->rounds_per_epoch > cands.value[0].rounds_per_epoch - 100 && ww->online_bytes_per_epoch < cands.value[0].online_bytes_per_epoch, "wmask moves fewer online bytes than generic");
    for (std::size_t i = 1; i < cands.value.size(); ++i)
        for (std::size_t j = i + 1; j < cands.value.size(); ++j) check(!same_plan(cands.value[i].plan, cands.value[j].plan), "candidate plans distinct");

    // installed objects with the matching version remove the install cost; a stale version does not
    SessionState st2 = st;
    for (const auto& o : wmask_objects_for_plan(make_plan("wmask", "wmask", 8, 4), sh)) {
        Json v = Json::object();
        v.set("model_version", "gpt2@rev1");
        v.set("contract_hash", st.contract_hash);
        st2.installed[o.first] = v;
    }
    Expected<Candidate> c2 = evaluate_plan(make_plan("wmask", "wmask", 8, 4), st2, cm, env, sh);
    check(c2.ok && c2.value.install_objects.empty() && c2.value.c_install_s == 0.0, "installed objects cost nothing");
    SessionState st3 = st2;
    st3.model_version = "gpt2@rev2";
    Expected<Candidate> c3 = evaluate_plan(make_plan("wmask", "wmask", 8, 4), st3, cm, env, sh);
    check(c3.ok && c3.value.install_objects.size() == 10, "stale model version is never reused");
    SessionState st4 = st2;
    st4.contract_hash = "other";
    Expected<Candidate> c4 = evaluate_plan(make_plan("wmask", "wmask", 8, 4), st4, cm, env, sh);
    check(c4.ok && c4.value.install_objects.size() == 10, "contract hash mismatch is never reused");

    // material stock: insufficient -> infeasible; with refill -> wait
    SessionState st5 = st;
    st5.stock["drelu"] = std::make_pair<std::int64_t, std::int64_t>(1, 10);
    Expected<Candidate> c5 = evaluate_plan(st.incumbent, st5, cm, env, sh);
    check(c5.ok && !c5.value.feasible && c5.value.reason == "INSUFFICIENT_MATERIAL", "material shortfall is infeasible without refill");
    Shapes sh_refill = sh;
    sh_refill.refill_bytes_per_s = 1e6;
    Expected<Candidate> c6 = evaluate_plan(st.incumbent, st5, cm, env, sh_refill);
    check(c6.ok && c6.value.feasible && c6.value.c_wait_s > 0.0, "refill converts the shortfall into a wait");
    Shapes sh_mem = sh;
    sh_mem.mem_limit_bytes = 1024;
    Expected<Candidate> c7 = evaluate_plan(st.incumbent, st, cm, env, sh_mem);
    check(c7.ok && !c7.value.feasible && c7.value.reason == "MEMORY_INFEASIBLE", "memory limit is a hard filter");
    // per-epoch call list (the Python plan_calls): attention products are H calls each, the final LayerNorm,
    // the LM head and the argmax see one row
    {
        Shapes pre = sh;
        pre.n_rows = 5;
        std::vector<PlanCall> calls = plan_calls(make_plan("wmask", "generic", 4, 2), pre);
        check(calls.size() == static_cast<std::size_t>(1 + 10 * pre.n_layers + 3), "call list length");
        check(calls[0].component == "embedding_wmask" && calls[0].n == 5 && calls[0].d_in == pre.vocab && calls[0].count == 1, "embedding call");
        check(calls[3].component == "matmul_shared" && calls[3].count == pre.n_head && calls[3].n == 5 && calls[3].d_in == pre.d_model / pre.n_head &&
                  calls[3].d_out == pre.kv_len,
              "Q K^T is one matmul per head");
        check(calls[4].component == "softmax_rows_k2" && calls[4].n == pre.n_head * 5 && calls[4].d_in == pre.kv_len, "softmax rows: heads x query rows of kv_len keys");
        const PlanCall& last = calls.back();
        check(last.component == "argmax_k4" && last.n == 1 && last.d_in == pre.vocab, "argmax over the vocabulary on one row");
        check(calls[calls.size() - 3].component == "layernorm_rows" && calls[calls.size() - 3].n == 1, "final LayerNorm on the last row only");
        Expected<Candidate> ce = evaluate_plan(make_plan("wmask", "generic", 4, 2), st, cm, env, pre);
        double mean = 0.0, var = 0.0;
        std::int64_t rounds = 0;
        for (const PlanCall& call : calls) {
            Expected<Prediction> pr = cm.predict(call.component, call.n, call.d_in, call.d_out, env);
            check(pr.ok, "call priced: " + call.component);
            mean += static_cast<double>(call.count) * pr.value.mean_s;
            var += static_cast<double>(call.count) * pr.value.sigma_s * pr.value.sigma_s;
            rounds += call.count * pr.value.rounds;
        }
        check(ce.ok && near(ce.value.exec_s_per_epoch, mean) && near(ce.value.exec_sigma, std::sqrt(var)) && ce.value.rounds_per_epoch == rounds,
              "the epoch cost is the count-weighted sum of the calls, sigmas independent");
    }
    // KARY_MAGNITUDE: the declared public operand bounds decide, per k, whether the fused nodes are admitted
    {
        check(kary_feasibility(make_plan("wmask", "wmask", 8, 4), sh).ok, "default declared bounds admit every arity");
        Shapes loud = sh;
        loud.logits_bound = 1500.0;   // differences up to 3000 real units: beyond the node of 8 (2^11), inside the node of 4 (2^12)
        Status k8 = kary_feasibility(make_plan("wmask", "wmask", 8, 2), loud);
        check(!k8.ok && k8.code_prefix() == "KARY_MAGNITUDE" && k8.code.find("argmax_k") != std::string::npos, "argmax_k = 8 refused: " + k8.code);
        check(kary_feasibility(make_plan("wmask", "wmask", 4, 2), loud).ok && kary_feasibility(make_plan("wmask", "wmask", 2, 2), loud).ok,
              "argmax_k 4 and 2 stay admitted under the same bound");
        Expected<std::vector<Candidate>> lc = enumerate_candidates(st, Catalog::full(), cm, env, loud);
        check(lc.ok && lc.value.size() == 24, "enumeration under a loud logits bound");
        int rejected = 0;
        for (const Candidate& cnd : lc.value) {
            const bool k_is_8 = cnd.plan.get("argmax_k").as_int() == 8;
            check(cnd.feasible == !k_is_8, "exactly the argmax_k = 8 plans are infeasible");
            if (!cnd.feasible) {
                check(cnd.reason == "KARY_MAGNITUDE" && prefix(cnd.detail, "KARY_MAGNITUDE"), "stable code KARY_MAGNITUDE with a detail");
                ++rejected;
            }
        }
        check(rejected == 8, "8 of 24 plans rejected");
        Shapes wide = sh;
        wide.score_bound = 3000.0;    // attention scores: beyond the node of 4, inside the binary bound
        Expected<Candidate> s4 = evaluate_plan(make_plan("generic", "generic", 2, 4), st, cm, env, wide);
        Expected<Candidate> s2 = evaluate_plan(make_plan("generic", "generic", 2, 2), st, cm, env, wide);
        check(s4.ok && !s4.value.feasible && s4.value.reason == "KARY_MAGNITUDE" && s4.value.detail.find("softmax") != std::string::npos, "max_k = 4 refused on the score bound");
        check(s2.ok && s2.value.feasible, "max_k = 2 admitted on the same score bound");
        Shapes none = sh;
        none.logits_bound = 0.0;      // no declared bound: nothing k-ary can be admitted (fail closed)
        Expected<Candidate> nb = evaluate_plan(make_plan("generic", "generic", 2, 2), st, cm, env, none);
        check(nb.ok && !nb.value.feasible && nb.value.reason == "KARY_MAGNITUDE", "a missing declared bound rejects the plan");
        // numerics are checked before resources: both violated -> KARY_MAGNITUDE names the reason
        Shapes both = loud;
        both.mem_limit_bytes = 1024;
        Expected<Candidate> bc = evaluate_plan(make_plan("wmask", "wmask", 8, 2), st, cm, env, both);
        Expected<Candidate> bm = evaluate_plan(make_plan("wmask", "wmask", 4, 2), st, cm, env, both);
        check(bc.ok && !bc.value.feasible && bc.value.reason == "KARY_MAGNITUDE" && bm.ok && bm.value.reason == "MEMORY_INFEASIBLE", "filter order: numerics, then memory");
        // the controller never selects a rejected plan, even when it is by far the cheapest
        SessionState kst = st;
        kst.incumbent = make_plan("generic", "generic", 2, 2);
        std::vector<Candidate> two;
        Expected<Candidate> keepc = evaluate_plan(kst.incumbent, kst, cm, env, loud);
        Expected<Candidate> fast8 = evaluate_plan(make_plan("generic", "generic", 8, 2), kst, cm, env, loud);
        check(keepc.ok && fast8.ok && !fast8.value.feasible, "challenger rejected by the filter");
        two.push_back(keepc.value);
        Candidate cheap = fast8.value;
        cheap.exec_s_per_epoch = 0.0;
        cheap.c_install_s = 0.0;
        two.push_back(cheap);
        ControllerConfig greedy_cfg;
        greedy_cfg.stationary_survival = true;
        greedy_cfg.use_uncertainty = false;
        greedy_cfg.hysteresis_s = 0.0;
        StateAwareController ctl(greedy_cfg);
        Expected<Decision> kd = ctl.decide(kst, two, 0);
        check(kd.ok && !kd.value.switched && same_plan(kd.value.chosen, kst.incumbent), "a KARY_MAGNITUDE plan is never chosen");
        Expected<Candidate> crt = Candidate::from_json(fast8.value.to_json());
        check(crt.ok && crt.value.reason == "KARY_MAGNITUDE" && crt.value.detail == fast8.value.detail && crt.value.to_json() == fast8.value.to_json(),
              "rejected candidate json round trip keeps code and detail");
        Expected<Shapes> lrt = Shapes::from_json(loud.to_json());
        check(lrt.ok && lrt.value.logits_bound == 1500.0 && lrt.value.score_bound == sh.score_bound, "declared bounds travel with the shapes");
    }
    check(prefix(validate_plan(make_plan("wmask", "wmask", 3, 2)).code, "PLAN_INVALID"), "argmax_k 3 invalid");
    check(prefix(evaluate_plan(Json("x"), st, cm, env, sh).code, "PLAN_INVALID"), "non-object plan invalid");
    Expected<Candidate> rt = Candidate::from_json(ww->to_json());
    check(rt.ok && rt.value.to_json() == ww->to_json(), "candidate json round trip");
    Expected<SessionState> srt = SessionState::from_json(st2.to_json());
    check(srt.ok && srt.value.to_json() == st2.to_json(), "state json round trip");
    Expected<Shapes> shrt = Shapes::from_json(sh.to_json());
    check(shrt.ok && shrt.value.to_json() == sh.to_json(), "shapes json round trip");
    std::printf("  candidates ok\n");
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------
static Candidate simple_candidate(const Json& plan, double exec, double install, const std::string& object, double sigma = 0.0,
                                  double install_sigma = 0.0) {
    Candidate c;
    c.plan = plan;
    c.exec_s_per_epoch = exec;
    c.exec_sigma = sigma;
    c.c_install_s = install;
    c.c_install_sigma = install_sigma;
    if (!object.empty() && install > 0.0) {
        c.install_objects.push_back(object);
        c.install_cost_by_object[object] = install;
    }
    return c;
}

static void test_controller_install_example() {
    const Json K = make_plan("generic", "generic", 2, 2);
    const Json A = make_plan("wmask", "wmask", 2, 2);
    const Json B = make_plan("generic", "wmask", 2, 2);
    SessionState st;
    st.incumbent = K;
    st.remaining_epochs = 5;
    std::vector<Candidate> cands = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 1.0, 100.0, "objA"), simple_candidate(B, 4.0, 1.0, "objB")};

    ControllerConfig cfg;
    cfg.stationary_survival = true;
    cfg.use_uncertainty = false;
    cfg.hysteresis_s = 0.0;
    cfg.min_residency = 0;
    StateAwareController ctl(cfg);
    Expected<Decision> d = ctl.decide(st, cands, 0);
    check(d.ok, "install example decide: " + d.code);
    check(near(d.value.q_keep, 50.0) && near(d.value.q_best, 21.0) && same_plan(d.value.chosen, B) && d.value.switched, "install example: keep 50 / A 105 / B 21 -> B");
    check(near(d.value.lower_bound_gain, 29.0) && d.value.u_diff == 0.0, "install example: lower bound gain 29");
    check(d.value.candidates.size() == 3 && near(d.value.candidates[1].get("q_mean").as_double(), 105.0), "install example: record A = 105");

    // the original rule (install ignored when ranking) picks A: the counterexample
    ControllerConfig naive = cfg;
    naive.ignore_install = true;
    StateAwareController ctl_naive(naive);
    Expected<Decision> dn = ctl_naive.decide(st, cands, 0);
    check(dn.ok && same_plan(dn.value.chosen, A), "install-blind ranking selects A (the counterexample)");

    // full B4 defaults with the stationary assumption: zero sigmas, hysteresis 0.5 -> still B
    ControllerConfig full;
    full.stationary_survival = true;
    StateAwareController ctl_full(full);
    Expected<Decision> df = ctl_full.decide(st, cands, 0);
    check(df.ok && same_plan(df.value.chosen, B) && df.value.switched && df.value.reason == "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS", "B4 default picks B");

    // the assumption set is exposed, with no unconditional no-harm claim
    bool has_no_harm = false, has_horizon = false;
    for (const std::string& a : df.value.assumptions) {
        if (prefix(a, "NO_UNCONDITIONAL_NO_HARM")) has_no_harm = true;
        if (prefix(a, "HORIZON_DECLARED_BUDGET")) has_horizon = true;
    }
    check(has_no_harm && has_horizon && df.value.assumptions.size() >= 6, "the decision record exposes the assumption set");
    Json dj = df.value.to_json();
    check(dj.get("assumptions").size() == df.value.assumptions.size() && dj.get("chosen") == B, "decision json");

    // uncertainty: large sigma on B makes its lower bound fall below hysteresis -> keep with a rejected challenger
    std::vector<Candidate> unc = {simple_candidate(K, 10.0, 0.0, "", 0.0), simple_candidate(A, 1.0, 100.0, "objA"),
                                  simple_candidate(B, 4.0, 1.0, "objB", 20.0)};
    StateAwareController ctl_u(full);
    Expected<Decision> du = ctl_u.decide(st, unc, 0);
    check(du.ok && !du.value.switched && du.value.reason == "GAIN_BELOW_UNCERTAINTY_OR_HYSTERESIS", "uncertainty blocks the switch");
    check(near(du.value.u_diff, 1.0 * std::sqrt(5.0 * 20.0 * 20.0)), "U_diff sums the per-epoch variance over the horizon");
    // with epoch_corr = 1 the variance is (sum s_j)^2 sigma^2
    ControllerConfig corr = full;
    corr.epoch_corr = 1.0;
    StateAwareController ctl_c(corr);
    Expected<Decision> dc = ctl_c.decide(st, unc, 0);
    check(dc.ok && !dc.value.switched, "fully correlated residuals: still keep");
    double u_b = -1.0;
    for (const Json& rec : dc.value.candidates)
        if (rec.get("plan") == B) u_b = rec.get("u_diff").as_double();
    check(near(u_b, 5.0 * 20.0), "fully correlated residuals: U_diff(B) = H sigma in the candidate record");

    // infeasible candidates are excluded, keep missing fails closed
    std::vector<Candidate> inf = cands;
    inf[2].feasible = false;
    inf[2].reason = "INSUFFICIENT_MATERIAL";
    StateAwareController ctl_i(cfg);
    Expected<Decision> di = ctl_i.decide(st, inf, 0);
    check(di.ok && !di.value.switched && di.value.reason == "INCUMBENT_OPTIMAL", "infeasible B excluded, A not worth it -> keep");
    SessionState st_nokeep = st;
    st_nokeep.incumbent = make_plan("wmask", "generic", 8, 4);
    check(prefix(ctl_i.decide(st_nokeep, cands, 0).code, "NO_KEEP_CANDIDATE"), "missing keep fails closed");
    check(prefix(ctl_i.decide(st, {}, 0).code, "NO_CANDIDATES"), "no candidates fails closed");

    // min_residency: a switch at epoch 0 blocks a better challenger at epoch 1, allowed at epoch 2
    ControllerConfig res = cfg;
    res.min_residency = 2;
    StateAwareController ctl_r(res);
    Expected<Decision> r0 = ctl_r.decide(st, cands, 0);
    check(r0.ok && r0.value.switched && same_plan(r0.value.chosen, B), "residency: first switch allowed");
    SessionState st_b = st;
    st_b.incumbent = B;
    st_b.remaining_epochs = 4;
    std::vector<Candidate> cands_b = {simple_candidate(B, 4.0, 0.0, ""), simple_candidate(A, 1.0, 2.0, "objA")};
    Expected<Decision> r1 = ctl_r.decide(st_b, cands_b, 1);
    check(r1.ok && !r1.value.switched && r1.value.reason == "MIN_RESIDENCY_HOLD", "residency hold at epoch 1");
    Expected<Decision> r2 = ctl_r.decide(st_b, cands_b, 2);
    check(r2.ok && r2.value.switched && same_plan(r2.value.chosen, A), "residency released at epoch 2");
    std::printf("  controller install example and assumption set ok\n");
}

static void test_controller_stationary_cold_start() {
    const Json G = make_plan("generic", "generic", 2, 2);
    const Json W = make_plan("wmask", "wmask", 2, 2);
    ControllerConfig cfg;   // defaults: dwell prior 5, z = 1, hysteresis 0.5, min_residency 1
    StateAwareController ctl(cfg);
    for (int i = 0; i < 10; ++i) {
        TelemetrySample s;
        s.seq = i;
        s.t = i * 1.0;
        s.bandwidth_bps = 100e6;
        s.rtt_s = 0.02;
        s.epoch = i;
        ctl.observe(s);
    }
    check(ctl.dwell().completed_dwells() == 0 && ctl.dwell().current_age() == 10 && ctl.dwell().transitions() == 9, "stationary link: no regime change");
    check(near(ctl.dwell().hazard(), 1.0 / 14.0) && near(ctl.dwell().mean_dwell(), 14.0), "hazard (0 + 1) / (9 transitions + prior 5)");
    std::vector<double> s = ctl.survival(20);
    check(s.size() == 20 && s[0] == 1.0 && near(s[1], 13.0 / 14.0) && near(s[19], std::pow(13.0 / 14.0, 19.0)), "geometric survival (1 - h)^j");
    SessionState st;
    st.incumbent = G;
    st.remaining_epochs = 20;
    std::vector<Candidate> cands = {simple_candidate(G, 10.0, 0.0, ""), simple_candidate(W, 2.0, 30.0, "objW", 0.5, 1.0)};
    Expected<Decision> d = ctl.decide(st, cands, 0);
    check(d.ok && d.value.switched && same_plan(d.value.chosen, W), "stationary cold start installs the cheaper lane");
    // after the install the incumbent is optimal: no further switch over the next epochs
    SessionState st2;
    st2.incumbent = W;
    st2.remaining_epochs = 19;
    std::vector<Candidate> after = {simple_candidate(W, 2.0, 0.0, "", 0.5), simple_candidate(G, 10.0, 0.0, "", 0.5)};
    for (int e = 1; e <= 5; ++e) {
        TelemetrySample s2;
        s2.seq = 10 + e;
        s2.bandwidth_bps = 100e6;
        s2.rtt_s = 0.02;
        ctl.observe(s2);
        st2.remaining_epochs = 20 - e;
        Expected<Decision> d2 = ctl.decide(st2, after, e);
        check(d2.ok && !d2.value.switched && d2.value.reason == "INCUMBENT_OPTIMAL", "stationary link: no further switching once the incumbent is optimal");
    }
    // a regime change shortens the horizon: the dwell model records a completed dwell
    TelemetrySample ch;
    ch.bandwidth_bps = 20e6;
    ch.rtt_s = 0.02;
    ctl.observe(ch);
    check(ctl.dwell().completed_dwells() == 1 && ctl.dwell().current_age() == 1, "bandwidth drop declares a new regime");
    check(ctl.dwell().mean_dwell() < 14.0, "observed dwell shortens the estimate");
    // a short-dwell history makes a long-payback install unattractive
    StateAwareController flappy(cfg);
    for (int i = 0; i < 12; ++i) {
        TelemetrySample s3;
        s3.bandwidth_bps = (i % 2 == 0) ? 100e6 : 20e6;
        s3.rtt_s = 0.02;
        flappy.observe(s3);
    }
    check(flappy.dwell().completed_dwells() == 11 && near(flappy.dwell().mean_dwell(), (11.0 + 5.0) / 12.0), "alternating link: dwell about 1.3 epochs");
    std::vector<Candidate> payback = {simple_candidate(G, 10.0, 0.0, ""), simple_candidate(W, 2.0, 30.0, "objW")};
    Expected<Decision> dfl = flappy.decide(st, payback, 0);
    check(dfl.ok && !dfl.value.switched, "short dwells: the 30 s install is not covered by the survival-weighted horizon");
    std::printf("  controller stationary cold start ok\n");
}

static void test_controller_brute_force() {
    SplitMix64 rng(2024);
    const int H = 5;
    const int C = 3;
    const Json plans[3] = {make_plan("generic", "generic", 2, 2), make_plan("wmask", "wmask", 2, 2), make_plan("generic", "wmask", 4, 2)};
    int switches = 0;
    for (int table = 0; table < 50; ++table) {
        std::vector<double> exec(C), install(C, 0.0);
        for (int p = 0; p < C; ++p) exec[static_cast<std::size_t>(p)] = 1.0 + 19.0 * rng.uniform01();
        for (int p = 1; p < C; ++p) install[static_cast<std::size_t>(p)] = 40.0 * rng.uniform01();
        std::vector<std::vector<double>> table_exec(static_cast<std::size_t>(H), exec);
        Expected<SequenceOptimum> opt = brute_force_sequence(table_exec, install, 0);
        check(opt.ok, "brute force sequence");

        ControllerConfig cfg;
        cfg.stationary_survival = true;
        cfg.use_uncertainty = false;
        cfg.hysteresis_s = 0.0;
        cfg.min_residency = 0;
        StateAwareController ctl(cfg);
        Expected<std::unique_ptr<Policy>> b2 = make_policy("B2_remaining_plus_install");
        check(b2.ok, "B2 policy");
        int cur = 0;
        std::vector<bool> used(C, false);
        used[0] = true;
        double realized = 0.0;
        for (int t = 0; t < H; ++t) {
            SessionState st;
            st.incumbent = plans[cur];
            st.remaining_epochs = H - t;
            std::vector<Candidate> cands;
            cands.push_back(simple_candidate(plans[cur], exec[static_cast<std::size_t>(cur)], 0.0, ""));
            for (int p = 0; p < C; ++p) {
                if (p == cur) continue;
                cands.push_back(simple_candidate(plans[p], exec[static_cast<std::size_t>(p)], used[static_cast<std::size_t>(p)] ? 0.0 : install[static_cast<std::size_t>(p)],
                                                 "obj" + std::to_string(p)));
            }
            Expected<Decision> d = ctl.decide(st, cands, t);
            check(d.ok, "decide in table");
            Expected<Decision> d2 = b2.value->decide(st, cands, Env(), t);
            check(d2.ok && same_plan(d2.value.chosen, d.value.chosen) && d2.value.switched == d.value.switched, "B2 policy equals the stationary controller");
            int next = cur;
            for (int p = 0; p < C; ++p)
                if (same_plan(d.value.chosen, plans[p])) next = p;
            if (next != cur) {
                ++switches;
                if (!used[static_cast<std::size_t>(next)]) {
                    realized += install[static_cast<std::size_t>(next)];
                    used[static_cast<std::size_t>(next)] = true;
                }
            }
            realized += exec[static_cast<std::size_t>(next)];
            cur = next;
        }
        check(near(realized, opt.value.total, 1e-9), "stationary rule realizes the brute-force optimum (table " + std::to_string(table) + ")");
    }
    check(switches > 0, "some tables switch");
    // a non-stationary counter check: brute force over a time-varying table differs from the stationary rule in general
    std::vector<std::vector<double>> tv = {{10, 20, 1}, {10, 20, 1}, {10, 2, 1}, {10, 2, 1}, {10, 2, 1}};
    Expected<SequenceOptimum> opt2 = brute_force_sequence(tv, {0.0, 1.0, 100.0}, 0);
    check(opt2.ok && near(opt2.value.total, 27.0) && opt2.value.sequence[0] == 0 && opt2.value.sequence[2] == 1, "time-varying brute force: keep, keep, B, B, B = 27");
    check(prefix(brute_force_sequence(tv, {0.0, 1.0}, 0).code, "BRUTE_FORCE_SHAPE"), "ragged table refused");
    std::printf("  controller brute force ok\n");
}

static void test_policies() {
    const Json K = make_plan("generic", "generic", 2, 2);
    const Json A = make_plan("wmask", "wmask", 2, 2);
    const Json B = make_plan("generic", "wmask", 2, 2);
    SessionState st;
    st.incumbent = K;
    st.remaining_epochs = 5;
    std::vector<Candidate> cands = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 1.0, 100.0, "objA"), simple_candidate(B, 20.0, 1.0, "objB")};
    OracleTrace tr;
    tr.plan_keys = {plan_key(K), plan_key(A), plan_key(B)};
    tr.exec_s = {{10, 1, 20}, {10, 1, 20}, {10, 1, 2}, {10, 1, 2}, {10, 1, 2}};
    Expected<OracleTrace> trt = OracleTrace::from_json(tr.to_json());
    check(trt.ok && trt.value.to_json() == tr.to_json(), "trace json round trip");

    Expected<std::unique_ptr<Policy>> b5 = make_policy("B5_oracle");
    check(b5.ok, "B5 policy");
    check(prefix(b5.value->decide(st, cands, Env(), 0).code, "ORACLE_NEEDS_TRACE"), "oracle without trace fails closed");
    b5.value->set_trace(tr);
    Expected<Decision> d5 = b5.value->decide(st, cands, Env(), 0);
    check(d5.ok && !d5.value.switched && near(d5.value.q_best, 27.0) && near(d5.value.q_keep, 50.0), "B5 exact DP: keep now, B later (27)");
    SessionState st2 = st;
    st2.remaining_epochs = 3;
    Expected<Decision> d5b = b5.value->decide(st2, cands, Env(), 2);
    check(d5b.ok && d5b.value.switched && same_plan(d5b.value.chosen, B) && near(d5b.value.q_best, 7.0), "B5 switches to B at epoch 2");

    Expected<std::unique_ptr<Policy>> b6 = make_policy("B6_zero_overhead");
    b6.value->set_trace(tr);
    Expected<Decision> d6 = b6.value->decide(st, cands, Env(), 0);
    check(d6.ok && d6.value.switched && same_plan(d6.value.chosen, A) && near(d6.value.q_best, 1.0), "B6 picks the per-epoch cheapest, install not charged");

    Expected<std::unique_ptr<Policy>> hind = make_policy("B0_fixed_best_hindsight");
    hind.value->set_trace(tr);
    Expected<Decision> dh = hind.value->decide(st, cands, Env(), 0);
    // fixed totals over the trace: K = 50, A = 100 + 5 = 105, B = 1 + 20 + 20 + 2 + 2 + 2 = 47
    check(dh.ok && dh.value.switched && same_plan(dh.value.chosen, B), "best fixed plan in hindsight is B (47 < 50 < 105)");
    Expected<Decision> dh2 = hind.value->decide(st, cands, Env(), 1);
    check(dh2.ok && same_plan(dh2.value.chosen, B), "hindsight fixed plan stays fixed");

    Expected<std::unique_ptr<Policy>> fixed = make_policy("B0_fixed:" + plan_key(B));
    check(fixed.ok, "fixed policy parse");
    Expected<Decision> dfx = fixed.value->decide(st, cands, Env(), 0);
    check(dfx.ok && dfx.value.switched && same_plan(dfx.value.chosen, B) && dfx.value.reason == "FIXED_PLAN_INSTALL", "fixed plan installs B at epoch 0");
    check(prefix(make_policy("B0_fixed:{\"embedding\":\"x\"}").code, "PLAN_INVALID"), "fixed policy validates its plan");

    Expected<std::unique_ptr<Policy>> b1 = make_policy("B1_naive");
    Expected<Decision> d1 = b1.value->decide(st, cands, Env(), 0);
    check(d1.ok && same_plan(d1.value.chosen, A), "B1 naive picks the cheapest per epoch regardless of install");

    std::vector<Candidate> example = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 1.0, 100.0, "objA"), simple_candidate(B, 4.0, 1.0, "objB")};
    for (const char* name : {"B2_remaining_plus_install", "B4_no_hyst", "B4_no_unc"}) {
        ControllerConfig base;
        base.stationary_survival = true;
        Expected<std::unique_ptr<Policy>> p = make_policy(name, base);
        check(p.ok, std::string("policy ") + name);
        Expected<Decision> d = p.value->decide(st, example, Env(), 0);
        check(d.ok && same_plan(d.value.chosen, B) && d.value.policy == name, std::string(name) + " picks B on the install example");
    }
    // Planner x controller factorial (paper, Section 4.4). B3 = B2 control on the static planner's
    // per-epoch assignment: on the install example the planner assigns A (cheapest per epoch), whose
    // total 105 loses against keep 50, so B3 keeps (B2, which ranks every candidate itself, moves to
    // B = 21). The planner of B2 and B4 ranks the candidates at the horizon cost instead.
    {
        ControllerConfig base;
        base.stationary_survival = true;
        Expected<std::unique_ptr<Policy>> b3 = make_policy("B3_planner_plus_B2", base);
        check(b3.ok && b3.value->name() == "B3_planner_plus_B2", "B3 policy");
        Expected<Decision> d3 = b3.value->decide(st, example, Env(), 0);
        check(d3.ok && !d3.value.switched && same_plan(d3.value.chosen, K) && d3.value.policy == "B3_planner_plus_B2", "B3 on the install example: planner assigns A, B2 control keeps");
        check(near(d3.value.q_keep, 50.0) && near(d3.value.lower_bound_gain, 50.0 - 105.0), "B3 compares keep with the assigned plan only");
        bool noted = false;
        for (const std::string& a : d3.value.assumptions) noted = noted || prefix(a, "PLANNER_ASSIGNMENT");
        check(noted, "the decision record names the planner assignment");
        int excluded = 0, assigned = 0;
        for (const Json& rec : d3.value.candidates) {
            if (rec.get("excluded").as_string() == "NOT_PLANNER_ASSIGNMENT") ++excluded;
            if (rec.get("planner_assigned").as_bool()) ++assigned;
        }
        check(d3.value.candidates.size() == 3 && excluded == 1 && assigned == 1, "all candidates are recorded; one assigned, one excluded");
        // an install that pays back inside the horizon: B3 follows the planner to A (25), B2 prefers B (21)
        std::vector<Candidate> pay = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 1.0, 20.0, "objA"), simple_candidate(B, 4.0, 1.0, "objB")};
        Expected<Decision> d3b = make_policy("B3_planner_plus_B2", base).value->decide(st, pay, Env(), 0);
        Expected<Decision> d2b = make_policy("B2_remaining_plus_install", base).value->decide(st, pay, Env(), 0);
        check(d3b.ok && d3b.value.switched && same_plan(d3b.value.chosen, A) && near(d3b.value.q_best, 25.0), "B3 switches to the planner's plan when B2 control accepts it");
        check(d2b.ok && same_plan(d2b.value.chosen, B), "B2 without a planner ranks all candidates and picks B");
        // B4 with a planner (name suffix): the planner assigns the least horizon cost (Qhat without the
        // install term plus the install: keep 50, A 105, B 21), the full rule decides on that challenger
        Expected<std::unique_ptr<Policy>> b4p = make_policy("B4_opencrab+planner:exact", base);
        check(b4p.ok && b4p.value->name() == "B4_opencrab+planner:exact", "B4 with the ExactAssign planner");
        Expected<Decision> d4 = b4p.value->decide(st, example, Env(), 0);
        check(d4.ok && d4.value.switched && same_plan(d4.value.chosen, B) && d4.value.reason == "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS" &&
                  d4.value.policy == "B4_opencrab+planner:exact",
              "B4 + planner on the install example: the planner assigns B (horizon cost 21), the rule switches");
        bool costed = false;
        for (const Json& rec : d4.value.candidates)
            if (rec.has("planner_cost") && same_plan(rec.get("plan"), A)) costed = near(rec.get("planner_cost").as_double(), 105.0);
        check(costed, "the excluded candidate records its horizon cost (A = 100 + 5 x 1)");
        Expected<Decision> d4b = make_policy("B4_opencrab+planner:ilp", base).value->decide(st, pay, Env(), 0);
        check(d4b.ok && d4b.value.switched && same_plan(d4b.value.chosen, B) && d4b.value.reason == "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS",
              "B4 + planner: B (21) beats A (25) at the horizon cost, the bound clears hysteresis");
        Expected<Decision> d4c = make_policy("B4_opencrab", base).value->decide(st, pay, Env(), 0);
        check(d4c.ok && same_plan(d4c.value.chosen, B), "B4 without a planner ranks all candidates (B = 21)");
        Expected<Decision> d2p = make_policy("B2_remaining_plus_install+planner:greedy", base).value->decide(st, pay, Env(), 0);
        check(d2p.ok && same_plan(d2p.value.chosen, B), "B2 + planner: the per-step horizon plus install assigns B, as B2 alone");
        Expected<Decision> d4n = make_policy("B4_no_install+planner:exact", base).value->decide(st, example, Env(), 0);
        check(d4n.ok && same_plan(d4n.value.chosen, A), "B4_no_install + planner ranks without the install (A = 5)");
        // a cold prefill on a fast link: per epoch the incumbent generic plan is cheaper, over the
        // horizon the weight-mask plan with its install is cheaper; every planner of B2 and B4 must
        // follow the horizon (the per-epoch assignment of B3 keeps the incumbent)
        {
            Candidate gk = simple_candidate(K, 10.0, 0.0, "");
            gk.exec_s_by_epoch = {10.0, 2.0};
            gk.exec_sigma_by_epoch = {0.0, 0.0};
            Candidate wm = simple_candidate(A, 11.0, 3.0, "objA");
            wm.exec_s_by_epoch = {11.0, 0.5};
            wm.exec_sigma_by_epoch = {0.0, 0.0};
            SessionState cold = st;
            cold.remaining_epochs = 8;   // keep 10 + 7 x 2 = 24, weight masks 3 + 11 + 7 x 0.5 = 17.5
            const std::vector<Candidate> prefill = {gk, wm};
            for (const char* pn : {"B4_opencrab+planner:exact", "B4_opencrab+planner:ilp", "B2_remaining_plus_install+planner:exact"}) {
                Expected<Decision> dp = make_policy(pn, base).value->decide(cold, prefill, Env(), 0);
                check(dp.ok && dp.value.switched && same_plan(dp.value.chosen, A), std::string(pn) + ": cold prefill installs the weight masks");
            }
            Expected<Decision> dplain = make_policy("B4_opencrab", base).value->decide(cold, prefill, Env(), 0);
            check(dplain.ok && same_plan(dplain.value.chosen, A), "B4 without a planner installs the weight masks too");
            Expected<Decision> dstatic = make_policy("B3_planner_plus_B2", base).value->decide(cold, prefill, Env(), 0);
            check(dstatic.ok && !dstatic.value.switched, "B3: the static per-epoch assignment is the incumbent (10 < 11)");
        }
        // every built-in planner assigns the same plan: the first feasible minimum of the given cost
        SplitMix64 rng(99);
        const Json plans[4] = {K, A, B, make_plan("wmask", "generic", 4, 4)};
        for (int trial = 0; trial < 40; ++trial) {
            std::vector<Candidate> cs;
            cs.push_back(simple_candidate(plans[0], 1.0 + 9.0 * rng.uniform01(), 0.0, ""));
            for (int i = 1; i < 4; ++i) {
                // Draws go into named locals: the order in which call arguments are evaluated differs between compilers.
                const double exec_s = trial % 5 == 0 ? 3.0 : 1.0 + 9.0 * rng.uniform01();
                const double install_s = 10.0 * rng.uniform01();
                Candidate cnd = simple_candidate(plans[i], exec_s, install_s, "o" + std::to_string(i));
                if (rng.uniform01() < 0.25) { cnd.feasible = false; cnd.reason = "INSUFFICIENT_MATERIAL"; }
                cs.push_back(cnd);
            }
            std::vector<double> per_epoch, other;
            for (const Candidate& c : cs) {
                per_epoch.push_back(c.exec_s_per_epoch);
                const double draw = trial % 7 == 0 ? 2.0 : 1.0 + 30.0 * rng.uniform01();   // ties included
                other.push_back(draw);
            }
            for (const std::vector<double>* cost : {&per_epoch, &other}) {
                int want = -1;
                for (std::size_t i = 0; i < cs.size(); ++i)
                    if (cs[i].feasible && (want < 0 || (*cost)[i] < (*cost)[static_cast<std::size_t>(want)])) want = static_cast<int>(i);
                for (const char* pn : {"exact", "ilp", "greedy", "argmin"}) {
                    Expected<PlannerHook> h = planner_hook(pn);
                    check(h.ok, std::string("planner hook ") + pn);
                    Expected<int> got = h.value(st, cs, Env(), *cost);
                    check(got.ok && got.value == want, std::string("planner ") + pn + " assigns the first feasible minimum of the cost (trial " + std::to_string(trial) + ")");
                }
            }
        }
        std::vector<Candidate> none = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 1.0, 1.0, "objA")};
        none[0].feasible = false;
        none[1].feasible = false;
        check(planner_hook("exact").value(st, none, Env(), {10.0, 1.0}).value == 0, "nothing feasible: the assignment is the incumbent");
        check(prefix(planner_hook("greedy").value(st, example, Env(), {1.0}).code, "PLANNER_PARAMS") &&
                  prefix(planner_hook("argmin").value(st, example, Env(), {}).code, "PLANNER_PARAMS"),
              "one cost per candidate is required");
        SessionState nokeep = st;
        nokeep.incumbent = make_plan("wmask", "generic", 8, 4);
        check(prefix(planner_hook("ilp").value(nokeep, example, Env(), {1.0, 2.0, 3.0}).code, "NO_KEEP_CANDIDATE"), "hook without keep fails closed");
        check(prefix(planner_hook("nope").code, "UNKNOWN_PLANNER") && prefix(make_policy("B4_opencrab+planner:nope", base).code, "UNKNOWN_PLANNER"), "unknown planner refused");
        check(prefix(make_policy("B5_oracle+planner:exact", base).code, "POLICY_PARAMS") && prefix(make_policy("delayed:2+planner:exact", base).code, "POLICY_PARAMS"),
              "policies without a controller take no planner");
        // explicit hook: any deterministic public assignment can be plugged in
        PlannerHook pick_b = [&](const SessionState&, const std::vector<Candidate>& cs, const Env&, const std::vector<double>&) -> Expected<int> {
            for (std::size_t i = 0; i < cs.size(); ++i)
                if (same_plan(cs[i].plan, B)) return Expected<int>::success(static_cast<int>(i));
            return Expected<int>::failure("NO_B");
        };
        Expected<std::unique_ptr<Policy>> custom = make_policy("B2_remaining_plus_install", base, pick_b);
        check(custom.ok, "policy with an explicit hook");
        Expected<Decision> dc = custom.value->decide(st, example, Env(), 0);
        check(dc.ok && dc.value.switched && same_plan(dc.value.chosen, B), "explicit hook: B2 control on the hook's assignment");
        std::vector<Candidate> no_b = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 1.0, 100.0, "objA")};
        check(prefix(custom.value->decide(st, no_b, Env(), 0).code, "PLANNER_FAILED"), "a failing hook fails the decision closed");
        PlannerHook wild = [](const SessionState&, const std::vector<Candidate>&, const Env&, const std::vector<double>&) { return Expected<int>::success(99); };
        check(prefix(make_policy("B4_opencrab", base, wild).value->decide(st, example, Env(), 0).code, "PLANNER_FAILED"), "an out-of-range assignment is refused");
        std::vector<Candidate> inf_b = example;
        inf_b[2].feasible = false;
        inf_b[2].reason = "KARY_MAGNITUDE";
        Expected<Decision> di = make_policy("B2_remaining_plus_install", base, pick_b).value->decide(st, inf_b, Env(), 0);
        check(di.ok && !di.value.switched && same_plan(di.value.chosen, K), "an infeasible assignment is never executed");
        check(prefix(make_policy("B4_opencrab+planner:exact", base, pick_b).code, "POLICY_PARAMS"), "hook and name suffix together are refused");
        Expected<std::unique_ptr<Policy>> plain = make_policy("B4_opencrab", base, PlannerHook());
        check(plain.ok && same_plan(plain.value->decide(st, example, Env(), 0).value.chosen, B), "an empty hook means no planner");
    }
    ControllerConfig base;
    base.stationary_survival = true;
    Expected<std::unique_ptr<Policy>> noinst = make_policy("B4_no_install", base);
    Expected<Decision> dni = noinst.value->decide(st, example, Env(), 0);
    check(dni.ok && same_plan(dni.value.chosen, A), "B4_no_install ablation picks A");

    // delayed:k is B1 with its replanning delayed (the Python NaivePolicy with a delay): the per-epoch
    // cheapest plan A is adopted once it has been the cheapest k + 1 = 3 epochs in a row
    Expected<std::unique_ptr<Policy>> del = make_policy("delayed:2", base);
    check(del.ok && del.value->name() == "delayed:2", "delayed policy");
    Expected<Decision> e0 = del.value->decide(st, example, Env(), 0);
    Expected<Decision> e1 = del.value->decide(st, example, Env(), 1);
    Expected<Decision> e2 = del.value->decide(st, example, Env(), 2);
    check(e0.ok && !e0.value.switched && e0.value.reason == "DELAYED_HOLD: 1/2" && near(e0.value.q_keep, 50.0) && near(e0.value.q_best, 5.0),
          "delayed: epoch 0 holds, install ignored as in B1");
    check(e1.ok && !e1.value.switched && e1.value.reason == "DELAYED_HOLD: 2/2", "delayed: epoch 1 holds");
    check(e2.ok && e2.value.switched && same_plan(e2.value.chosen, A) && e2.value.reason == "DELAYED_REPLAN: k=2",
          "delayed: epoch 2 adopts the per-epoch cheapest plan A");
    {
        // a change of the cheapest plan restarts the count; the incumbent being cheapest keeps it
        Expected<std::unique_ptr<Policy>> d2 = make_policy("delayed:2", base);
        std::vector<Candidate> b_cheap = {simple_candidate(K, 10.0, 0.0, ""), simple_candidate(A, 5.0, 100.0, "objA"), simple_candidate(B, 4.0, 1.0, "objB")};
        Expected<Decision> f0 = d2.value->decide(st, example, Env(), 0);
        Expected<Decision> f1 = d2.value->decide(st, b_cheap, Env(), 1);
        Expected<Decision> f2 = d2.value->decide(st, b_cheap, Env(), 2);
        Expected<Decision> f3 = d2.value->decide(st, b_cheap, Env(), 3);
        check(f0.ok && f1.ok && f2.ok && f3.ok && !f1.value.switched && f1.value.reason == "DELAYED_HOLD: 1/2" && !f2.value.switched &&
                  f3.value.switched && same_plan(f3.value.chosen, B),
              "delayed: a new cheapest plan restarts the count");
        SessionState on_a = st;
        on_a.incumbent = A;
        Expected<Decision> f4 = d2.value->decide(on_a, example, Env(), 4);
        check(f4.ok && !f4.value.switched && f4.value.reason == "ALREADY_CHEAPEST", "delayed: the incumbent is the cheapest");
    }
    check(prefix(make_policy("delayed:0").code, "POLICY_PARAMS"), "delayed:0 refused");
    check(prefix(make_policy("nope").code, "UNKNOWN_POLICY"), "unknown policy");
    check(policy_names().size() == 12, "12 policy names");
    std::printf("  policies ok\n");
}

// Per-step horizon pricing: the candidates carry the predicted cost of every remaining step (the
// prefill is dear, the decode steps are cheap), the controller sums the steps instead of H times
// the current epoch, and the baselines follow (B1 excepted: per epoch by definition).
static void test_controller_schedule() {
    const Json K = make_plan("generic", "generic", 2, 2);
    const Json A = make_plan("wmask", "wmask", 2, 2);
    const Json B = make_plan("generic", "wmask", 2, 2);
    SessionState st;
    st.incumbent = K;
    st.remaining_epochs = 5;
    // keep: a dear prefill then cheap decode steps (list as long as the horizon); A: a shorter list,
    // its last entry stands for the further steps; B: a longer list, the surplus is not read
    Candidate keep = simple_candidate(K, 10.0, 0.0, "", 0.5);
    keep.exec_s_by_epoch = {10.0, 2.0, 2.0, 2.5, 2.5};
    keep.exec_sigma_by_epoch = {0.5, 0.1, 0.1, 0.1, 0.1};
    Candidate a = simple_candidate(A, 8.0, 12.0, "objA", 0.4, 0.2);
    a.exec_s_by_epoch = {8.0, 1.0, 1.0};
    a.exec_sigma_by_epoch = {0.4, 0.05, 0.05};
    Candidate b = simple_candidate(B, 4.0, 1.0, "objB", 0.3);
    b.exec_s_by_epoch = std::vector<double>(7, 4.0);
    b.exec_sigma_by_epoch = std::vector<double>(7, 0.3);
    const std::vector<Candidate> cands = {keep, a, b};
    const std::vector<double> surv = {1.0, 0.8, 0.64, 0.512, 0.4096};

    // the objective by hand
    {
        double ex = 0.0, a1 = 0.0, a2 = 0.0;
        for (std::size_t j = 0; j < surv.size(); ++j) {
            ex += surv[j] * keep.exec_s_by_epoch[j];
            const double aj = surv[j] * keep.exec_sigma_by_epoch[j];
            a1 += aj;
            a2 += aj * aj;
        }
        const QHat qk = q_hat(keep, st, surv, true, 0.0);
        check(near(qk.mean, ex) && near(qk.exec_sum_s, ex) && near(qk.variance, a2), "per-step Qhat(keep): weighted step costs, independent step variances");
        const QHat qk1 = q_hat(keep, st, surv, true, 1.0);
        check(near(qk1.variance, a1 * a1), "epoch_corr = 1: the variance is (sum a_j)^2");
        double exa = 12.0, b1 = 0.0, b2 = 0.0;
        for (std::size_t j = 0; j < surv.size(); ++j) {
            const std::size_t k = std::min(j, a.exec_s_by_epoch.size() - 1);
            exa += surv[j] * a.exec_s_by_epoch[k];
            const double aj = surv[j] * a.exec_sigma_by_epoch[k];
            b1 += aj;
            b2 += aj * aj;
        }
        const QHat qa = q_hat(a, st, surv, true, 0.5);
        check(near(qa.mean, exa) && near(qa.variance, b2 + 0.5 * (b1 * b1 - b2) + 0.2 * 0.2), "a shorter list: the last entry stands for the further steps; install sigma added");
        const QHat qa0 = q_hat(a, st, surv, false, 0.5);
        check(near(qa0.mean, exa - 12.0) && near(qa0.variance, b2 + 0.5 * (b1 * b1 - b2)), "install excluded from mean and variance");
        const QHat qb = q_hat(b, st, surv, true, 0.0);
        double s1 = 0.0, s2 = 0.0;
        for (double s : surv) { s1 += s; s2 += s * s; }
        check(near(qb.mean, 1.0 + s1 * 4.0) && near(qb.variance, s2 * 0.09), "a longer list with equal entries prices like one shape; the surplus is not read");
        Candidate flat = b;
        flat.exec_s_by_epoch.clear();
        flat.exec_sigma_by_epoch.clear();
        const QHat qf = q_hat(flat, st, surv, true, 0.3);
        check(near(qf.mean, 1.0 + s1 * 4.0) && near(qf.variance, 0.09 * (s2 + 0.3 * (s1 * s1 - s2))), "no lists: the one-shape form");
        check(near(horizon_exec(keep, 5), 19.0) && near(horizon_exec(a, 5), 12.0) && near(horizon_exec(b, 5), 20.0) && near(horizon_exec(flat, 5), 20.0) &&
                  near(horizon_exec(a, 2), 9.0) && horizon_exec(a, 0) == 0.0,
              "horizon_exec: per-step sum with the last entry repeated, else H times the epoch cost");
        check(exec_at(a, 0) == 8.0 && exec_at(a, 7) == 1.0 && exec_sigma_at(a, 9) == 0.05 && exec_at(flat, 3) == 4.0 && exec_sigma_at(flat, 3) == 0.3,
              "exec_at / exec_sigma_at index the lists with the last entry repeated");
    }
    // the lists decide: B2 on the per-step sums keeps (19 < 21 < 24), on one shape it would take B (50 vs 21)
    {
        ControllerConfig b2cfg;
        b2cfg.stationary_survival = true;
        b2cfg.use_uncertainty = false;
        b2cfg.hysteresis_s = 0.0;
        b2cfg.min_residency = 0;
        Expected<Decision> d2 = make_policy("B2_remaining_plus_install", b2cfg).value->decide(st, cands, Env(), 0);
        check(d2.ok && !d2.value.switched && near(d2.value.q_keep, 19.0) && d2.value.reason == "INCUMBENT_OPTIMAL", "B2 on the per-step sums keeps (19 against 24 and 21)");
        std::vector<Candidate> one = cands;
        for (Candidate& c : one) { c.exec_s_by_epoch.clear(); c.exec_sigma_by_epoch.clear(); }
        Expected<Decision> d2f = make_policy("B2_remaining_plus_install", b2cfg).value->decide(st, one, Env(), 0);
        check(d2f.ok && d2f.value.switched && same_plan(d2f.value.chosen, B) && near(d2f.value.q_keep, 50.0) && near(d2f.value.q_best, 21.0),
              "the same candidates without the lists switch to B: the lists decide");
        // B1 is per epoch: the lists are not read, the cheapest current epoch wins (B at 4 s, not A whose steps sum to 12)
        Expected<Decision> d1 = make_policy("B1_naive").value->decide(st, cands, Env(), 0);
        check(d1.ok && d1.value.switched && same_plan(d1.value.chosen, B) && near(d1.value.q_keep, 50.0) && near(d1.value.q_best, 20.0), "B1 ignores the per-step lists");
        // B0 reports the per-step horizon of keep and of the fixed plan
        Expected<Decision> d0 = make_policy("B0_fixed:" + plan_key(A)).value->decide(st, cands, Env(), 0);
        check(d0.ok && d0.value.switched && near(d0.value.q_keep, 19.0) && near(d0.value.q_best, 24.0), "B0 prices keep and the fixed plan step by step");
        bool horizon_recorded = false;
        for (const Json& rec : d0.value.candidates)
            if (same_plan(rec.get("plan"), A) && near(rec.get("exec_horizon_s").as_double(), 12.0)) horizon_recorded = true;
        check(horizon_recorded, "the baseline record carries the per-step horizon of every candidate");
        // B3: the planner assigns the cheapest current epoch (B), B2 control keeps (19 < 21)
        Expected<Decision> d3 = make_policy("B3_planner_plus_B2", b2cfg).value->decide(st, cands, Env(), 0);
        check(d3.ok && !d3.value.switched && near(d3.value.q_keep, 19.0) && near(d3.value.lower_bound_gain, 19.0 - 21.0), "B3: planner picks B per epoch, B2 control keeps on the per-step sums");
        // B4 with the default rule and stationary survival: no challenger has a positive lower bound
        ControllerConfig b4cfg;
        b4cfg.stationary_survival = true;
        Expected<Decision> d4 = make_policy("B4_opencrab", b4cfg).value->decide(st, cands, Env(), 0);
        check(d4.ok && !d4.value.switched && d4.value.reason == "INCUMBENT_OPTIMAL" && near(d4.value.q_keep, 19.0), "B4 keeps on the per-step objective");
        // the oracle falls back to the per-step prediction where the trace has no entry
        OracleTrace empty;
        Expected<std::unique_ptr<Policy>> b5 = make_policy("B5_oracle");
        b5.value->set_trace(empty);
        Expected<Decision> d5 = b5.value->decide(st, cands, Env(), 0);
        // the exact DP over switching sequences: B for the dear first step (1 + 4), then back to keep's
        // cheap decode steps (2 + 2 + 2.5 + 2.5) = 14, against keep throughout (19) and B throughout (21)
        check(d5.ok && d5.value.switched && same_plan(d5.value.chosen, B) && near(d5.value.q_keep, 19.0) && near(d5.value.q_best, 14.0),
              "B5 without trace entries prices every epoch on the per-step lists");
        Expected<Decision> d5f = b5.value->decide(st, one, Env(), 0);
        check(d5f.ok && d5f.value.switched && same_plan(d5f.value.chosen, B) && near(d5f.value.q_keep, 50.0) && near(d5f.value.q_best, 21.0),
              "the same oracle on one shape: B throughout (21) against keep (50)");
        Expected<std::unique_ptr<Policy>> b6 = make_policy("B6_zero_overhead");
        b6.value->set_trace(empty);
        Expected<Decision> d6 = b6.value->decide(st, cands, Env(), 0);
        check(d6.ok && d6.value.switched && same_plan(d6.value.chosen, B) && near(d6.value.q_best, 4.0), "B6 compares the current epoch (entry 0 of the lists)");
        Expected<std::unique_ptr<Policy>> hind = make_policy("B0_fixed_best_hindsight");
        hind.value->set_trace(empty);
        Expected<Decision> dh = hind.value->decide(st, cands, Env(), 0);
        check(dh.ok && !dh.value.switched && near(dh.value.q_keep, 19.0), "hindsight on the per-step lists: keep (19) is the best fixed plan");
    }
    // inconsistent lists fail closed, JSON carries the lists
    {
        Candidate bad = a;
        bad.exec_sigma_by_epoch.pop_back();
        check(prefix(validate_schedule(bad).code, "CANDIDATE_SCHEDULE"), "unequal list lengths are refused");
        ControllerConfig cfg;
        cfg.stationary_survival = true;
        StateAwareController ctl(cfg);
        check(prefix(ctl.decide(st, {keep, bad, b}, 0).code, "CANDIDATE_SCHEDULE"), "the decision fails closed on an inconsistent candidate");
        Json bj = bad.to_json();
        check(bj.get("exec_s_by_epoch").size() == 3 && bj.get("exec_sigma_by_epoch").size() == 2, "to_json writes both lists");
        check(prefix(Candidate::from_json(bj).code, "CANDIDATE_SCHEDULE"), "from_json refuses unequal lists");
        Expected<Candidate> rt = Candidate::from_json(a.to_json());
        check(rt.ok && rt.value.exec_s_by_epoch == a.exec_s_by_epoch && rt.value.exec_sigma_by_epoch == a.exec_sigma_by_epoch && rt.value.to_json() == a.to_json(),
              "candidate json round trip keeps the per-step lists");
        Json noj = keep.to_json();
        noj.erase("exec_s_by_epoch");
        noj.erase("exec_sigma_by_epoch");
        Expected<Candidate> missing = Candidate::from_json(noj);
        check(missing.ok && missing.value.exec_s_by_epoch.empty() && missing.value.exec_sigma_by_epoch.empty(), "absent lists read as one shape");
        Json notarr = keep.to_json();
        notarr.set("exec_s_by_epoch", 3.0);
        check(prefix(Candidate::from_json(notarr).code, "CANDIDATE_JSON"), "a non-array list is refused");
    }
    // the enumeration on a schedule: entry 0 is the current epoch, every step priced on its own shape, each shape once
    {
        FittedCostModel cm = historical_prior_model();
        Env env;
        env.bandwidth_bps = 100e6;
        env.one_way_latency_s = 0.01;
        Shapes sh = small_shapes();
        sh.n_rows = 8;    // the prefill of an 8-token prompt
        sh.kv_len = 8;
        SessionState est;
        est.incumbent = make_plan("generic", "generic", 2, 2);
        est.remaining_epochs = 4;
        est.kv_bucket = 32;
        est.model_version = "gpt2@rev1";
        est.contract_hash = default_contract().contract_hash();
        est.session_id = "s-sched";
        const std::vector<Shapes> sched = horizon_schedule(sh, 0, 4, 8, 32);
        check(sched.size() == 4 && shape_key(sched[0]) == shape_key(sh) && sched[1].n_rows == 1 && sched[1].kv_len == 32 && sched[3].kv_len == 32 &&
                  sched[2].n_layers == sh.n_layers && sched[2].vocab == sh.vocab,
              "horizon_schedule: the current epoch, then one decode row per later epoch over the bucketed KV length");
        const std::vector<Shapes> fine = horizon_schedule(sh, 1, 5, 8, 4);
        check(fine.size() == 4 && fine[1].kv_len == 12 && fine[2].kv_len == 12 && fine[3].kv_len == 12, "horizon_schedule with a bucket of 4: 10, 11, 12 keys round up to 12");
        check(horizon_schedule(sh, 3, 4, 8, 32).size() == 1 && horizon_schedule(sh, 5, 4, 8, 32).size() == 1, "the last epoch schedules itself alone");
        Expected<std::vector<Candidate>> cs = enumerate_candidates(est, Catalog::full(), cm, env, sh, default_contract(), sched);
        check(cs.ok && cs.value.size() == 24, "enumerate with a schedule: " + cs.code);
        Expected<std::vector<Candidate>> plain = enumerate_candidates(est, Catalog::full(), cm, env, sh);
        check(plain.ok && plain.value.size() == 24, "enumerate without a schedule");
        for (std::size_t i = 0; i < cs.value.size(); ++i) {
            const Candidate& c = cs.value[i];
            check(c.exec_s_by_epoch.size() == 4 && c.exec_sigma_by_epoch.size() == 4, "one entry per scheduled step");
            check(c.exec_s_by_epoch[0] == c.exec_s_per_epoch && c.exec_sigma_by_epoch[0] == c.exec_sigma, "entry 0 is the current epoch");
            check(c.exec_s_per_epoch == plain.value[i].exec_s_per_epoch && c.exec_sigma == plain.value[i].exec_sigma && c.c_install_s == plain.value[i].c_install_s &&
                      c.material_bytes_per_epoch == plain.value[i].material_bytes_per_epoch && c.feasible == plain.value[i].feasible,
                  "the schedule changes nothing else about the candidate");
            check(c.exec_s_by_epoch[1] == c.exec_s_by_epoch[2] && c.exec_s_by_epoch[2] == c.exec_s_by_epoch[3] && c.exec_sigma_by_epoch[1] == c.exec_sigma_by_epoch[3],
                  "one shape priced once serves every step of that shape");
            Expected<std::pair<double, double>> step = exec_cost(c.plan, sched[1], cm, env);
            Expected<Candidate> alone = evaluate_plan(c.plan, est, cm, env, sched[1]);
            check(step.ok && alone.ok && step.value.first == c.exec_s_by_epoch[1] && step.value.second == c.exec_sigma_by_epoch[1] &&
                      alone.value.exec_s_per_epoch == c.exec_s_by_epoch[1] && alone.value.exec_sigma == c.exec_sigma_by_epoch[1],
                  "a later step is priced exactly as an epoch of that shape");
            check(c.exec_s_by_epoch[1] < c.exec_s_by_epoch[0], "a decode row is cheaper than the prefill");
            check(plain.value[i].exec_s_by_epoch.empty() && plain.value[i].exec_sigma_by_epoch.empty(), "no schedule, no lists");
        }
        std::vector<Shapes> wrong = sched;
        wrong.erase(wrong.begin());
        check(prefix(enumerate_candidates(est, Catalog::full(), cm, env, sh, default_contract(), wrong).code, "SCHEDULE_FIRST_STEP") &&
                  prefix(evaluate_plan(est.incumbent, est, cm, env, sh, default_contract(), wrong).code, "SCHEDULE_FIRST_STEP"),
              "a schedule that does not start with the current epoch is refused");
        // the per-step controller on the enumerated candidates: Qhat(keep) is the per-step sum
        ControllerConfig cfg;
        cfg.stationary_survival = true;
        StateAwareController ctl(cfg);
        Expected<Decision> d = ctl.decide(est, cs.value, 0);
        check(d.ok && near(d.value.q_keep, horizon_exec(cs.value[0], 4) + cs.value[0].c_install_s), "Qhat(keep) of the enumerated keep is its per-step horizon plus install");
        check(near(horizon_exec(cs.value[0], 4), cs.value[0].exec_s_by_epoch[0] + 3.0 * cs.value[0].exec_s_by_epoch[1]), "horizon_exec sums the four scheduled steps");
    }
    std::printf("  controller schedule ok\n");
}

// ---------------------------------------------------------------------------
// Controller vectors (cross-language)
// ---------------------------------------------------------------------------
// count instances priced on one shape, then `scheduled` instances whose candidates carry per-step
// costs (lists as long as the horizon, two shorter or two longer); every scheduled instance also
// records the decision one-shape pricing of the same candidates takes (single_shape).
static Json random_vectors(int count, int scheduled) {
    SplitMix64 rng(777);
    const std::string embs[2] = {"wmask", "generic"};
    const int aks[3] = {2, 4, 8};
    const int mks[2] = {2, 4};
    // Every draw below goes into a named local, one statement each, so the stream of the generator is consumed in the
    // same order under every compiler (the order in which call arguments are evaluated is unspecified).
    auto random_plan = [&]() {
        const std::string& embedding = embs[rng.below(2)];
        const std::string& linear = embs[rng.below(2)];
        const int argmax_k = aks[rng.below(3)];
        const int max_k = mks[rng.below(2)];
        return make_plan(embedding, linear, argmax_k, max_k);
    };
    Json instances = Json::array();
    for (int id = 0; id < count; ++id) {
        ControllerConfig cfg;
        const int preset = static_cast<int>(rng.below(4));
        if (preset == 1) { cfg.stationary_survival = true; cfg.use_uncertainty = false; cfg.hysteresis_s = 0.0; cfg.min_residency = 0; }
        if (preset == 2) { cfg.z = 2.0; cfg.hysteresis_s = 1.0; }
        if (preset == 3) { cfg.ignore_install = true; cfg.stationary_survival = true; }
        StateAwareController ctl(cfg);
        Json telemetry = Json::array();
        const int n_tel = static_cast<int>(rng.below(7));
        double bw = 50e6 + 100e6 * rng.uniform01();
        double rtt = 0.005 + 0.04 * rng.uniform01();
        for (int i = 0; i < n_tel; ++i) {
            if (rng.uniform01() < 0.3) bw *= (rng.uniform01() < 0.5 ? 0.5 : 2.0);
            TelemetrySample s;
            s.seq = i;
            s.t = i;
            s.bandwidth_bps = bw;
            s.rtt_s = rtt;
            s.epoch = i;
            ctl.observe(s);
            telemetry.push_back(s.to_json());
        }
        SessionState st;
        st.incumbent = random_plan();
        st.remaining_epochs = 3 + static_cast<int>(rng.below(10));
        st.kv_bucket = 32;
        st.session_id = "vec" + std::to_string(id);
        st.model_version = "gpt2@test";
        st.contract_hash = default_contract().contract_hash();
        std::vector<Candidate> cands;
        {
            const double exec_s = 1.0 + 9.0 * rng.uniform01();
            const double install_s = rng.uniform01() < 0.3 ? 5.0 * rng.uniform01() : 0.0;
            const double sigma_s = 0.5 * rng.uniform01();
            cands.push_back(simple_candidate(st.incumbent, exec_s, install_s, "lm_head", sigma_s));
        }
        const int extra = 2 + static_cast<int>(rng.below(3));
        for (int k = 0; k < extra; ++k) {
            Json plan = random_plan();
            bool dup = false;
            for (const Candidate& c : cands) dup = dup || same_plan(c.plan, plan);
            if (dup) continue;
            const double exec_s = 1.0 + 9.0 * rng.uniform01();
            const double install_s = 30.0 * rng.uniform01();
            const double sigma_s = 0.5 * rng.uniform01();
            const double install_sigma_s = 0.2 * rng.uniform01();
            Candidate c = simple_candidate(plan, exec_s, install_s, "obj" + std::to_string(k), sigma_s, install_sigma_s);
            if (rng.uniform01() < 0.15) { c.feasible = false; c.reason = "INSUFFICIENT_MATERIAL"; }
            if (rng.uniform01() < 0.2) c.c_wait_s = 2.0 * rng.uniform01();
            cands.push_back(c);
        }
        const int epoch = static_cast<int>(rng.below(5));
        std::vector<double> surv = ctl.survival(st.remaining_epochs);
        Expected<Decision> d = ctl.decide(st, cands, epoch);
        check(d.ok, "vector decision");
        Json inst = Json::object();
        inst.set("id", id);
        inst.set("config", cfg.to_json());
        inst.set("telemetry", telemetry);
        inst.set("state", st.to_json());
        Json cj = Json::array();
        for (const Candidate& c : cands) cj.push_back(c.to_json());
        inst.set("candidates", cj);
        inst.set("epoch", epoch);
        Json sj = Json::array();
        for (double v : surv) sj.push_back(v);
        inst.set("survival", sj);
        Json dec = Json::object();
        dec.set("chosen", d.value.chosen);
        dec.set("switched", d.value.switched);
        dec.set("q_keep", d.value.q_keep);
        dec.set("q_best", d.value.q_best);
        dec.set("lower_bound_gain", d.value.lower_bound_gain);
        dec.set("u_diff", d.value.u_diff);
        dec.set("reason", d.value.reason);
        inst.set("decision", dec);
        instances.push_back(inst);
    }
    // scheduled instances: a dear first step (the prefill), cheaper later steps, a longer KV bucket near the end
    int n_differs = 0;
    for (int k = 0; k < scheduled; ++k) {
        const int id = count + k;
        ControllerConfig cfg;
        const int preset = static_cast<int>(rng.below(4));
        if (preset == 1) { cfg.stationary_survival = true; cfg.use_uncertainty = false; cfg.hysteresis_s = 0.0; cfg.min_residency = 0; }
        if (preset == 2) { cfg.epoch_corr = 0.5; cfg.z = 0.5; }
        if (preset == 3) { cfg.ignore_install = true; cfg.epoch_corr = 1.0; }
        StateAwareController ctl(cfg);
        Json telemetry = Json::array();
        const int n_tel = static_cast<int>(rng.below(6));
        double bw = 50e6 + 900e6 * rng.uniform01();
        double rtt = 0.0005 + 0.02 * rng.uniform01();
        for (int i = 0; i < n_tel; ++i) {
            if (rng.uniform01() < 0.3) bw *= (rng.uniform01() < 0.5 ? 0.5 : 2.0);
            TelemetrySample s;
            s.seq = i;
            s.t = i;
            s.bandwidth_bps = bw;
            s.rtt_s = rtt;
            s.epoch = i;
            ctl.observe(s);
            telemetry.push_back(s.to_json());
        }
        SessionState st;
        st.incumbent = random_plan();
        st.remaining_epochs = 2 + static_cast<int>(rng.below(8));
        st.kv_bucket = 32;
        st.session_id = "vec" + std::to_string(id);
        st.model_version = "gpt2@test";
        st.contract_hash = default_contract().contract_hash();
        const int H = st.remaining_epochs;
        const int n_list = (k % 4 < 2) ? H : (k % 4 == 2 ? std::max(1, H - 2) : H + 2);
        auto per_step = [&](Candidate& c) {
            const double first = 4.0 + 16.0 * rng.uniform01();
            const double later = 1.0 + 9.0 * rng.uniform01();
            const double s_first = 0.6 * rng.uniform01();
            const double s_later = 0.3 * rng.uniform01();
            const double tail = later * (1.0 + 0.2 * rng.uniform01());
            const int cut = std::max(1, H - 2);
            c.exec_s_per_epoch = first;
            c.exec_sigma = s_first;
            c.exec_s_by_epoch.push_back(first);
            c.exec_sigma_by_epoch.push_back(s_first);
            for (int j = 1; j < n_list; ++j) {
                c.exec_s_by_epoch.push_back(j < cut ? later : tail);
                c.exec_sigma_by_epoch.push_back(s_later);
            }
        };
        std::vector<Candidate> cands;
        {
            const double install_s = rng.uniform01() < 0.3 ? 5.0 * rng.uniform01() : 0.0;
            Candidate keep = simple_candidate(st.incumbent, 0.0, install_s, "lm_head");
            per_step(keep);
            cands.push_back(keep);
        }
        const int extra = 2 + static_cast<int>(rng.below(3));
        for (int x = 0; x < extra; ++x) {
            Json plan = random_plan();
            bool dup = false;
            for (const Candidate& c : cands) dup = dup || same_plan(c.plan, plan);
            if (dup) continue;
            const double install_s = 20.0 * rng.uniform01();
            const double install_sigma_s = 0.2 * rng.uniform01();
            Candidate c = simple_candidate(plan, 0.0, install_s, "obj" + std::to_string(x), 0.0, install_sigma_s);
            per_step(c);
            if (rng.uniform01() < 0.15) { c.feasible = false; c.reason = "INSUFFICIENT_MATERIAL"; }
            if (rng.uniform01() < 0.2) c.c_wait_s = 2.0 * rng.uniform01();
            cands.push_back(c);
        }
        const int epoch = static_cast<int>(rng.below(5));
        std::vector<double> surv = ctl.survival(H);
        Expected<Decision> d = ctl.decide(st, cands, epoch);
        check(d.ok, "scheduled vector decision");
        // one-shape pricing of the same candidates (the lists dropped), on a fresh controller with the same survival
        std::vector<Candidate> flat = cands;
        for (Candidate& c : flat) { c.exec_s_by_epoch.clear(); c.exec_sigma_by_epoch.clear(); }
        StateAwareController one(cfg);
        Expected<Decision> od = one.decide_with_survival(st, flat, epoch, surv);
        check(od.ok, "scheduled vector one-shape decision");
        const bool differs = !same_plan(od.value.chosen, d.value.chosen) || od.value.switched != d.value.switched;
        if (differs) ++n_differs;
        Json inst = Json::object();
        inst.set("id", id);
        inst.set("config", cfg.to_json());
        inst.set("telemetry", telemetry);
        inst.set("state", st.to_json());
        Json cj = Json::array();
        for (const Candidate& c : cands) cj.push_back(c.to_json());
        inst.set("candidates", cj);
        inst.set("epoch", epoch);
        Json sj = Json::array();
        for (double v : surv) sj.push_back(v);
        inst.set("survival", sj);
        Json dec = Json::object();
        dec.set("chosen", d.value.chosen);
        dec.set("switched", d.value.switched);
        dec.set("q_keep", d.value.q_keep);
        dec.set("q_best", d.value.q_best);
        dec.set("lower_bound_gain", d.value.lower_bound_gain);
        dec.set("u_diff", d.value.u_diff);
        dec.set("reason", d.value.reason);
        inst.set("decision", dec);
        Json single = Json::object();
        single.set("chosen", od.value.chosen);
        single.set("switched", od.value.switched);
        single.set("q_keep", od.value.q_keep);
        single.set("differs", differs);
        inst.set("single_shape", single);
        instances.push_back(inst);
    }
    Json root = Json::object();
    root.set("format", "opencrab2-controller-vectors-v1");
    root.set("generator", "cpp/tests/test_decision.cpp --vectors (SplitMix64 seed 777)");
    root.set("semantics",
             "For each instance feed the telemetry samples to StateAwareController(config).observe in order, then call decide(state, candidates, epoch). "
             "Qhat(p) = c_install (unless ignore_install) + sum_j s_j exec_j + c_wait_s over H = remaining_epochs with the survival vector given "
             "(s_j = (1 - 1/D)^j, D = (dwell_prior + sum completed dwells + current age) / (1 + completed), or all ones when stationary_survival); "
             "exec_j = exec_s_by_epoch[min(j, n - 1)] when the candidate carries that list (n its length), else exec_s_per_epoch. "
             "U_diff = sqrt(var_keep + var_p); with a_j = s_j sigma_j (sigma_j = exec_sigma_by_epoch[min(j, n - 1)], else exec_sigma) "
             "var = sum a_j^2 + epoch_corr ((sum a_j)^2 - sum a_j^2) + c_install_sigma^2, "
             "lower_bound_gain = Qhat(keep) - Qhat(p) - z U_diff. "
             "Switch iff the best feasible challenger has lower_bound_gain > hysteresis_s and > 0 (min_residency never binds on a fresh controller). "
             "A scheduled instance also records single_shape: the decision on the same candidates with the lists dropped.");
    root.set("instances_scheduled", scheduled);
    root.set("instances_scheduled_single_shape_differs", n_differs);
    root.set("instances", instances);
    return root;
}

static void check_vectors(const Json& root) {
    check(root.get("format").as_string() == "opencrab2-controller-vectors-v1", "vector file format");
    const Json& instances = root.get("instances");
    check(instances.size() >= 20, "at least 20 vector instances");
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const Json& inst = instances[i];
        Expected<ControllerConfig> cfg = ControllerConfig::from_json(inst.get("config"));
        check(cfg.ok, "vector config");
        StateAwareController ctl(cfg.value);
        for (const Json& tj : inst.get("telemetry").as_array()) {
            Expected<TelemetrySample> s = TelemetrySample::from_json(tj);
            check(s.ok, "vector telemetry");
            ctl.observe(s.value);
        }
        Expected<SessionState> st = SessionState::from_json(inst.get("state"));
        check(st.ok, "vector state");
        std::vector<Candidate> cands;
        for (const Json& cj : inst.get("candidates").as_array()) {
            Expected<Candidate> c = Candidate::from_json(cj);
            check(c.ok, "vector candidate");
            cands.push_back(c.value);
        }
        std::vector<double> surv = ctl.survival(st.value.remaining_epochs);
        const Json& sj = inst.get("survival");
        bool surv_ok = sj.size() == surv.size();
        for (std::size_t k = 0; surv_ok && k < surv.size(); ++k) surv_ok = near(surv[k], sj[k].as_double(), 1e-12);
        check(surv_ok, "vector survival reproduced");
        Expected<Decision> d = ctl.decide(st.value, cands, static_cast<int>(inst.get("epoch").as_int()));
        check(d.ok, "vector decide");
        const Json& dec = inst.get("decision");
        check(d.value.chosen == dec.get("chosen") && d.value.switched == dec.get("switched").as_bool(), "vector chosen/switched match");
        check(near(d.value.q_keep, dec.get("q_keep").as_double(), 1e-9) && near(d.value.q_best, dec.get("q_best").as_double(), 1e-9) &&
                  near(d.value.lower_bound_gain, dec.get("lower_bound_gain").as_double(), 1e-9) && near(d.value.u_diff, dec.get("u_diff").as_double(), 1e-9),
              "vector numeric fields match");
        check(d.value.reason == dec.get("reason").as_string(), "vector reason matches");
        if (inst.has("single_shape")) {
            // the lists decide: dropping them reproduces the recorded one-shape decision
            std::vector<Candidate> flat = cands;
            for (Candidate& c : flat) { c.exec_s_by_epoch.clear(); c.exec_sigma_by_epoch.clear(); }
            StateAwareController one(cfg.value);
            Expected<Decision> od = one.decide_with_survival(st.value, flat, static_cast<int>(inst.get("epoch").as_int()), surv);
            const Json& single = inst.get("single_shape");
            check(od.ok && od.value.chosen == single.get("chosen") && od.value.switched == single.get("switched").as_bool() &&
                      near(od.value.q_keep, single.get("q_keep").as_double(), 1e-9),
                  "vector one-shape decision matches");
            check(single.get("differs").as_bool() == (!(od.value.chosen == d.value.chosen) || od.value.switched != d.value.switched), "vector differs flag matches");
        }
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------
static Realization mk_real(const std::string& name, const std::string& in, const std::string& out, double cost, int consumed = 0,
                           std::vector<std::string> objs = {}, int reset = -1) {
    Realization r;
    r.name = name;
    r.in_domain = in;
    r.out_domain = out;
    r.fixed_cost_s = cost;
    r.level_consumed = consumed;
    r.install_objects = std::move(objs);
    r.level_reset_to = reset;
    return r;
}

static void test_planner_typed_state() {
    PlannerCatalog cat;
    cat.realizations["op1"] = {mk_real("cheap", "H_BFV", "H_BFV", 1.0, 2), mk_real("careful", "H_BFV", "H_BFV", 2.0, 1)};
    cat.realizations["op2"] = {mk_real("deep", "H_BFV", "H_BFV", 1.0, 2)};
    std::vector<ChainOp> chain = {{"l1", "op1", 1, 1, 1}, {"l2", "op2", 1, 1, 1}};
    PlannerState st;
    st.start_domain = "H_BFV";
    st.start_level = 3;
    Env env;
    Expected<PlanResult> typed = ExactAssign(true).plan(chain, cat, env, nullptr, st);
    check(typed.ok, "typed DP finds a plan: " + typed.code);
    check(typed.value.assignment == std::vector<std::string>({"careful", "deep"}) && near(typed.value.cost_s, 3.0) && typed.value.end_level == 0, "typed plan careful+deep = 3");
    Expected<PlanResult> domain_only = ExactAssign(false).plan(chain, cat, env, nullptr, st);
    check(!domain_only.ok && prefix(domain_only.code, "NO_FEASIBLE_PLAN"), "domain-only DP discards the level-feasible prefix");
    Expected<PlanResult> bf = brute_force(chain, cat, env, nullptr, st);
    check(bf.ok && near(bf.value.cost_s, 3.0), "typed state: brute force agrees");
    Expected<PlanResult> ilp = ILPAdapter().plan(chain, cat, env, nullptr, st);
    check(ilp.ok && near(ilp.value.cost_s, 3.0) && ilp.value.assignment == typed.value.assignment, "typed state: ILP agrees");
    Expected<PlanResult> greedy = GreedyPerOperator().plan(chain, cat, env, nullptr, st);
    check(!greedy.ok, "typed state: greedy takes cheap and dead-ends");
    // JSON round trip of the catalog
    Expected<PlannerCatalog> crt = PlannerCatalog::from_json(cat.to_json());
    check(crt.ok && crt.value.to_json() == cat.to_json(), "planner catalog json round trip");
    std::printf("  planner typed state ok\n");
}

static void test_planner_random() {
    SplitMix64 rng(4242);
    const std::vector<std::string> domains = {"A64", "H_BFV", "B"};
    const std::vector<std::string> objects = {"o1", "o2", "o3"};
    int successes = 0, failures = 0;
    for (int trial = 0; trial < 60; ++trial) {
        PlannerCatalog cat;
        for (const auto& o : objects) cat.install_cost_s[o] = 5.0 * rng.uniform01();
        for (int k = 0; k < 3; ++k) {
            std::vector<Realization> rs;
            const int count = 2 + static_cast<int>(rng.below(2));
            for (int r = 0; r < count; ++r) {
                const std::string in = domains[rng.below(3)];
                const std::string out = domains[rng.below(3)];
                Realization re = mk_real("k" + std::to_string(k) + "r" + std::to_string(r), in, out, 0.1 + 9.9 * rng.uniform01());
                if (in == "H_BFV") re.level_consumed = static_cast<int>(rng.below(3));
                if (out == "H_BFV" && rng.uniform01() < 0.2) re.level_reset_to = 4;
                if (rng.uniform01() < 0.4) re.install_objects.push_back(objects[rng.below(3)]);
                rs.push_back(re);
            }
            cat.realizations["k" + std::to_string(k)] = rs;
        }
        for (const auto& a : domains)
            for (const auto& b : domains) {
                if (a == b || rng.uniform01() < 0.3) continue;
                Conversion c;
                c.from_domain = a;
                c.to_domain = b;
                c.fixed_cost_s = 0.5 + 3.0 * rng.uniform01();
                if (b == "H_BFV") c.level_reset_to = 4;
                cat.conversions.push_back(c);
            }
        const int n = 1 + static_cast<int>(rng.below(6));
        std::vector<ChainOp> chain;
        for (int i = 0; i < n; ++i) chain.push_back({"op" + std::to_string(i), "k" + std::to_string(rng.below(3)), 1, 1, 1});
        PlannerState st;
        st.start_domain = domains[rng.below(3)];
        st.start_level = 4;
        if (rng.uniform01() < 0.3) st.required_end_domain = "A64";
        Env env;
        Expected<PlanResult> ex = ExactAssign(true).plan(chain, cat, env, nullptr, st);
        Expected<PlanResult> il = ILPAdapter().plan(chain, cat, env, nullptr, st);
        Expected<PlanResult> bf = brute_force(chain, cat, env, nullptr, st);
        check(ex.ok == bf.ok && il.ok == bf.ok, "planners agree on feasibility (trial " + std::to_string(trial) + ")");
        if (!bf.ok) { ++failures; continue; }
        ++successes;
        check(near(ex.value.cost_s, bf.value.cost_s, 1e-9) && near(il.value.cost_s, bf.value.cost_s, 1e-9), "ExactAssign == ILPAdapter == brute_force (trial " + std::to_string(trial) + ")");
        Expected<PlanResult> re = evaluate_assignment(chain, cat, env, nullptr, st, ex.value.assignment);
        Expected<PlanResult> ri = evaluate_assignment(chain, cat, env, nullptr, st, il.value.assignment);
        check(re.ok && ri.ok && near(re.value.cost_s, ex.value.cost_s) && near(ri.value.cost_s, il.value.cost_s), "assignments re-price to the same cost");
        Expected<PlanResult> gr = GreedyPerOperator().plan(chain, cat, env, nullptr, st);
        if (gr.ok) check(gr.value.cost_s >= bf.value.cost_s - 1e-9, "greedy never beats the exact optimum");
    }
    check(successes >= 20 && failures >= 1, "random chains cover feasible and infeasible instances (" + std::to_string(successes) + "/" + std::to_string(failures) + ")");
    // component-priced realizations through the cost model
    FittedCostModel cm = synthetic_model();
    PlannerCatalog cat;
    Realization w = mk_real("linear.wmask", "A64", "A64", 0.0, 0, {"h0.fc"});
    w.component = "linear_wmask";
    Realization g = mk_real("linear.generic", "A64", "A64", 0.0);
    g.component = "linear_generic";
    cat.realizations["linear"] = {w, g};
    cat.install_cost_s["h0.fc"] = 100.0;
    std::vector<ChainOp> chain = {{"fc", "linear", 4, 768, 3072}};
    PlannerState st;
    Env env;
    env.bandwidth_bps = 100e6;
    env.one_way_latency_s = 0.01;
    Expected<PlanResult> ex = ExactAssign(true).plan(chain, cat, env, &cm, st);
    check(ex.ok && ex.value.assignment[0] == "linear.generic", "install cost of 100 s makes generic the cheaper single call");
    cat.install_cost_s["h0.fc"] = 0.0;
    Expected<PlanResult> ex2 = ExactAssign(true).plan(chain, cat, env, &cm, st);
    check(ex2.ok && ex2.value.assignment[0] == "linear.wmask", "with the object installed for free wmask wins");
    check(prefix(ExactAssign(true).plan(chain, cat, env, nullptr, st).code, "PLANNER_NO_COSTMODEL"), "component pricing needs a cost model");
    Realization bad = mk_real("bad", "A64", "A64", 1.0, 0, {"unknown"});
    cat.realizations["linear"] = {bad};
    check(prefix(ExactAssign(true).plan(chain, cat, env, &cm, st).code, "UNKNOWN_INSTALL_OBJECT"), "unknown install object fails closed");
    std::printf("  planner random ok\n");
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
static void test_statistics() {
    Expected<double> z = zero_failure_upper_bound(10);
    check(z.ok && near(z.value, 1.0 - std::pow(0.05, 0.1), 1e-12), "zero failure bound n = 10");
    check(prefix(zero_failure_upper_bound(0).code, "STATISTICS_PARAM"), "n = 0 refused");
    check(prefix(zero_failure_upper_bound(5, 1.5).code, "STATISTICS_PARAM"), "alpha out of range refused");
    std::vector<double> diffs;
    for (int i = 1; i <= 10; ++i) diffs.push_back(i);
    Expected<BootstrapCI> ci = paired_bootstrap_ci(diffs, 2000, 0.05, 3);
    check(ci.ok && near(ci.value.point, 5.5) && ci.value.lo <= 5.5 && 5.5 <= ci.value.hi && ci.value.lo < ci.value.hi, "bootstrap CI brackets the mean");
    Expected<BootstrapCI> ci2 = paired_bootstrap_ci(diffs, 2000, 0.05, 3);
    check(ci2.ok && ci2.value.lo == ci.value.lo && ci2.value.hi == ci.value.hi, "identical seeds give identical intervals");
    // continuous-valued sample: integer diffs give means on a 0.1 grid where two seeds often tie
    std::vector<double> cont;
    for (int i = 0; i < 20; ++i) cont.push_back(std::sqrt(static_cast<double>(i + 1)));
    Expected<BootstrapCI> c3 = paired_bootstrap_ci(cont, 500, 0.05, 3);
    Expected<BootstrapCI> c4 = paired_bootstrap_ci(cont, 500, 0.05, 4);
    check(c3.ok && c4.ok && (c3.value.lo != c4.value.lo || c3.value.hi != c4.value.hi), "different seeds give different intervals");
    check(SplitMix64(3).next() != SplitMix64(4).next(), "seed changes the stream");
    check(prefix(paired_bootstrap_ci({}, 100).code, "STATISTICS_EMPTY"), "empty diffs refused");
    check(prefix(paired_bootstrap_ci({1.0, std::nan("")}, 100).code, "STATISTICS_NONFINITE"), "NaN refused");
    check(prefix(paired_bootstrap_ci(diffs, 0).code, "STATISTICS_PARAM"), "n_boot 0 refused");
    std::vector<double> a = {2, 4, 6, 8}, b = {1, 2, 3, 4};
    Expected<BootstrapCI> r = ratio_ci(a, b, 1000, 0.05, 1);
    check(r.ok && near(r.value.point, 2.0) && r.value.lo <= 2.0 + 1e-12 && r.value.hi >= 2.0 - 1e-12, "ratio CI brackets the point ratio");
    check(prefix(ratio_ci(a, {0.0, 0.0}, 100).code, "STATISTICS_RATIO_DENOMINATOR"), "zero denominator refused");
    Expected<Summary> s = summarize({1, 2, 3, 4, 5});
    check(s.ok && s.value.n == 5 && near(s.value.mean, 3.0) && near(s.value.median, 3.0) && near(s.value.p95, 4.8) && s.value.min == 1 && s.value.max == 5 &&
              near(s.value.std, std::sqrt(2.5)),
          "summary of 1..5");
    check(summarize({7.0}).value.std == 0.0, "single value std 0");
    check(near(percentile_linear({4, 1, 3, 2}, 50).value, 2.5), "numpy linear percentile");
    Json sj = s.value.to_json();
    check(sj.get("n").as_int() == 5 && sj.has("p95"), "summary json");
    std::printf("  statistics ok\n");
}

int main(int argc, char** argv) {
    std::string write_path, check_path, formulas_out;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vectors") == 0 && i + 1 < argc) write_path = argv[++i];
        else if (std::strcmp(argv[i], "--check-vectors") == 0 && i + 1 < argc) check_path = argv[++i];
        else if (std::strcmp(argv[i], "--formulas-out") == 0 && i + 1 < argc) formulas_out = argv[++i];
    }
    test_json();
    test_contract();
    test_costmodel();
    test_formulas_and_fitted_model();
    if (!formulas_out.empty()) {
        // the C++ side of the shared document, with the shared constants of the tree when present
        Expected<FormulasReport> lf = load_formulas("");
        check(lf.ok, "load formulas for the document: " + lf.code);
        Status st = Json::save_file(formulas_out, formulas_document(), true);
        check(st.ok, "write formulas document: " + st.code);
        std::printf("  wrote %s\n", formulas_out.c_str());
        reset_formula_constants();
    }
    test_candidates();
    test_controller_install_example();
    test_controller_stationary_cold_start();
    test_controller_brute_force();
    test_policies();
    test_controller_schedule();
    Json vectors = random_vectors(20, 12);
    check_vectors(vectors);
    check(vectors.get("instances").size() == 32 && vectors.get("instances_scheduled").as_int() == 12 &&
              vectors.get("instances_scheduled_single_shape_differs").as_int() > 0,
          "the vector set carries scheduled instances on which one-shape pricing differs");
    Expected<Json> reparsed = Json::parse(vectors.dump_pretty(2));
    check(reparsed.ok && reparsed.value == vectors, "vectors survive a serialization round trip");
    check_vectors(reparsed.value);
    if (!write_path.empty()) {
        Status st = Json::save_file(write_path, vectors, true);
        check(st.ok, "write vectors: " + st.code);
        std::printf("  wrote %s\n", write_path.c_str());
    }
    if (!check_path.empty()) {
        Expected<Json> loaded = Json::load_file(check_path);
        check(loaded.ok, "load vectors: " + loaded.code);
        check_vectors(loaded.value);
        std::printf("  checked %s\n", check_path.c_str());
    }
    test_planner_typed_state();
    test_planner_random();
    test_statistics();
    std::printf("PASS %d checks\n", g_checks);
    return 0;
}
