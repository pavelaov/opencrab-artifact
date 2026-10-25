// OpenCrab v2 C++17 mirror: secure toy-BFV bridge tests (paper, Section 5.2).
//
// Two party threads over a loopback socket pair, an in-process dealer link, real Party gadgets.
//   1. Karatsuba split / recombine and the shared negacyclic product (both convolution modes)
//      against lowlevel::negacyclic_mul.
//   2. End to end: share -> a2h_toy -> eval_toy(W) -> h2a_ref -> reconstruct == W x mod t at
//      (N = 128, d_in = d_out = 8) and (N = 512, d_in = 64, d_out = 8), 200 trials each, with the
//      noise bound respected, in the default mode (Karatsuba leaves, Python parity); the
//      outer-triple mode runs 200 trials at N = 128 and 20 at N = 512.
//   3. Direction of every frame: P1 sends the ciphertext and never receives one. Frames and bytes
//      are counted per phase and per direction, they are identical in every trial (sizes depend
//      on public shapes only), and no frame received by P1 has the ciphertext shape or contains
//      a component of the fresh or the evaluated ciphertext; no frame received by P0 contains s.
//   4. Role violations and wrong truncation material fail closed (BRIDGE_PARTY, BRIDGE_SHAPE,
//      MATERIAL_PARAMS), the former before any message.
//
// What is NOT shown here: security of the toy parameters (there is none, q = 2^64 and N <= 512),
// malicious parties, or anything about real-parameter BFV. The content checks on frames show the
// absence of the ciphertext in the clear; that the masked openings are independent of it rests
// on the Beaver and truncation-gate arguments, not on this test.
//
// Prints "PASS <n> checks" and exits 0, or prints the failing check and exits 1.
#include <opencrab2/highlevel/bridge.hpp>
#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/contract.hpp>
#include <opencrab2/highlevel/dealer.hpp>
#include <opencrab2/highlevel/ledger.hpp>
#include <opencrab2/highlevel/party.hpp>
#include <opencrab2/lowlevel/toy_bfv.hpp>
#include <opencrab2/lowlevel/trunc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace opencrab2;
using namespace opencrab2::highlevel;
using lowlevel::Ciphertext;
using lowlevel::Matrix;
using lowlevel::Poly;
using lowlevel::ToyBFV;
using lowlevel::ToyParams;
using u64 = uint64_t;

namespace {

long g_checks = 0;

[[noreturn]] void die(const std::string& what, int line, const std::string& detail) {
    std::printf("FAIL %s (line %d)%s%s\n", what.c_str(), line, detail.empty() ? "" : ": ", detail.c_str());
    std::fflush(stdout);
    std::exit(1);
}

#define CHECK(cond, msg)                         \
    do {                                         \
        if (!(cond)) die(msg, __LINE__, #cond);  \
        ++g_checks;                              \
    } while (0)

#define CHECK_OK(expr, msg)                              \
    do {                                                 \
        const auto& oc2_r = (expr);                      \
        if (!oc2_r.ok) die(msg, __LINE__, oc2_r.code);   \
        ++g_checks;                                      \
    } while (0)

#define CHECK_CODE(expr, prefix, msg)                                                         \
    do {                                                                                      \
        const auto& oc2_r = (expr);                                                           \
        if (oc2_r.ok) die(msg, __LINE__, "expected failure " prefix ", got ok");              \
        if (oc2_r.code.rfind(prefix, 0) != 0) die(msg, __LINE__, "expected " prefix ", got " + oc2_r.code); \
        ++g_checks;                                                                           \
    } while (0)

std::string temp_path(const std::string& name) {
    static int counter = 0;
    const char* base = std::getenv("TEMP");
    if (!base) base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::string(base) + "/oc2_bridge_" + name + "_" + std::to_string(static_cast<long long>(stamp)) + "_" +
           std::to_string(counter++) + ".jsonl";
}

// Records every frame a party sends and receives (rounds are accounted at this level).
class SpyChannel : public Channel {
public:
    explicit SpyChannel(Channel& inner) : Channel(inner.party), inner_(inner) {}
    std::vector<Value> sent;
    std::vector<Value> received;

    Expected<Unit> send(const Value& obj) override {
        sent.push_back(obj);
        auto r = inner_.send(obj);
        inner_.clear_pending();
        sync();
        if (r.ok) pending_send_ = true;
        return r;
    }
    Expected<Value> recv() override {
        auto r = inner_.recv();
        sync();
        if (r.ok) {
            received.push_back(r.value);
            if (pending_send_) {
                stats.rounds += 1;
                pending_send_ = false;
            }
        }
        return r;
    }
    Expected<Value> exchange(const Value& obj) override {
        sent.push_back(obj);
        stats.rounds += 1;
        pending_send_ = false;
        auto r = inner_.exchange(obj);
        sync();
        if (r.ok) received.push_back(r.value);
        return r;
    }
    void close() override { inner_.close(); }

private:
    void sync() {
        stats.bytes_sent = inner_.stats.bytes_sent;
        stats.bytes_recv = inner_.stats.bytes_recv;
        stats.messages_sent = inner_.stats.messages_sent;
        stats.messages_recv = inner_.stats.messages_recv;
    }
    Channel& inner_;
};

struct Rig {
    NumericContract contract = default_contract();
    DealerConfig dc;
    std::shared_ptr<Dealer> dealer;
    std::shared_ptr<InProcessDealerLink> link;
    std::unique_ptr<Channel> raw[2];
    std::unique_ptr<SpyChannel> ch[2];
    std::unique_ptr<MaterialLedger> ledger[2];
    std::unique_ptr<DealerClient> client[2];
    std::string paths[2];
    Bytes key[2];

    ~Rig() {
        for (int p = 0; p < 2; ++p) {
            client[p].reset();
            ledger[p].reset();
            if (!paths[p].empty()) std::remove(paths[p].c_str());
        }
    }
};

std::unique_ptr<Rig> make_rig(const std::string& tag, u64 dealer_seed) {
    auto rig = std::make_unique<Rig>();
    rig->dc.setup_id = "setup-" + tag;
    rig->dc.model_version = "model-" + tag;
    rig->dc.numeric_contract_hash = rig->contract.contract_hash();
    rig->key[0] = Bytes(32, 0x11);
    rig->key[1] = Bytes(32, 0x22);
    rig->dealer = std::make_shared<Dealer>(rig->dc, ll::Rng::from_seed(dealer_seed), rig->key[0], rig->key[1]);
    rig->link = std::make_shared<InProcessDealerLink>(rig->dealer);
    auto pair = loopback_pair();
    CHECK_OK(pair, "loopback pair");
    rig->raw[0] = std::move(pair.value.first);
    rig->raw[1] = std::move(pair.value.second);
    const Value binding = dict({{"session_id", "sess-" + tag}, {"setup_id", rig->dc.setup_id}, {"model_version", rig->dc.model_version},
                                {"numeric_contract_hash", rig->dc.numeric_contract_hash}, {"profile_id", rig->dc.profile_id}});
    for (int p = 0; p < 2; ++p) {
        rig->ch[p] = std::make_unique<SpyChannel>(*rig->raw[p]);
        rig->paths[p] = temp_path(tag + "_p" + std::to_string(p));
        rig->ledger[p] = std::make_unique<MaterialLedger>(rig->paths[p], binding);
        CHECK_OK(rig->ledger[p]->open(), "ledger open");
        rig->client[p] = std::make_unique<DealerClient>(p, *rig->ledger[p], rig->key[p], rig->dc.numeric_contract_hash);
        rig->link->attach(*rig->client[p]);
    }
    return rig;
}

using Body = std::function<Expected<Unit>(Party&, int)>;

// Runs the body for both parties on two threads; returns both results.
std::array<Expected<Unit>, 2> run_both(Rig& rig, const Body& body, std::vector<GadgetStat>* stats_out = nullptr) {
    std::array<Expected<Unit>, 2> out;
    auto run = [&](int p) {
        ReleasePolicy rel(1, {"token"}, 1);
        rig.client[p]->set_gate("bridge", 0);
        Party party(p, *rig.ch[p], *rig.client[p], rig.contract, rel, ll::Rng::from_seed(900 + static_cast<u64>(p)), rig.dc.model_version);
        out[p] = body(party, p);
        if (!out[p].ok) rig.ch[p]->close();   // unblock the peer (fail closed)
        if (stats_out) stats_out[p] = party.stats;
    };
    std::thread t0(run, 0);
    std::thread t1(run, 1);
    t0.join();
    t1.join();
    return out;
}

// Truncation material with L = 64 for `calls` h2a_ref calls, delivered to both parties.
void provision_h2a_trunc(Rig& rig, const ToyParams& params, size_t calls) {
    const auto bp = bridge::h2a_trunc_bundles(*rig.dealer, params, calls);
    CHECK_OK(bp, "h2a truncation bundles");
    CHECK_OK(rig.client[0]->ingest(bp.value.for0), "ingest trunc P0");
    CHECK_OK(rig.client[1]->ingest(bp.value.for1), "ingest trunc P1");
}

bool contains_run(const std::vector<u64>& hay, const std::vector<u64>& needle) {
    if (needle.empty() || hay.size() < needle.size()) return false;
    return std::search(hay.begin(), hay.end(), needle.begin(), needle.end()) != hay.end();
}

const size_t kFrameOverhead = 8 + 3 + 16;   // 8-byte length prefix, ndarray tag/dtype/ndim, two u64 dims

// ---------------------------------------------------------------------------------------------
// 1. Karatsuba split / recombine (plain)
// ---------------------------------------------------------------------------------------------
void test_karatsuba_plain() {
    lowlevel::RandomSource rng = lowlevel::deterministic_source(11);
    CHECK(bridge::karatsuba_leaf_count(1) == 1 && bridge::karatsuba_leaf_count(2) == 3 && bridge::karatsuba_leaf_count(128) == 2187 &&
              bridge::karatsuba_leaf_count(512) == 19683,
          "leaf counts 3^log2(N)");
    CHECK(bridge::karatsuba_leaf_count(0) == 0 && bridge::karatsuba_leaf_count(12) == 0, "leaf count of a non power of two");
    for (size_t N : {size_t(1), size_t(2), size_t(4), size_t(8), size_t(64), size_t(128), size_t(256), size_t(512)}) {
        for (int trial = 0; trial < (N <= 64 ? 20 : 4); ++trial) {
            Poly a(N), b(N);
            for (auto& v : a) v = rng();
            for (auto& v : b) v = rng();
            if (trial == 1) std::fill(b.begin(), b.end(), 0);
            if (trial == 2) {   // ternary operand, as a secret key
                for (auto& v : b) {
                    const u64 r = rng() % 3;
                    v = r == 2 ? ~u64(0) : r;
                }
            }
            const auto ops = bridge::karatsuba_leaf_operands(a, b);
            CHECK_OK(ops, "leaf operands");
            CHECK(ops.value.first.size() == bridge::karatsuba_leaf_count(N) && ops.value.second.size() == ops.value.first.size(), "leaf operand count");
            std::vector<u64> prod(ops.value.first.size());
            for (size_t i = 0; i < prod.size(); ++i) prod[i] = ops.value.first[i] * ops.value.second[i];
            const auto rec = bridge::karatsuba_recombine(prod, N);
            CHECK_OK(rec, "recombine");
            CHECK(rec.value == lowlevel::negacyclic_mul(a, b), "Karatsuba equals the schoolbook negacyclic product");
        }
    }
    // linearity: the split of a sum of shares is the sum of the splits (what the shared product uses)
    {
        const size_t N = 64;
        Poly a0(N), a1(N), z(N, 0);
        for (auto& v : a0) v = rng();
        for (auto& v : a1) v = rng();
        Poly a(N);
        for (size_t i = 0; i < N; ++i) a[i] = a0[i] + a1[i];
        const auto l0 = bridge::karatsuba_leaf_operands(a0, z).value.first, l1 = bridge::karatsuba_leaf_operands(a1, z).value.first,
                   l = bridge::karatsuba_leaf_operands(a, z).value.first;
        bool same = true;
        for (size_t i = 0; i < l.size(); ++i) same = same && (l0[i] + l1[i] == l[i]);
        CHECK(same, "leaf split is linear in the shares");
    }
    CHECK_CODE(bridge::karatsuba_leaf_operands(Poly(12), Poly(12)), "BRIDGE_SHAPE", "non power of two refused");
    CHECK_CODE(bridge::karatsuba_leaf_operands(Poly(8), Poly(16)), "BRIDGE_SHAPE", "operand length mismatch refused");
    CHECK_CODE(bridge::karatsuba_leaf_operands(Poly(), Poly()), "BRIDGE_SHAPE", "empty operand refused");
    CHECK_CODE(bridge::karatsuba_recombine(std::vector<u64>(10), 8), "BRIDGE_SHAPE", "wrong leaf count refused");
    CHECK_CODE(bridge::karatsuba_recombine(std::vector<u64>(27), 12), "BRIDGE_SHAPE", "non power of two refused in recombine");
}

// ---------------------------------------------------------------------------------------------
// 2. Shared negacyclic product through Party, both modes
// ---------------------------------------------------------------------------------------------
void test_shared_product() {
    auto rig = make_rig("prod", 21);
    lowlevel::RandomSource rng = lowlevel::deterministic_source(22);
    struct Case {
        size_t N;
        bridge::Convolution conv;
        int trials;
    };
    const Case cases[] = {{8, bridge::Convolution::KARATSUBA_LEAVES, 6},   {128, bridge::Convolution::KARATSUBA_LEAVES, 6},
                          {512, bridge::Convolution::KARATSUBA_LEAVES, 3}, {8, bridge::Convolution::OUTER_TRIPLE, 6},
                          {128, bridge::Convolution::OUTER_TRIPLE, 6},     {512, bridge::Convolution::OUTER_TRIPLE, 2}};
    for (const Case& c : cases) {
        for (int trial = 0; trial < c.trials; ++trial) {
            const size_t N = c.N;
            Poly a(N), b(N), share[2][2] = {{Poly(N), Poly(N)}, {Poly(N), Poly(N)}};   // share[party][operand]
            for (size_t i = 0; i < N; ++i) {
                a[i] = rng();
                b[i] = rng();
                share[0][0][i] = rng();
                share[1][0][i] = a[i] - share[0][0][i];
                share[0][1][i] = rng();
                share[1][1][i] = b[i] - share[0][1][i];
            }
            Poly out[2];
            u64 rounds0[2], bytes0[2], rounds1[2], bytes1[2];
            const auto r = run_both(*rig, [&](Party& party, int p) -> Expected<Unit> {
                rounds0[p] = party.channel().stats.rounds;
                bytes0[p] = party.channel().stats.bytes_sent;
                OC2_TRY(z, bridge::negacyclic_mul_shared(party, share[p][0], share[p][1], c.conv));
                rounds1[p] = party.channel().stats.rounds;
                bytes1[p] = party.channel().stats.bytes_sent;
                out[p] = std::move(z);
                return ok_unit();
            });
            CHECK_OK(r[0], "shared product P0");
            CHECK_OK(r[1], "shared product P1");
            Poly rec(N);
            for (size_t i = 0; i < N; ++i) rec[i] = out[0][i] + out[1][i];
            CHECK(rec == lowlevel::negacyclic_mul(a, b), "shared negacyclic product reconstructs");
            const size_t words = c.conv == bridge::Convolution::KARATSUBA_LEAVES ? 2 * bridge::karatsuba_leaf_count(N) : 2 * N;
            for (int p = 0; p < 2; ++p) {
                CHECK(rounds1[p] - rounds0[p] == 1, "one Beaver round");
                CHECK(bytes1[p] - bytes0[p] == 8 * words + kFrameOverhead, "Beaver opening bytes depend on N and the mode only");
            }
        }
    }
    // shape errors fail before any message
    const u64 sent_before = rig->ch[0]->stats.messages_sent;
    ReleasePolicy rel(1, {"token"}, 1);
    Party lone(0, *rig->ch[0], *rig->client[0], rig->contract, rel, ll::Rng::from_seed(1), rig->dc.model_version);
    CHECK_CODE(bridge::negacyclic_mul_shared(lone, Poly(12), Poly(12)), "BRIDGE_SHAPE", "non power of two refused");
    CHECK_CODE(bridge::negacyclic_mul_shared(lone, Poly(8), Poly(4)), "BRIDGE_SHAPE", "length mismatch refused");
    CHECK(rig->ch[0]->stats.messages_sent == sent_before, "no message on a refused call");
}

// ---------------------------------------------------------------------------------------------
// 3. End to end
// ---------------------------------------------------------------------------------------------
struct PhaseCounts {
    u64 sent = 0, recv = 0, bytes_sent = 0, bytes_recv = 0, rounds = 0;
};

PhaseCounts snap(const Channel& ch) {
    PhaseCounts c;
    c.sent = ch.stats.messages_sent;
    c.recv = ch.stats.messages_recv;
    c.bytes_sent = ch.stats.bytes_sent;
    c.bytes_recv = ch.stats.bytes_recv;
    c.rounds = ch.stats.rounds;
    return c;
}

PhaseCounts diff(const PhaseCounts& a, const PhaseCounts& b) {
    PhaseCounts c;
    c.sent = b.sent - a.sent;
    c.recv = b.recv - a.recv;
    c.bytes_sent = b.bytes_sent - a.bytes_sent;
    c.bytes_recv = b.bytes_recv - a.bytes_recv;
    c.rounds = b.rounds - a.rounds;
    return c;
}

struct TrialOut {
    PhaseCounts share, a2h, eval, h2a;   // per phase, this party's view
    std::vector<u64> y_share;
    Ciphertext ct_x, ct_y;               // P0 only
    size_t first_recv = 0, last_recv = 0;   // range of this party's received frames for the trial
    double h2a_seconds = 0.0;
    u64 h2a_material = 0;
};

void run_end_to_end(int N, int d_in, int d_out, int trials, bridge::Convolution conv, u64 seed) {
    const int log_t = 16;
    const auto params_r = ToyParams::make(N, log_t);
    CHECK_OK(params_r, "toy parameters");
    const ToyParams params = params_r.value;
    const u64 t = params.t();
    const char* mode = conv == bridge::Convolution::KARATSUBA_LEAVES ? "karatsuba_leaves" : "outer_triple";
    auto rig = make_rig(std::string("e2e") + std::to_string(N) + mode, seed);
    lowlevel::RandomSource trng = lowlevel::deterministic_source(seed * 7919 + 1);

    // P1's key pair (the test plays the client's key generation; sk never enters P0's body)
    ToyBFV bfv_keys(params);
    lowlevel::RandomSource krng = lowlevel::deterministic_source(seed * 31 + 5);
    const ToyBFV::KeyPair kp = bfv_keys.keygen(krng);
    lowlevel::RandomSource enc_rng = lowlevel::deterministic_source(seed * 131 + 9);   // P1's encryption randomness

    const bridge::H2aMaterial need = bridge::h2a_material(params, conv);
    CHECK(need.trunc_f == 64 - log_t && need.trunc_L == 64 && need.trunc_elements == static_cast<size_t>(N) && need.rounds == 2, "material need");
    std::vector<int64_t> positions(static_cast<size_t>(d_out));
    for (int k = 0; k < d_out; ++k) positions[static_cast<size_t>(k)] = static_cast<int64_t>(k + 1) * d_in - 1;   // public

    const size_t beaver_words = need.open_bytes_beaver / 8;
    const size_t beaver_frame = need.open_bytes_beaver + kFrameOverhead;
    const size_t trunc_frame = need.open_bytes_trunc + kFrameOverhead;
    const size_t ct_frame = 16 * static_cast<size_t>(N) + kFrameOverhead;
    const size_t share_frame = 8 * static_cast<size_t>(d_in) + kFrameOverhead;

    const int chunk = 10;
    int done = 0;
    long mismatches = 0;
    double h2a_seconds_sum = 0.0;
    u64 h2a_material = 0;
    while (done < trials) {
        const int n = std::min(chunk, trials - done);
        // fresh inputs of the chunk: x is P1's (full-ring values), W is P0's (entries mod t)
        std::vector<std::vector<u64>> X(static_cast<size_t>(n), std::vector<u64>(static_cast<size_t>(d_in)));
        std::vector<Matrix> W(static_cast<size_t>(n), Matrix(static_cast<size_t>(d_out), std::vector<u64>(static_cast<size_t>(d_in))));
        for (int i = 0; i < n; ++i) {
            for (auto& v : X[static_cast<size_t>(i)]) v = trng();
            for (auto& row : W[static_cast<size_t>(i)])
                for (auto& v : row) v = trng() % t;
        }
        if (done == 0) {   // boundary inputs in the first trial
            std::fill(X[0].begin(), X[0].end(), ~u64(0));          // x = -1 in every slot
            for (auto& row : W[0]) std::fill(row.begin(), row.end(), t - 1);   // w = -1 mod t
        }
        provision_h2a_trunc(*rig, params, static_cast<size_t>(n));
        for (int p = 0; p < 2; ++p) {
            rig->ch[p]->received.clear();
            rig->ch[p]->sent.clear();
        }
        std::vector<TrialOut> outs[2] = {std::vector<TrialOut>(static_cast<size_t>(n)), std::vector<TrialOut>(static_cast<size_t>(n))};
        const auto r = run_both(*rig, [&](Party& party, int p) -> Expected<Unit> {
            ToyBFV bfv(params);   // each party's own instance of the public scheme
            SpyChannel& spy = *rig->ch[p];
            for (int i = 0; i < n; ++i) {
                TrialOut& o = outs[p][static_cast<size_t>(i)];
                o.first_recv = spy.received.size();
                const PhaseCounts c0 = snap(spy);
                // share: x is the client's input, one frame P1 -> P0
                Mat x_enc;
                if (p == 1) x_enc = Mat::row_vector(X[static_cast<size_t>(i)]);
                OC2_TRY(sx, party.share_input(p == 1 ? &x_enc : nullptr, 1, 1, static_cast<size_t>(d_in)));
                const PhaseCounts c1 = snap(spy);
                // A2H_toy: one frame P1 -> P0
                lowlevel::RandomSource* rng_ptr = p == 1 ? &enc_rng : nullptr;
                OC2_TRY(ct, bridge::a2h_toy(party, sx.a, bfv, p == 1 ? &kp.pk : nullptr, rng_ptr));
                const PhaseCounts c2 = snap(spy);
                // Eval_toy: P0 only, local
                Ciphertext ct_y;
                if (p == 0) {
                    OC2_TRY(mv, bridge::eval_toy(bfv, ct, W[static_cast<size_t>(i)]));
                    if (mv.positions != positions) return fail("TEST_POSITIONS: positions differ from the public formula");
                    ct_y = std::move(mv.ct_y);
                    o.ct_x = ct;
                    o.ct_y = ct_y;
                } else if (!ct.c0.empty() || !ct.c1.empty()) {
                    return fail("TEST_P1_CIPHERTEXT: a2h_toy returned a ciphertext to P1");
                }
                const PhaseCounts c3 = snap(spy);
                // H2A_ref: two rounds
                const double t0 = now_s();
                const u64 m0 = party.material_bytes();
                OC2_TRY(ys, bridge::h2a_ref(party, bfv, p == 0 ? &ct_y : nullptr, p == 1 ? &kp.sk : nullptr, positions, conv));
                o.h2a_seconds = now_s() - t0;
                o.h2a_material = party.material_bytes() - m0;
                const PhaseCounts c4 = snap(spy);
                o.share = diff(c0, c1);
                o.a2h = diff(c1, c2);
                o.eval = diff(c2, c3);
                o.h2a = diff(c3, c4);
                o.y_share = std::move(ys);
                o.last_recv = spy.received.size();
            }
            return ok_unit();
        });
        CHECK_OK(r[0], "end to end P0");
        CHECK_OK(r[1], "end to end P1");

        for (int i = 0; i < n; ++i) {
            const size_t ui = static_cast<size_t>(i);
            const TrialOut& o0 = outs[0][ui];
            const TrialOut& o1 = outs[1][ui];
            // correctness: reconstruct == W x mod t
            const auto want = bfv_keys.matvec_reference(W[ui], X[ui]);
            CHECK_OK(want, "matvec reference");
            CHECK(o0.y_share.size() == static_cast<size_t>(d_out) && o1.y_share.size() == static_cast<size_t>(d_out), "output share length");
            for (int k = 0; k < d_out; ++k) {
                const u64 got = (o0.y_share[static_cast<size_t>(k)] + o1.y_share[static_cast<size_t>(k)]) & (t - 1);
                if (got != want.value[static_cast<size_t>(k)]) ++mismatches;
            }
            // the noise bound is respected and the plain decryption agrees (test-only use of sk)
            CHECK(bfv_keys.noise_budget_ok(d_in, bfv_keys.w_inf(W[ui])).value, "noise bound respected");
            const auto dec_x = bfv_keys.decrypt(kp.sk, o0.ct_x);
            const auto dec_y = bfv_keys.decrypt(kp.sk, o0.ct_y);
            CHECK(dec_x.ok && dec_y.ok, "reference decryption");
            bool x_ok = true, y_ok = true;
            for (int j = 0; j < d_in; ++j) x_ok = x_ok && (dec_x.value[static_cast<size_t>(j)] == (X[ui][static_cast<size_t>(j)] & (t - 1)));
            for (int j = d_in; j < N; ++j) x_ok = x_ok && (dec_x.value[static_cast<size_t>(j)] == 0);
            for (int k = 0; k < d_out; ++k) y_ok = y_ok && (dec_y.value[static_cast<size_t>(positions[static_cast<size_t>(k)])] == want.value[static_cast<size_t>(k)]);
            CHECK(x_ok, "A2H_toy: P0 holds an encryption of x mod t");
            CHECK(y_ok, "Eval_toy: the evaluated ciphertext decrypts to W x mod t at the positions");

            // --- frames by direction ---------------------------------------------------------------
            // share: one frame P1 -> P0
            CHECK(o1.share.sent == 1 && o1.share.recv == 0 && o0.share.sent == 0 && o0.share.recv == 1, "share: one frame P1 -> P0");
            CHECK(o1.share.bytes_sent == share_frame && o0.share.bytes_recv == share_frame, "share: frame size");
            // A2H_toy: one frame P1 -> P0, nothing towards P1
            CHECK(o1.a2h.sent == 1 && o1.a2h.recv == 0, "A2H_toy: P1 sends the ciphertext and receives nothing");
            CHECK(o0.a2h.sent == 0 && o0.a2h.recv == 1, "A2H_toy: P0 receives the ciphertext and sends nothing");
            CHECK(o1.a2h.bytes_sent == ct_frame && o0.a2h.bytes_recv == ct_frame && o1.a2h.bytes_recv == 0, "A2H_toy: 2 * 8 * N data bytes, one direction");
            CHECK(o0.a2h.rounds == 1 && o1.a2h.rounds == 1, "A2H_toy: one round");
            // Eval_toy: no frame at all
            CHECK(o0.eval.sent == 0 && o0.eval.recv == 0 && o1.eval.sent == 0 && o1.eval.recv == 0, "Eval_toy: local to P0");
            // H2A_ref: two exchanges, symmetric sizes that depend on public shapes only
            for (const TrialOut* o : {&o0, &o1}) {
                CHECK(o->h2a.sent == 2 && o->h2a.recv == 2, "H2A_ref: two frames each way");
                CHECK(o->h2a.rounds == 2, "H2A_ref: two rounds");
                CHECK(o->h2a.bytes_sent == beaver_frame + trunc_frame && o->h2a.bytes_recv == beaver_frame + trunc_frame,
                      "H2A_ref: opening bytes are a function of N and the mode");
                CHECK(o->h2a_material == outs[0][0].h2a_material, "H2A_ref: material per call is constant");
            }
            // --- what P1 received in the whole trial: exactly the two masked openings ------------------
            CHECK(o1.last_recv - o1.first_recv == 2, "P1 receives two frames per trial");
            for (size_t f = o1.first_recv; f < o1.last_recv; ++f) {
                const auto arr = rig->ch[1]->received[f].as_array();
                CHECK_OK(arr, "P1 frame is an array");
                const bool ct_shape = arr.value.dims.size() == 2 && arr.value.dims[0] == 2 && arr.value.dims[1] == static_cast<u64>(N);
                CHECK(!ct_shape, "P1 never receives a frame of ciphertext shape (2, N)");
                const std::vector<u64> words = arr.value.as_u64();
                CHECK(words.size() == (f == o1.first_recv ? beaver_words : static_cast<size_t>(N)), "P1 frame is an opening of the expected size");
                CHECK(!contains_run(words, o0.ct_y.c0) && !contains_run(words, o0.ct_y.c1), "no component of the evaluated ciphertext reaches P1");
                CHECK(!contains_run(words, o0.ct_x.c0), "P0's updated c0 never reaches P1");
            }
            // --- what P0 received: the input share, the fresh ciphertext, the two openings; never s ----
            CHECK(o0.last_recv - o0.first_recv == 4, "P0 receives four frames per trial");
            for (size_t f = o0.first_recv; f < o0.last_recv; ++f) {
                const std::vector<u64> words = rig->ch[0]->received[f].as_array().value.as_u64();
                CHECK(!contains_run(words, kp.sk), "the secret key never reaches P0");
            }
            h2a_seconds_sum += o0.h2a_seconds;
            h2a_material = o0.h2a_material;
        }
        done += n;
    }
    CHECK(mismatches == 0, "share -> a2h_toy -> eval_toy -> h2a_ref -> reconstruct equals W x mod t in every trial");
    // material: the Beaver part plus N truncation elements with (f, L) = (64 - log_t, 64)
    const u64 trunc_bytes = static_cast<u64>(N) * lowlevel::trunc::trunc_bytes_per_element(64 - log_t);
    const u64 beaver_bytes = conv == bridge::Convolution::KARATSUBA_LEAVES
                                 ? 24 * static_cast<u64>(need.beaver_triples)
                                 : 8 * (2 * static_cast<u64>(N) + static_cast<u64>(N) * static_cast<u64>(N));
    CHECK(h2a_material == trunc_bytes + beaver_bytes, "H2A_ref material bytes per call");
    std::fprintf(stderr,
                 "[test_bridge] {\"gadget\":\"h2a_ref\",\"status\":\"TEST_ONLY toy parameters\",\"N\":%d,\"log_t\":%d,\"d_in\":%d,\"d_out\":%d,"
                 "\"mode\":\"%s\",\"trials\":%d,\"mismatches\":%ld,\"rounds\":2,\"bytes_sent_per_party\":%zu,\"beaver_open_data_bytes\":%zu,"
                 "\"trunc_open_data_bytes\":%zu,\"material_bytes_per_party\":%llu,\"mean_wall_s\":%.6f}\n",
                 N, log_t, d_in, d_out, mode, trials, mismatches, beaver_frame + trunc_frame, need.open_bytes_beaver, need.open_bytes_trunc,
                 static_cast<unsigned long long>(h2a_material), h2a_seconds_sum / trials);
}

// ---------------------------------------------------------------------------------------------
// 4. Fail closed
// ---------------------------------------------------------------------------------------------
void test_fail_closed() {
    const ToyParams params = ToyParams::make(128, 16).value;
    ToyBFV bfv(params);
    lowlevel::RandomSource krng = lowlevel::deterministic_source(77);
    const ToyBFV::KeyPair kp = bfv.keygen(krng);
    lowlevel::RandomSource erng = lowlevel::deterministic_source(78);
    const std::vector<int64_t> positions = {7, 15};
    const Ciphertext some_ct = bfv.encrypt(kp.pk, Poly(128, 0), erng).value;

    // role violations are refused before any message leaves either party
    {
        auto rig = make_rig("roles", 31);
        ReleasePolicy rel(1, {"token"}, 1);
        Party p0(0, *rig->ch[0], *rig->client[0], rig->contract, rel, ll::Rng::from_seed(1), rig->dc.model_version);
        Party p1(1, *rig->ch[1], *rig->client[1], rig->contract, rel, ll::Rng::from_seed(2), rig->dc.model_version);
        CHECK_CODE(bridge::h2a_ref(p1, bfv, &some_ct, &kp.sk, positions), "BRIDGE_PARTY", "P1 handed a ciphertext is refused");
        CHECK_CODE(bridge::h2a_ref(p1, bfv, nullptr, nullptr, positions), "BRIDGE_PARTY", "P1 without its key is refused");
        CHECK_CODE(bridge::h2a_ref(p0, bfv, &some_ct, &kp.sk, positions), "BRIDGE_PARTY", "P0 handed the secret key is refused");
        CHECK_CODE(bridge::h2a_ref(p0, bfv, nullptr, nullptr, positions), "BRIDGE_PARTY", "P0 without a ciphertext is refused");
        CHECK_CODE(bridge::h2a_ref(p0, bfv, &some_ct, nullptr, {128}), "BRIDGE_SHAPE", "position out of range refused");
        CHECK_CODE(bridge::h2a_ref(p0, bfv, &some_ct, nullptr, {-1}), "BRIDGE_SHAPE", "negative position refused");
        Ciphertext short_ct;
        short_ct.c0.assign(64, 0);
        short_ct.c1.assign(64, 0);
        CHECK_CODE(bridge::h2a_ref(p0, bfv, &short_ct, nullptr, positions), "BRIDGE_SHAPE", "ciphertext of the wrong length refused");
        const Poly short_sk(64, 0);
        CHECK_CODE(bridge::h2a_ref(p1, bfv, nullptr, &short_sk, positions), "BRIDGE_SHAPE", "key of the wrong length refused");
        CHECK_CODE(bridge::a2h_toy(p1, std::vector<u64>(8, 1), bfv, nullptr, &erng), "BRIDGE_PARTY", "P1 without a public key is refused");
        CHECK_CODE(bridge::a2h_toy(p1, std::vector<u64>(8, 1), bfv, &kp.pk, nullptr), "BRIDGE_PARTY", "P1 without randomness is refused");
        CHECK_CODE(bridge::a2h_toy(p1, std::vector<u64>(129, 1), bfv, &kp.pk, &erng), "BRIDGE_SHAPE", "share longer than N refused");
        CHECK_CODE(bridge::a2h_toy(p0, std::vector<u64>(129, 1), bfv, nullptr, nullptr), "BRIDGE_SHAPE", "share longer than N refused on P0");
        for (int p = 0; p < 2; ++p)
            CHECK(rig->ch[p]->stats.messages_sent == 0 && rig->ch[p]->stats.messages_recv == 0 && rig->ch[p]->sent.empty(),
                  "no frame left either party on a refused call");
        // a frame that is not a (2, N) ciphertext is refused by P0
        std::thread peer([&]() { (void)rig->ch[1]->send(u64_array(std::vector<u64>(256, 1), {1, 256})); });
        CHECK_CODE(bridge::a2h_toy(p0, std::vector<u64>(8, 1), bfv, nullptr, nullptr), "BRIDGE_SHAPE", "frame of the wrong shape refused");
        peer.join();
        CHECK_CODE(bridge::h2a_trunc_bundles(*rig->dealer, params, 0), "BRIDGE_PARAMS", "zero calls refused");
    }
    // noise budget: eval_toy refuses a parameter set whose worst case reaches Delta / 2
    {
        const ToyParams tight = ToyParams::make(512, 32).value;   // Delta = 2^32
        ToyBFV b2(tight);
        lowlevel::RandomSource r2 = lowlevel::deterministic_source(79);
        const ToyBFV::KeyPair k2 = b2.keygen(r2);
        const Ciphertext c = b2.encrypt(k2.pk, Poly(512, 0), r2).value;
        Matrix W(8, std::vector<u64>(64, (u64(1) << 31) - 1));
        CHECK_CODE(bridge::eval_toy(b2, c, W), "TOY_BFV_NOISE_BUDGET", "violating parameter set refused");
    }
    // Truncation material that was not provisioned by hand. The refill request of
    // DealerClient::take_trunc carries (f, L) (the Python spelling {f, L}), and the pool is keyed
    // by (f, L): h2a_ref obtains (f = 64 - log_t, L = 64) material through the refill link although
    // the contract width is L = 40. (Before the fix the request carried only f: the dealer built
    // L = 40 material, which was journaled as spent and then refused with MATERIAL_PARAMS, and
    // (f = 48, L = 40) could not be built at all.)
    {
        auto rig = make_rig("refillL16", 41);
        std::vector<u64> ys[2];
        const auto r = run_both(*rig, [&](Party& party, int p) -> Expected<Unit> {
            ToyBFV b(params);
            const u64 sent0 = party.channel().stats.messages_sent;
            OC2_TRY(y, bridge::h2a_ref(party, b, p == 0 ? &some_ct : nullptr, p == 1 ? &kp.sk : nullptr, positions));
            if (party.channel().stats.messages_sent - sent0 != 2) return fail("TEST_FRAMES: expected the Beaver and the truncation opening");
            ys[p] = std::move(y);
            return ok_unit();
        });
        CHECK_OK(r[0], "P0: (f = 48, L = 64) material arrives through the refill link");
        CHECK_OK(r[1], "P1: (f = 48, L = 64) material arrives through the refill link");
        CHECK(ys[0].size() == positions.size() && ys[1].size() == positions.size(), "one share per requested position");
        for (int p = 0; p < 2; ++p) {
            CHECK(rig->client[p]->pooled_count("trunc:48|40") == 0, "no material of the contract width was requested");
            CHECK_CODE(rig->client[p]->take_trunc(1, 48, 40), "MATERIAL_PARAMS", "(f = 48, L = 40) is refused before any request");
        }
    }
    // two widths at the same shift live in two pools and are never mixed
    {
        auto rig = make_rig("twoL", 44);
        for (int p = 0; p < 2; ++p) {
            auto a = rig->client[p]->take_trunc(3, 16, 40);
            auto b = rig->client[p]->take_trunc(2, 16, 64);
            CHECK_OK(a, "take (16, 40)");
            CHECK_OK(b, "take (16, 64)");
            CHECK(a.value.M() == 3 && a.value.L == 40 && b.value.M() == 2 && b.value.L == 64, "each request gets its own width");
        }
    }
    {
        const ToyParams p32 = ToyParams::make(128, 32).value;
        ToyBFV b32(p32);
        lowlevel::RandomSource r32 = lowlevel::deterministic_source(80);
        const ToyBFV::KeyPair k32 = b32.keygen(r32);
        const Ciphertext c32 = b32.encrypt(k32.pk, Poly(128, 0), r32).value;
        auto rig = make_rig("refillL32", 42);
        std::vector<u64> yr[2];
        const auto r = run_both(*rig, [&](Party& party, int p) -> Expected<Unit> {
            ToyBFV b(p32);
            OC2_TRY(y, bridge::h2a_ref(party, b, p == 0 ? &c32 : nullptr, p == 1 ? &k32.sk : nullptr, positions));
            yr[p] = std::move(y);
            return ok_unit();
        });
        CHECK_OK(r[0], "P0: h2a_ref at log_t = 32 without hand provisioning");
        CHECK_OK(r[1], "P1: h2a_ref at log_t = 32 without hand provisioning");
        for (size_t k = 0; k < positions.size(); ++k)
            CHECK(((yr[0][k] + yr[1][k]) & (p32.t() - 1)) == 0, "refilled (f = 32, L = 64) material decrypts an encryption of zero to zero");
        // with the right material the same parameter set works (log_t = 32, f = 32)
        auto rig2 = make_rig("rightL32", 43);
        provision_h2a_trunc(*rig2, p32, 1);
        std::vector<u64> ys[2];
        const auto r2 = run_both(*rig2, [&](Party& party, int p) -> Expected<Unit> {
            ToyBFV b(p32);
            OC2_TRY(y, bridge::h2a_ref(party, b, p == 0 ? &c32 : nullptr, p == 1 ? &k32.sk : nullptr, positions));
            ys[p] = std::move(y);
            return ok_unit();
        });
        CHECK_OK(r2[0], "h2a_ref at log_t = 32, P0");
        CHECK_OK(r2[1], "h2a_ref at log_t = 32, P1");
        for (size_t k = 0; k < positions.size(); ++k)
            CHECK(((ys[0][k] + ys[1][k]) & (p32.t() - 1)) == 0, "an encryption of zero decrypts to zero through h2a_ref at log_t = 32");
    }
}

}  // namespace

int main(int argc, char** argv) {
    int trials = 200;
    if (argc > 1) trials = std::atoi(argv[1]);   // optional: fewer trials for a quick run
    if (trials < 1) trials = 1;
    CHECK_OK(ensure_sockets(), "sockets");
    test_karatsuba_plain();
    std::fprintf(stderr, "[test_bridge] karatsuba ok (%ld checks so far)\n", g_checks);
    test_shared_product();
    std::fprintf(stderr, "[test_bridge] shared product ok (%ld checks so far)\n", g_checks);
    // 1000 random trials in total over the two admitted shapes: 800 + 200
    run_end_to_end(128, 8, 8, 4 * trials, bridge::Convolution::KARATSUBA_LEAVES, 101);
    run_end_to_end(512, 64, 8, trials, bridge::Convolution::KARATSUBA_LEAVES, 102);
    run_end_to_end(128, 8, 8, trials, bridge::Convolution::OUTER_TRIPLE, 103);
    run_end_to_end(512, 64, 8, std::max(1, trials / 10), bridge::Convolution::OUTER_TRIPLE, 104);
    std::fprintf(stderr, "[test_bridge] end to end ok (%ld checks so far)\n", g_checks);
    test_fail_closed();
    std::printf("PASS %ld checks\n", g_checks);
    return 0;
}
