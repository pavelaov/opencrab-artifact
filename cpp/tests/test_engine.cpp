// OpenCrab v2 C++17 mirror: engine tests for the high-level layer.
//
// Covers: wire codec round trips and the canonical JSON / Python float repr,
// SHA-256 and HMAC-SHA256 vectors (FIPS 180-4, RFC 4231), the WallClockShaper,
// the loopback Channel and its round accounting, the MaterialLedger invariants
// (reuse refusal, atomic reserve, recover burns RESERVED, retransmission log,
// versioned installed objects, binding), dealer bundles (sealing, tamper /
// recipient / duplicate rejection, pooled partial consumption), the material
// estimate, the polynomial contracts, and an in-process two-party loop (two
// threads over a loopback socket pair) that checks every Party gadget against
// double references with the declared tolerances and the k-ary argmax
// exact-equality property (k in {2, 4, 8}, lowest index on ties).
//
// Added with the parity pass against the Python implementation: the FUSED
// two-round k-ary tournament is checked for its
// exact accounting on every (n, k) of the loop (rounds = 2 levels, channel
// bytes and material bytes equal to the formula of what the Python realization
// issues) and against 18 values measured on python Party._tournament, with a
// public mask (publicly decided pairs open nothing) and just inside the
// magnitude bound of every arity; dealer headers carry the Python spellings
// (shape_a / shape_b / shape) while the earlier spellings are still read; the
// kary_select material type has its own pool; DealerClient::burn_outstanding
// burns RESERVED and untouched bundles through the ledger; Party::trunc_local
// equals the lowlevel gate (explicit arithmetic shift); probe_channel refuses
// samples < 1 with TELEMETRY_PARAMS before any message.
//
// Prints "PASS <n> checks" and exits 0, or prints the failing check and exits 1.
#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/dealer.hpp>
#include <opencrab2/highlevel/gadgets.hpp>
#include <opencrab2/highlevel/ledger.hpp>
#include <opencrab2/highlevel/party.hpp>
#include <opencrab2/highlevel/telemetry.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace opencrab2;
using namespace opencrab2::highlevel;

static int g_checks = 0;

#define CHECK(cond, msg)                                                                  \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
            return 1;                                                                     \
        }                                                                                 \
        ++g_checks;                                                                       \
    } while (0)

#define CHECK_OK(expr, msg)                                                               \
    do {                                                                                  \
        auto&& oc2_r = (expr);                                                            \
        if (!oc2_r.ok) {                                                                  \
            std::fprintf(stderr, "FAIL: %s: %s (line %d)\n", msg, oc2_r.code.c_str(), __LINE__); \
            return 1;                                                                     \
        }                                                                                 \
        ++g_checks;                                                                       \
    } while (0)

static std::string temp_path(const std::string& name) {
    const char* base = std::getenv("TEMP");
    if (!base) base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::string(base) + "/oc2_engine_" + name + "_" + std::to_string(static_cast<long long>(stamp)) + ".jsonl";
}

// ---------------------------------------------------------------------------
// codec
// ---------------------------------------------------------------------------
static int test_codec() {
    Value v = dict({{"b", static_cast<int64_t>(1)},
                    {"a", Value(List{Value(1.5), Value("x"), Value(true), Value(None{})})},
                    {"c", dict({{"z", 1e-05}})}});
    CHECK_OK(canonical_json(v), "canonical json");
    CHECK(canonical_json(v).value == "{\"a\":[1.5,\"x\",true,null],\"b\":1,\"c\":{\"z\":1e-05}}", "canonical JSON bytes");
    CHECK(python_float_repr(0.1) == "0.1", "repr 0.1");
    CHECK(python_float_repr(100.0) == "100.0", "repr 100.0");
    CHECK(python_float_repr(1e16) == "1e+16", "repr 1e16");
    CHECK(python_float_repr(1e-5) == "1e-05", "repr 1e-05");
    CHECK(python_float_repr(0.0001) == "0.0001", "repr 0.0001");
    CHECK(python_float_repr(123456789012345680.0) == "1.2345678901234568e+17", "repr large");
    CHECK(python_float_repr(-2.5) == "-2.5", "repr negative");
    CHECK(python_float_repr(1e15) == "1000000000000000.0", "repr 1e15");
    // wire round trip of every tag
    Value w = dict({{"bytes", Value(Bytes{1, 2, 3})},
                    {"arr", u64_array({1, 2, 0xffffffffffffffffULL}, {3})},
                    {"farr", f64_array({0.5, -1.25}, {2, 1})},
                    {"i", static_cast<int64_t>(-7)},
                    {"d", 3.25},
                    {"s", "hello"},
                    {"l", Value(List{Value(static_cast<int64_t>(1)), Value("a")})},
                    {"n", Value(None{})},
                    {"t", true}});
    Bytes enc = encode_value(w);
    auto dec = decode_value(enc);
    CHECK_OK(dec, "decode");
    CHECK(encode_value(dec.value) == enc, "codec round trip bytes");
    CHECK(dec.value.get_array("arr").value.as_u64()[2] == 0xffffffffffffffffULL, "u64 array value");
    CHECK(dec.value.get_array("farr").value.as_f64()[1] == -1.25, "f64 array value");
    CHECK(dec.value.get_int("i").value == -7, "int value");
    Bytes frame = encode_frame(w);
    CHECK(frame.size() == enc.size() + 8 && get_u64be(frame.data()) == enc.size(), "frame prefix");
    // tag check: hand-encoded int 5 = 0x04 + 8 bytes BE
    Bytes five = {0x04, 0, 0, 0, 0, 0, 0, 0, 5};
    CHECK(decode_value(five).ok && decode_value(five).value.as_int().value == 5, "int tag layout");
    Bytes bad = {0x42};
    CHECK(!decode_value(bad).ok && decode_value(bad).code == "CODEC_TAG", "bad tag rejected");
    Bytes trunc = {0x06, 0, 0, 0, 5, 'a'};
    CHECK(!decode_value(trunc).ok, "truncated string rejected");
    // sha256 / hmac vectors
    CHECK(sha256_hex(std::string("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    CHECK(sha256_hex(std::string("")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    Bytes key(reinterpret_cast<const uint8_t*>("Jefe"), reinterpret_cast<const uint8_t*>("Jefe") + 4);
    std::string data = "what do ya want for nothing?";
    Bytes msg(data.begin(), data.end());
    CHECK(hmac_sha256_hex(key, msg) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843", "HMAC RFC4231 case 2");
    Bytes key1(20, 0x0b);
    std::string hi = "Hi There";
    CHECK(hmac_sha256_hex(key1, Bytes(hi.begin(), hi.end())) == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
          "HMAC RFC4231 case 1");
    auto tag = hmac_tag(key, v);
    CHECK_OK(tag, "hmac_tag");
    CHECK(hmac_verify(key, v, tag.value).value, "hmac verify ok");
    std::string bad_tag = tag.value;
    bad_tag[0] = bad_tag[0] == 'a' ? 'b' : 'a';
    CHECK(!hmac_verify(key, v, bad_tag).value, "hmac verify tampered");
    // Json bridge
    Json j = value_to_json(v);
    CHECK(j.dump() == "{\"a\":[1.5,\"x\",true,null],\"b\":1,\"c\":{\"z\":1e-05}}", "value_to_json canonical");
    Value back = json_to_value(j);
    CHECK(canonical_json(back).value == canonical_json(v).value, "json_to_value round trip");
    auto parsed = parse_json("{\"x\": [1, 2.5, \"s\", null, true], \"y\": {\"z\": -3}}");
    CHECK_OK(parsed, "parse_json");
    CHECK(parsed.value.get("y")->get_int("z").value == -3, "parse_json nested");
    return 0;
}

// ---------------------------------------------------------------------------
// shaper / channel
// ---------------------------------------------------------------------------
static int test_shaper_and_channel() {
    std::vector<TracePoint> tr = {{0.0, LinkState{1e6, 0.01}}, {1.0, LinkState{2e6, 0.02}}};
    WallClockShaper sh(tr, 100.0);
    // 2e6 bits: 1 s at 1 Mbit/s, then 0.5 s at 2 Mbit/s; the last byte leaves in the second segment: 20 ms owd
    double d = sh.delay_for(250000, 100.0);
    CHECK(std::fabs(d - 1.52) < 1e-6, "shaper integrates across the segment boundary; delay of the state at the END");
    double d2 = sh.delay_for(125000, 100.0);  // link busy until 101.5: starts there, 0.5 s at 2 Mbit/s + 20 ms owd
    CHECK(std::fabs(d2 - (1.5 + 0.5 + 0.02)) < 1e-6, "shaper token bucket carries the busy time");
    CHECK(sh.state_at(100.5).bandwidth_bps == 1e6 && sh.state_at(101.5).bandwidth_bps == 2e6, "state_at");
    WallClockShaper unshaped({{0.0, LinkState{0.0, 0.0}}}, 0.0);
    CHECK(unshaped.delay_for(1 << 20, 5.0) == 0.0, "unshaped segment");
    {
        // the shared rule on a four-segment trace (also in vectors/shaper_vectors.json)
        const std::vector<TracePoint> probe = {{0.0, LinkState{8000.0, 0.010}}, {1.0, LinkState{80000.0, 0.050}},
                                               {2.0, LinkState{0.0, 0.200}}, {3.0, LinkState{800.0, 0.0}}};
        auto at = [&](size_t n, double w) { return WallClockShaper(probe, 0.0).delay_for(n, w); };
        CHECK(std::fabs(at(1000, 0.0) - 1.050) < 1e-9, "1000 bytes at 0: 1 s, delay of the second segment");
        CHECK(std::fabs(at(1500, 0.0) - 1.100) < 1e-9, "1500 bytes at 0");
        CHECK(std::fabs(at(20000, 0.5) - 1.700) < 1e-9, "20000 bytes at 0.5: the zero-bandwidth segment is unshaped, its delay applies");
        CHECK(std::fabs(at(100, 2.5) - 0.200) < 1e-9, "100 bytes inside the unshaped segment");
        CHECK(std::fabs(WallClockShaper(probe, 0.0).serialization_time(1500, 0.0) - 1.05) < 1e-9, "serialization_time is the pure drain time");
        // validation: negative or non-finite values are refused, never repaired
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        for (const LinkState& bad : {LinkState{-1.0, 0.0}, LinkState{nan, 0.0}, LinkState{inf, 0.0}, LinkState{1e6, -1.0},
                                     LinkState{1e6, nan}, LinkState{1e6, inf}}) {
            auto m = WallClockShaper::make({{0.0, bad}}, 0.0);
            CHECK(!m.ok && m.code.rfind("CHANNEL_SHAPER_PARAMS", 0) == 0, "invalid link state refused");
            WallClockShaper raw({{0.0, bad}}, 0.0);
            CHECK(!raw.valid() && std::isnan(raw.delay_for(10, 0.0)), "the plain constructor marks the shaper invalid");
        }
        CHECK(!WallClockShaper::make({}, 0.0).ok && !WallClockShaper::make({{nan, LinkState{1e6, 0.0}}}, 0.0).ok &&
                  !WallClockShaper::make({{0.0, LinkState{1e6, 0.0}}}, nan).ok,
              "empty trace, non-finite time and non-finite origin refused");
        CHECK(WallClockShaper::make(probe, 0.0).ok, "a valid trace is accepted");
        // a channel never sends unshaped through an invalid shaper
        auto bp = loopback_pair(std::make_shared<WallClockShaper>(std::vector<TracePoint>{{0.0, LinkState{-5.0, 0.0}}}, 0.0), nullptr);
        CHECK_OK(bp, "loopback pair with an invalid shaper on one side");
        auto sr = bp.value.first->send(Value(static_cast<int64_t>(1)));
        CHECK(!sr.ok && sr.code.rfind("CHANNEL_SHAPER_PARAMS", 0) == 0 && bp.value.first->stats.bytes_sent == 0,
              "send fails closed on an invalid shaper and writes nothing");
    }

    auto pair = loopback_pair();
    CHECK_OK(pair, "loopback pair");
    Channel& c0 = *pair.value.first;
    Channel& c1 = *pair.value.second;
    Value payload = dict({{"k", u64_array({1, 2, 3}, {3})}});
    Expected<Value> got1;
    std::thread t([&]() { got1 = c1.exchange(Value("from1")); });
    auto got0 = c0.exchange(payload);
    t.join();
    CHECK_OK(got0, "exchange P0");
    CHECK_OK(got1, "exchange P1");
    CHECK(got0.value.as_string().value == "from1", "P0 received P1 message");
    CHECK(got1.value.get_array("k").value.as_u64()[2] == 3, "P1 received P0 message");
    CHECK(c0.stats.rounds == 1 && c1.stats.rounds == 1, "exchange counts one round");
    CHECK(c0.stats.bytes_sent == c1.stats.bytes_recv && c1.stats.bytes_sent == c0.stats.bytes_recv, "byte counters agree");
    CHECK(c0.stats.bytes_sent == 8 + encode_value(payload).size(), "bytes include the 8-byte prefix");
    // bare send then recv closes one round
    std::thread t2([&]() {
        auto r = c1.recv();
        (void)r;
        auto s = c1.send(Value(static_cast<int64_t>(2)));
        (void)s;
    });
    CHECK_OK(c0.send(Value(static_cast<int64_t>(1))), "bare send");
    auto r0 = c0.recv();
    t2.join();
    CHECK_OK(r0, "recv after bare send");
    CHECK(c0.stats.rounds == 2, "send then recv counts one round");
    return 0;
}

// ---------------------------------------------------------------------------
// ledger
// ---------------------------------------------------------------------------
static int test_ledger() {
    std::string path = temp_path("ledger");
    Value binding = dict({{"session_id", "s1"}, {"setup_id", "setup-A"}, {"model_version", "m1"},
                          {"numeric_contract_hash", "h1"}, {"profile_id", "SH2P-TrustedOffline-v1"}});
    Value ver = dict({{"setup_id", "setup-A"}, {"model_version", "m1"}, {"numeric_contract_hash", "h1"},
                      {"profile_id", "SH2P-TrustedOffline-v1"}});
    {
        MaterialLedger L(path, binding);
        CHECK_OK(L.open(), "ledger open");
        CHECK_OK(L.add_available("m-1", "beaver", 240, ver), "add m-1");
        CHECK_OK(L.add_available("m-2", "beaver", 240, ver), "add m-2");
        CHECK_OK(L.add_available("m-3", "drelu", 1000, ver), "add m-3");
        CHECK(!L.add_available("m-1", "beaver", 240, ver).ok, "duplicate id refused");
        CHECK(L.stock("beaver").first == 2 && L.stock("beaver").second == 480, "stock counts AVAILABLE");
        CHECK(!L.reserve({"m-1", "m-9"}, "g", 0).ok && L.state("m-1").value == "AVAILABLE", "reserve is atomic");
        CHECK_OK(L.reserve({"m-1", "m-2"}, "g", 0), "reserve");
        CHECK(L.state("m-1").value == "RESERVED", "state RESERVED");
        CHECK(!L.spend({"m-3"}).ok, "spend requires RESERVED");
        CHECK_OK(L.spend({"m-1"}), "spend m-1");
        CHECK(L.state("m-1").value == "SPENT", "state SPENT");
        CHECK(!L.reserve({"m-1"}, "g", 1).ok && L.reserve({"m-1"}, "g", 1).code.rfind("MATERIAL_REUSE", 0) == 0,
              "SPENT id never returned");
        CHECK_OK(L.burn({"m-3"}, "cancel"), "burn");
        CHECK(L.state("m-3").value == "BURNED", "state BURNED");
        CHECK(!L.reserve({"m-3"}, "g", 1).ok, "BURNED id never returned");
        CHECK_OK(L.record_send("g", 0, "d0"), "record_send");
        CHECK(L.retransmit_allowed("g", 0, "d0"), "retransmit identical");
        CHECK(!L.retransmit_allowed("g", 0, "d1"), "retransmit different digest refused");
        CHECK(!L.record_send("g", 0, "d1").ok, "seq reuse with a new payload refused");
        Value v1 = dict({{"model_version", "m1"}, {"client_key_domain", "c1"}});
        Value v2 = dict({{"model_version", "m2"}, {"client_key_domain", "c1"}});
        CHECK_OK(L.register_installed("wmask_install(w)", v1), "register installed");
        CHECK(L.installed("wmask_install(w)", v1), "installed with the same version");
        CHECK(!L.installed("wmask_install(w)", v2), "installed with another version is false");
        Value foreign = dict({{"setup_id", "setup-B"}, {"model_version", "m1"}, {"numeric_contract_hash", "h1"},
                              {"profile_id", "SH2P-TrustedOffline-v1"}});
        CHECK(!L.check_binding(foreign).ok, "foreign binding rejected");
        CHECK_OK(L.check_binding(binding), "own binding accepted");
        // m-2 stays RESERVED: a restart must burn it
    }
    {
        MaterialLedger L2(path, binding);
        auto rec = L2.recover();
        CHECK_OK(rec, "recover");
        CHECK(L2.state("m-2").value == "BURNED", "recover burns RESERVED");
        CHECK(L2.state("m-1").value == "SPENT", "recover keeps SPENT");
        CHECK(rec.value.get_int("reserved_burned").value == 1, "recover count");
        CHECK(L2.retransmit_allowed("g", 0, "d0"), "send log survives restart");
        CHECK(L2.installed("wmask_install(w)", dict({{"model_version", "m1"}, {"client_key_domain", "c1"}})),
              "installed objects survive restart");
        CHECK(!L2.reserve({"m-2"}, "g", 2).ok, "burned after recover is terminal");
    }
    std::remove(path.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// dealer
// ---------------------------------------------------------------------------
struct TwoParty {
    NumericContract contract = default_contract();
    std::shared_ptr<Dealer> dealer;
    std::shared_ptr<InProcessDealerLink> link;
    std::unique_ptr<Channel> ch[2];
    std::unique_ptr<MaterialLedger> ledger[2];
    std::unique_ptr<DealerClient> client[2];
    std::string paths[2];
    Bytes key[2];
};

static Expected<std::unique_ptr<TwoParty>> make_two_party(const std::string& tag) {
    auto tp = std::make_unique<TwoParty>();
    DealerConfig dc;
    dc.setup_id = "setup-" + tag;
    dc.model_version = "model-" + tag;
    dc.numeric_contract_hash = tp->contract.contract_hash();
    tp->key[0] = Bytes(32, 0x11);
    tp->key[1] = Bytes(32, 0x22);
    tp->dealer = std::make_shared<Dealer>(dc, ll::Rng::from_seed(42), tp->key[0], tp->key[1]);
    tp->link = std::make_shared<InProcessDealerLink>(tp->dealer);
    OC2_TRY(pair, loopback_pair());
    tp->ch[0] = std::move(pair.first);
    tp->ch[1] = std::move(pair.second);
    for (int p = 0; p < 2; ++p) {
        tp->paths[p] = temp_path(tag + "_p" + std::to_string(p));
        Value binding = dict({{"session_id", "sess-" + tag}, {"setup_id", dc.setup_id}, {"model_version", dc.model_version},
                              {"numeric_contract_hash", dc.numeric_contract_hash}, {"profile_id", dc.profile_id}});
        tp->ledger[p] = std::make_unique<MaterialLedger>(tp->paths[p], binding);
        OC2_CHECK(tp->ledger[p]->open());
        tp->client[p] = std::make_unique<DealerClient>(p, *tp->ledger[p], tp->key[p], dc.numeric_contract_hash);
        tp->link->attach(*tp->client[p]);
    }
    return ok(std::move(tp));
}

static int test_dealer() {
    auto tpr = make_two_party("dealer");
    CHECK_OK(tpr, "two party setup");
    TwoParty& tp = *tpr.value;
    auto bp = tp.dealer->make_beaver(10);
    CHECK_OK(bp, "make beaver");
    CHECK(bp.value.for0.header.get_int("recipient").value == 0 && bp.value.for1.header.get_int("recipient").value == 1,
          "recipient fields");
    CHECK(bp.value.for0.header.get_string("material_id").value == bp.value.for1.header.get_string("material_id").value,
          "pair shares a material_id");
    // reconstruct a*b = c from the two payloads
    {
        size_t at0 = 0, at1 = 0;
        auto a0 = ll::get_u64_vec(bp.value.for0.payload.data(), bp.value.for0.payload.size(), 10, &at0).value;
        auto b0 = ll::get_u64_vec(bp.value.for0.payload.data(), bp.value.for0.payload.size(), 10, &at0).value;
        auto c0 = ll::get_u64_vec(bp.value.for0.payload.data(), bp.value.for0.payload.size(), 10, &at0).value;
        auto a1 = ll::get_u64_vec(bp.value.for1.payload.data(), bp.value.for1.payload.size(), 10, &at1).value;
        auto b1 = ll::get_u64_vec(bp.value.for1.payload.data(), bp.value.for1.payload.size(), 10, &at1).value;
        auto c1 = ll::get_u64_vec(bp.value.for1.payload.data(), bp.value.for1.payload.size(), 10, &at1).value;
        bool okk = true;
        for (size_t i = 0; i < 10; ++i) okk = okk && ((a0[i] + a1[i]) * (b0[i] + b1[i]) == c0[i] + c1[i]);
        CHECK(okk, "beaver correlation");
    }
    // wrong recipient / tamper / duplicate
    CHECK(!tp.client[1]->ingest(bp.value.for0).ok, "wrong recipient rejected");
    Bundle tampered = bp.value.for0;
    tampered.payload[0] ^= 1;
    CHECK(!tp.client[0]->ingest(tampered).ok, "tampered payload rejected");
    Bundle wrong_bytes = bp.value.for0;
    wrong_bytes.header.set("bytes", static_cast<int64_t>(7));
    CHECK(!tp.client[0]->ingest(wrong_bytes).ok, "header tamper rejected by the tag");
    CHECK_OK(tp.client[0]->ingest(bp.value.for0), "ingest P0");
    CHECK(!tp.client[0]->ingest(bp.value.for0).ok, "duplicate material_id rejected");
    CHECK_OK(tp.client[1]->ingest(bp.value.for1), "ingest P1");
    CHECK(tp.ledger[0]->stock("beaver").first == 1, "ledger AVAILABLE after ingest");
    // partial consumption across bundles
    auto bp2 = tp.dealer->make_beaver(5);
    CHECK_OK(tp.client[0]->ingest(bp2.value.for0), "ingest second");
    CHECK_OK(tp.client[1]->ingest(bp2.value.for1), "ingest second P1");
    tp.client[0]->set_gate("g", 0);
    auto t1 = tp.client[0]->take_beaver(7);
    CHECK_OK(t1, "take 7 of 10");
    CHECK(tp.ledger[0]->state(bp.value.for0.header.get_string("material_id").value).value == "SPENT", "bundle SPENT on first use");
    CHECK(tp.ledger[0]->state(bp2.value.for0.header.get_string("material_id").value).value == "AVAILABLE", "second bundle untouched");
    auto t2 = tp.client[0]->take_beaver(6);  // 3 left + 3 of the second
    CHECK_OK(t2, "take across bundles");
    CHECK(tp.client[0]->pooled_count("beaver") == 2, "pool remainder");
    CHECK(tp.client[0]->bytes_consumed() == 24 * 13, "bytes consumed accounting");
    // contract mismatch
    DealerConfig other;
    other.numeric_contract_hash = "0000";
    Dealer d2(other, ll::Rng::from_seed(1), tp.key[0], tp.key[1]);
    auto bp3 = d2.make_beaver(1);
    CHECK(!tp.client[0]->ingest(bp3.value.for0).ok, "contract hash mismatch rejected");
    // typed material through the link (refill on miss)
    auto dr = tp.client[0]->take_drelu(3, 32, false);
    CHECK_OK(dr, "drelu via refill link");
    CHECK(dr.value.M() == 3 && !dr.value.with_product(), "drelu slice shape");
    auto dr1 = tp.client[1]->take_drelu(3, 32, false);
    CHECK_OK(dr1, "drelu P1 paired");
    CHECK(tp.link->generations() == 1, "paired requests trigger one generation");
    auto tm = tp.client[0]->take_trunc(4, 16, 40);
    CHECK_OK(tm, "trunc via refill");
    CHECK(tm.value.M() == 4 && tm.value.f == 16 && tm.value.L == 40, "trunc material params");
    auto rf = tp.client[0]->take_r2f(2, 68720050177ULL);
    CHECK_OK(rf, "r2f via refill");
    auto mt = tp.client[0]->take_matrix_triple(2, 3, 4);
    CHECK_OK(mt, "matrix triple via refill");
    CHECK(mt.value.A.size() == 6 && mt.value.B.size() == 12 && mt.value.C.size() == 8, "matrix triple shapes");
    // wmask mask + calls
    auto wm = tp.dealer->make_wmask_mask("W", 3, 2);
    CHECK_OK(wm, "wmask mask");
    CHECK_OK(tp.client[0]->ingest(wm.value.for0), "ingest mask");
    auto wc = tp.dealer->make_wmask_calls("W", 4, 2);
    CHECK_OK(wc, "wmask calls");
    CHECK_OK(tp.client[0]->ingest(wc.value.for0), "ingest calls");
    auto call = tp.client[0]->take_wmask_call("W", 4);
    CHECK_OK(call, "take wmask call");
    CHECK(call.value.a.size() == 12 && call.value.c.size() == 8, "wmask call shapes");
    CHECK(!tp.dealer->make_wmask_calls("nope", 1, 1).ok, "unknown mask refused");
    // vector share
    auto vs = tp.dealer->make_vector_share("v", {1, 2, 3});
    CHECK_OK(vs, "vector share");
    CHECK_OK(tp.client[0]->ingest(vs.value.for0), "ingest vector");
    CHECK_OK(tp.client[1]->ingest(vs.value.for1), "ingest vector P1");
    auto v0 = tp.client[0]->take_vector_share("v").value, v1 = tp.client[1]->take_vector_share("v").value;
    CHECK(v0[2] + v1[2] == 3, "vector share reconstructs");
    // material estimate: comparisons use P = 1 keys (1672 B at n = 32), relu and fused nodes P = 2 keys (2200 B)
    auto pair_of = [](const Value& est, const char* key, size_t i) { return est.get(key)->as_list().value->at(i).as_int().value; };
    auto est = material_estimate("drelu", {100}, 32, 16);
    CHECK_OK(est, "estimate");
    CHECK(pair_of(est.value, "drelu", 0) == 100, "estimate drelu count");
    CHECK(ll::drelu_bytes_per_element(32, false) == 1672 && ll::drelu_bytes_per_element(32, true) == 2200, "DReLU material sizes");
    CHECK(static_cast<uint64_t>(est.value.get_int("bytes").value) == 100 * ll::drelu_bytes_per_element(32, false), "estimate bytes (P = 1)");
    auto est_relu = material_estimate("relu", {100}, 32, 16);
    CHECK(est_relu.ok && static_cast<uint64_t>(est_relu.value.get_int("bytes").value) == 100 * ll::drelu_bytes_per_element(32, true),
          "relu estimate uses payload keys");
    // fused tournament, one node of 8 at the first level (public indices): 28 pairwise + 7 payload DReLUs
    auto est2 = material_estimate("argmax_row", {1, 8}, 32, 16, 8);
    CHECK_OK(est2, "estimate argmax k8");
    CHECK(pair_of(est2.value, "drelu_p1", 0) == 28 && pair_of(est2.value, "drelu_p2", 0) == 7 && pair_of(est2.value, "kary_select", 0) == 0,
          "argmax k8 over 8: 28 pairwise comparisons + 7 payload DReLUs, no Beaver select");
    CHECK(pair_of(est2.value, "drelu", 0) == 35 && pair_of(est2.value, "drelu", 1) == 28 * 1672 + 7 * 2200, "argmax k8 DReLU bytes");
    CHECK(est2.value.get_int("bytes").value == 28 * 1672 + 7 * 2200, "argmax k8 total material");
    // binary argmax over 8 with index: 7 comparisons; selects 4 (level 1, public index) + 2*2 + 2*1
    auto est3 = material_estimate("argmax_k", {3, 8}, 32, 16, 2);
    CHECK(est3.ok && pair_of(est3.value, "drelu_p1", 0) == 3 * 7 && pair_of(est3.value, "drelu_p2", 0) == 0 &&
              pair_of(est3.value, "kary_select", 0) == 3 * 10 && pair_of(est3.value, "kary_select", 1) == 3 * 10 * 24,
          "binary argmax: 7 comparisons and 10 Beaver selects per row");
    auto est4 = material_estimate("install_wmask", {6, 5}, 32, 16);
    CHECK(est4.ok && pair_of(est4.value, "wmask_mask", 0) == 30 && est4.value.get_int("bytes").value == 240, "install consumes the mask share");
    auto est5 = material_estimate("linear_wmask", {3, 6, 5}, 32, 16);
    CHECK(est5.ok && pair_of(est5.value, "wmask_call", 0) == 3 * 6 + 3 * 5, "linear_wmask consumes one wmask_call");
    CHECK(!material_estimate("nope", {1}, 32, 16).ok, "unknown op refused");
    // tournament_material: exactly the gates Party::tournament issues (python dealer.tournament_material)
    {
        auto t = tournament_material(5, 4, false);  // level 1: nodes (4), (1); level 2: node (2)
        CHECK(t.ok && t.value.drelu_p1 == 6 + 1 && t.value.drelu_p2 == 3 && t.value.selects == 1 && t.value.levels == 2 && t.value.rounds == 4,
              "tournament_material(5, 4)");
        auto ti = tournament_material(5, 4, true);  // the index is secret from level 2 on: one more select
        CHECK(ti.ok && ti.value.drelu_p1 == 7 && ti.value.drelu_p2 == 3 && ti.value.selects == 2, "tournament_material(5, 4, index)");
        auto t2 = tournament_material(50257, 2, true);
        CHECK(t2.ok && t2.value.drelu_p1 == 50256 && t2.value.drelu_p2 == 0 && t2.value.levels == 16 && t2.value.rounds == 32, "binary levels");
        CHECK(t2.value.selects == 50256 + (50256 - 25128), "binary argmax selects: value everywhere, index after the first level");
        auto t8 = tournament_material(50257, 8, true);
        CHECK(t8.ok && t8.value.levels == 6 && t8.value.rounds == 12, "8-ary tournament: 6 levels, 12 rounds");
        auto t1 = tournament_material(1, 4, true);
        CHECK(t1.ok && t1.value.drelu_p1 == 0 && t1.value.levels == 0, "single candidate needs no gate");
        CHECK(!tournament_material(0, 2, false).ok && tournament_material(4, 3, false).code.rfind("TOURNAMENT_PARAMS", 0) == 0,
              "tournament params fail closed");
    }
    for (int p = 0; p < 2; ++p) std::remove(tp.paths[p].c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// dealer: Python header spellings, kary_select, request spellings, burn on cancel
// ---------------------------------------------------------------------------
static bool shape_is(const Value& params, const char* key, std::vector<int64_t> want) {
    const Value* v = params.get(key);
    if (!v || !v->is_list()) return false;
    const List& l = *v->as_list().value;
    if (l.size() != want.size()) return false;
    for (size_t i = 0; i < want.size(); ++i)
        if (!l[i].is_int() || l[i].as_int().value != want[i]) return false;
    return true;
}

static int test_dealer_spellings_and_burn() {
    auto tpr = make_two_party("spell");
    CHECK_OK(tpr, "two party setup");
    TwoParty& tp = *tpr.value;
    // --- headers carry the Python spellings
    auto bv = tp.dealer->make_beaver(10);
    CHECK_OK(bv, "beaver");
    const Value& bparams = *bv.value.for0.header.get("params");
    CHECK(shape_is(bparams, "shape", {10}) && bparams.get_int("elements").value == 10, "beaver header: shape and elements");
    auto mt = tp.dealer->make_matrix_triples(2, 3, 4, 2);
    CHECK_OK(mt, "matrix triples");
    const Value& mparams = *mt.value.for0.header.get("params");
    CHECK(shape_is(mparams, "shape_a", {2, 3}) && shape_is(mparams, "shape_b", {3, 4}), "matrix_triple header: shape_a and shape_b");
    CHECK(!mparams.has("n") && !mparams.has("k") && !mparams.has("m"), "matrix_triple header drops the n / k / m spelling");
    auto wm = tp.dealer->make_wmask_mask("W", 3, 2);
    CHECK_OK(wm, "wmask mask");
    const Value& wparams = *wm.value.for0.header.get("params");
    CHECK(shape_is(wparams, "shape", {3, 2}) && wparams.get_string("mat_id").value == "W" && !wparams.has("d"),
          "wmask_mask header: mat_id and shape");
    CHECK(wparams.get_string("model_version").value == "model-spell", "wmask_mask header carries the model version");
    auto vs = tp.dealer->make_vector_share("v", {7, 8, 9});
    CHECK_OK(vs, "vector share");
    CHECK(shape_is(*vs.value.for0.header.get("params"), "shape", {3}), "vector_share header: shape");
    for (const char* k : {"profile_id", "setup_id", "model_version", "numeric_contract_hash", "recipient", "gate_id", "material_type",
                          "material_id", "count", "bytes", "params", "dealer_mode", "insecure"})
        CHECK(bv.value.for0.header.has(k), "bundle header field present");
    // --- both parties ingest; a multi-object bundle serves every object (bundle SPENT once)
    for (int p = 0; p < 2; ++p) {
        CHECK_OK(tp.client[p]->ingest(p == 0 ? mt.value.for0 : mt.value.for1), "ingest matrix triples");
        CHECK_OK(tp.client[p]->ingest(p == 0 ? wm.value.for0 : wm.value.for1), "ingest wmask mask");
        CHECK_OK(tp.client[p]->ingest(p == 0 ? vs.value.for0 : vs.value.for1), "ingest vector share");
        tp.client[p]->set_gate("g", 0);
    }
    {
        auto a0 = tp.client[0]->take_matrix_triple(2, 3, 4), a1 = tp.client[1]->take_matrix_triple(2, 3, 4);
        auto b0 = tp.client[0]->take_matrix_triple(2, 3, 4), b1 = tp.client[1]->take_matrix_triple(2, 3, 4);
        CHECK(a0.ok && a1.ok && b0.ok && b1.ok, "both triples of one bundle are served");
        std::vector<uint64_t> A(6), B(12), C(8), AB;
        for (size_t i = 0; i < 6; ++i) A[i] = b0.value.A[i] + b1.value.A[i];
        for (size_t i = 0; i < 12; ++i) B[i] = b0.value.B[i] + b1.value.B[i];
        for (size_t i = 0; i < 8; ++i) C[i] = b0.value.C[i] + b1.value.C[i];
        ring_matmul(A, 2, 3, B, 4, AB);
        CHECK(AB == C, "second matrix triple of the bundle reconstructs C = A B");
        CHECK(tp.link->generations() == 0, "no refill was needed");
        auto w0 = tp.client[0]->wmask_mask_share("W", 3, 2), w1 = tp.client[1]->wmask_mask_share("W", 3, 2);
        CHECK(w0.ok && w1.ok && w0.value.d == 3 && w0.value.m == 2, "mask parsed from shape");
        auto v0 = tp.client[0]->take_vector_share("v"), v1 = tp.client[1]->take_vector_share("v");
        CHECK(v0.ok && v1.ok && v0.value[1] + v1.value[1] == 8, "vector share parsed from shape");
    }
    // --- the earlier C++ spellings are still read (a dealer service that echoes them keeps working)
    {
        auto legacy = tp.dealer->make_matrix_triples(1, 2, 3, 1);
        CHECK_OK(legacy, "legacy source bundle");
        Value h = legacy.value.for0.header;
        h.set("params", dict({{"n", static_cast<int64_t>(1)}, {"k", static_cast<int64_t>(2)}, {"m", static_cast<int64_t>(3)}}));
        h.set("material_id", "legacy-matrix-triple-0");
        auto resealed = tp.dealer->seal(0, h, legacy.value.for0.payload);
        CHECK_OK(resealed, "reseal legacy spelling");
        CHECK_OK(tp.client[0]->ingest(resealed.value), "legacy {n, k, m} header accepted");
        CHECK(tp.client[0]->take_matrix_triple(1, 2, 3).ok, "legacy bundle pooled under the same key");
        auto lm = tp.dealer->make_wmask_mask("L", 2, 2);
        Value hm = lm.value.for0.header;
        hm.set("params", dict({{"mat_id", "L"}, {"d", static_cast<int64_t>(2)}, {"m", static_cast<int64_t>(2)}}));
        auto rs = tp.dealer->seal(0, hm, lm.value.for0.payload);
        CHECK(rs.ok && tp.client[0]->ingest(rs.value).ok, "legacy {mat_id, d, m} header accepted");
        Value bad = legacy.value.for0.header;
        bad.set("params", dict({{"shape_a", Value(List{Value(static_cast<int64_t>(1)), Value(static_cast<int64_t>(2))})},
                                {"shape_b", Value(List{Value(static_cast<int64_t>(5)), Value(static_cast<int64_t>(3))})}}));
        bad.set("material_id", "bad-matrix-triple-0");
        auto rb = tp.dealer->seal(0, bad, legacy.value.for0.payload);
        CHECK(rb.ok && tp.client[0]->ingest(rb.value).code.rfind("MATERIAL_PARAMS", 0) == 0, "inconsistent shape_a / shape_b refused");
        Value badb = bv.value.for0.header;
        badb.set("params", dict({{"shape", Value(List{Value(static_cast<int64_t>(11))})}}));
        badb.set("material_id", "bad-beaver-0");
        auto rbb = tp.dealer->seal(0, badb, bv.value.for0.payload);
        CHECK(rbb.ok && tp.client[0]->ingest(rbb.value).code.rfind("MATERIAL_PARAMS", 0) == 0, "Beaver shape that disagrees with count refused");
    }
    // --- request dispatch with the Python spellings
    {
        Value sh23 = Value(List{Value(static_cast<int64_t>(2)), Value(static_cast<int64_t>(3))});
        auto r1 = tp.dealer->make("beaver", dict({{"shape", sh23}}), 2);
        CHECK(r1.ok && r1.value.for0.header.get_int("count").value == 12 && r1.value.for0.payload.size() == 3 * 12 * 8,
              "beaver request: count bundles of shape -> 12 elements");
        auto r2 = tp.dealer->make("matrix_triple", dict({{"shape_a", sh23}, {"shape_b", Value(List{Value(static_cast<int64_t>(3)), Value(static_cast<int64_t>(1))})}}), 1);
        CHECK(r2.ok && r2.value.for0.payload.size() == 8 * (6 + 3 + 2), "matrix_triple request with shape_a / shape_b");
        auto r3 = tp.dealer->make("drelu", dict({{"M", static_cast<int64_t>(4)}, {"n_bits", static_cast<int64_t>(32)}, {"with_product", false}}), 2);
        CHECK(r3.ok && r3.value.for0.header.get_int("count").value == 8, "drelu request: count batches of M gates");
        auto r4 = tp.dealer->make("kary_select", dict({{"shape", Value(List{Value(static_cast<int64_t>(5))})}}), 1);
        CHECK(r4.ok && r4.value.for0.header.get_string("material_type").value == "kary_select", "kary_select request");
        auto r5 = tp.dealer->make("matrix_triple", dict({{"n", static_cast<int64_t>(1)}, {"k", static_cast<int64_t>(1)}, {"m", static_cast<int64_t>(1)}}), 1);
        CHECK(r5.ok, "legacy matrix_triple request still dispatched");
        CHECK(tp.dealer->make("drelu", dict({{"M", static_cast<int64_t>(0)}, {"n_bits", static_cast<int64_t>(32)}}), 1).code.rfind("MATERIAL_PARAMS", 0) == 0,
              "M = 0 refused");
        CHECK(tp.dealer->make("nope", dict({}), 1).code.rfind("DEALER_UNKNOWN_TYPE", 0) == 0, "unknown type refused");
    }
    // --- kary_select: own material type and pool; Beaver correlation holds across the two halves
    {
        auto ks = tp.dealer->make_kary_select(6);
        CHECK_OK(ks, "make kary_select");
        CHECK(ks.value.for0.header.get_string("material_type").value == "kary_select", "kary_select type in the header");
        CHECK_OK(tp.client[0]->ingest(ks.value.for0), "ingest kary P0");
        CHECK_OK(tp.client[1]->ingest(ks.value.for1), "ingest kary P1");
        CHECK(tp.ledger[0]->stock("kary_select").first == 1 && tp.ledger[0]->stock("kary_select").second == 144, "ledger stock by type kary_select");
        CHECK(tp.client[0]->pooled_count("kary_select") == 6 && tp.client[0]->pooled_count("beaver") == 0, "kary pool separate from beaver");
        auto k0 = tp.client[0]->take_kary_select(4), k1 = tp.client[1]->take_kary_select(4);
        CHECK(k0.ok && k1.ok, "take kary_select");
        bool okk = true;
        for (size_t i = 0; i < 4; ++i) okk = okk && ((k0.value.a[i] + k1.value.a[i]) * (k0.value.b[i] + k1.value.b[i]) == k0.value.c[i] + k1.value.c[i]);
        CHECK(okk, "kary_select triples are Beaver triples");
        CHECK(tp.client[0]->pooled_count("kary_select") == 2, "kary pool remainder");
        uint64_t gen = tp.link->generations();
        auto k2 = tp.client[0]->take_kary_select(5), k3 = tp.client[1]->take_kary_select(5);
        CHECK(k2.ok && k3.ok && tp.link->generations() == gen + 1, "kary_select refill through the link (one paired generation)");
        okk = true;
        for (size_t i = 0; i < 5; ++i) okk = okk && ((k2.value.a[i] + k3.value.a[i]) * (k2.value.b[i] + k3.value.b[i]) == k2.value.c[i] + k3.value.c[i]);
        CHECK(okk, "refilled kary_select triples pair across the parties");
    }
    // --- burn on cancellation: RESERVED and untouched AVAILABLE bundles are burned through the ledger
    {
        DealerClient& c0 = *tp.client[0];
        auto b1 = tp.dealer->make_beaver(8), b2 = tp.dealer->make_beaver(4), b3 = tp.dealer->make_drelu(5, 32, false);
        CHECK(b1.ok && b2.ok && b3.ok, "bundles for the burn test");
        CHECK_OK(c0.ingest(b1.value.for0), "ingest b1");
        CHECK_OK(c0.ingest(b2.value.for0), "ingest b2");
        CHECK_OK(c0.ingest(b3.value.for0), "ingest b3");
        std::string id1 = b1.value.for0.header.get_string("material_id").value;
        std::string id2 = b2.value.for0.header.get_string("material_id").value;
        std::string id3 = b3.value.for0.header.get_string("material_id").value;
        CHECK(c0.take_beaver(3).ok && tp.ledger[0]->state(id1).value == "SPENT", "first bundle partially consumed (SPENT)");
        CHECK_OK(tp.ledger[0]->reserve({id2}, "epoch-9", 9), "reserve for a gate that will not run");
        CHECK(c0.reserved_ids() == std::vector<std::string>{id2}, "reserved_ids lists the RESERVED bundle");
        uint64_t burned_before = tp.ledger[0]->bytes_burned();
        auto rep = c0.burn_outstanding("CANCEL");
        CHECK_OK(rep, "burn_outstanding");
        CHECK(rep.value.reserved_ids == 1 && rep.value.available_ids >= 1, "one RESERVED and the untouched AVAILABLE bundles burned");
        CHECK(tp.ledger[0]->state(id2).value == "BURNED" && tp.ledger[0]->state(id3).value == "BURNED", "RESERVED and AVAILABLE became BURNED");
        CHECK(tp.ledger[0]->state(id1).value == "SPENT", "a SPENT bundle stays SPENT (its remainder is only dropped)");
        CHECK(rep.value.bytes_dropped >= 24 * 5, "the unconsumed remainder of the SPENT bundle is reported as dropped");
        CHECK(rep.value.bytes_burned == tp.ledger[0]->bytes_burned() - burned_before && rep.value.bytes_burned >= 96 + 5 * 1672,
              "burned bytes are the ledger bytes of the burned bundles");
        CHECK(c0.reserved_ids().empty(), "no RESERVED id is left");
        CHECK(c0.pooled_count("beaver") == 0 && c0.pooled_count("kary_select") == 0 && c0.pooled_count("drelu:32|1") == 0, "one-time pools are empty");
        CHECK(c0.has_wmask_mask("W") && c0.has_vector_share("v"), "installed masks and vector shares are kept");
        CHECK(!tp.ledger[0]->reserve({id2}, "g", 1).ok, "a burned bundle is never served again");
        auto rep2 = c0.burn_outstanding("CANCEL");
        CHECK(rep2.ok && rep2.value.reserved_ids == 0 && rep2.value.available_ids == 0 && rep2.value.bytes_burned == 0, "burn is idempotent");
        // a resumed generation drops the masks too and tags its requests
        auto rep3 = c0.burn_outstanding("RESUME", true);
        CHECK(rep3.ok && !c0.has_wmask_mask("W"), "resume drops the per-matrix masks");
        c0.set_generation(1);
        tp.client[1]->set_generation(1);
        CHECK_OK(tp.client[1]->burn_outstanding("RESUME", true), "peer drops its masks");
        auto nm0 = c0.wmask_mask_share("W", 3, 2), nm1 = tp.client[1]->wmask_mask_share("W", 3, 2);
        CHECK(nm0.ok && nm1.ok, "generation 1 gets a fresh mask for the same matrix");
        CHECK(tp.dealer->make_wmask_mask("W", 3, 2, "*", 1).code.rfind("DEALER_MASK_EXISTS", 0) == 0, "one mask per matrix and generation");
        // journal replay: nothing RESERVED survives
        Value binding = tp.ledger[0]->binding();
        MaterialLedger replay(tp.paths[0], binding);
        auto rec = replay.recover();
        CHECK(rec.ok && rec.value.get_int("reserved_burned").value == 0, "journal replay finds no RESERVED entry");
    }
    for (int p = 0; p < 2; ++p) std::remove(tp.paths[p].c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// polynomial contracts
// ---------------------------------------------------------------------------
static int test_poly() {
    auto g = poly_contract_by_name("gelu_clip12");
    CHECK_OK(g, "gelu_clip12 contract");
    CHECK(g.value->degree == 12 && g.value->coeffs.size() == 13 && !g.value->even_in_u, "gelu degree 12 plain fit");
    CHECK(std::fabs(g.value->max_abs_err - 4.84e-3) < 2e-5, "gelu_clip12 max_abs_err 4.84e-3");
    CHECK(std::fabs(g.value->coeffs[1] - 2.5) < 1e-9, "gelu c1 = 2.5 (GELU'(0) R)");
    CHECK(std::fabs(g.value->coeffs[2] - 9.4289812820942807) < 1e-8, "gelu c2 matches numpy lstsq");
    auto s = poly_contract_by_name("silu_clip12");
    CHECK_OK(s, "silu contract");
    CHECK(std::fabs(s.value->max_abs_err - 1.277e-2) < 2e-5, "silu_clip12 max_abs_err 1.28e-2");
    auto r = poly_contract_by_name("recip_norm");
    CHECK_OK(r, "recip contract");
    CHECK(r.value->affine && r.value->max_abs_err < 1e-4 && std::fabs(r.value->coeffs[0] - 1.3334034202395495) < 1e-9,
          "recip_norm fit");
    auto q = poly_contract_by_name("rsqrt_norm");
    CHECK_OK(q, "rsqrt contract");
    CHECK(q.value->max_abs_err < 1.3e-3, "rsqrt_norm fit");
    CHECK(!poly_contract_by_name("nope").ok, "unknown contract refused");
    return 0;
}

// ---------------------------------------------------------------------------
// two-party gadget loop
// ---------------------------------------------------------------------------
using Body = std::function<Expected<Value>(Party&, int)>;

static std::array<Expected<Value>, 2> run_both(TwoParty& tp, const Body& body) {
    std::array<Expected<Value>, 2> out;
    auto run = [&](int p) {
        ll::Rng rng = ll::Rng::from_seed(1000 + static_cast<uint64_t>(p));
        ReleasePolicy rel(1, {"token", "class"}, 4);
        tp.client[p]->set_gate("test", 0);
        Party party(p, *tp.ch[p], *tp.client[p], tp.contract, rel, rng);
        out[p] = body(party, p);
        if (!out[p].ok) tp.ch[p]->close();  // unblock the peer (fail closed)
    };
    std::thread t0(run, 0);
    std::thread t1(run, 1);
    t0.join();
    t1.join();
    return out;
}

static std::vector<double> rand_vec(ll::Rng& rng, size_t n, double lo, double hi) {
    std::vector<double> v(n);
    for (auto& x : v) x = lo + (hi - lo) * ll::rng_uniform(rng);
    return v;
}

// Shares a public-to-the-test plaintext as P1's input and returns the shares.
static Expected<Mat> share_plain(Party& p, const std::vector<double>& x, size_t rows, size_t cols) {
    if (p.party() == 1) {
        OC2_TRY(enc, p.encode(x, rows, cols));
        return p.share_input(&enc, 1, rows, cols);
    }
    return p.share_input(nullptr, 1, rows, cols);
}

static int64_t rounds_of(const Party& p, const std::string& kind) {
    for (auto it = p.stats.rbegin(); it != p.stats.rend(); ++it)
        if (it->kind == kind) return static_cast<int64_t>(it->rounds);
    return -1;
}

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Exact accounting of one unmasked fused tournament call (paper, Section 5.3): the
// rounds, the channel bytes (both directions) and the material bytes must equal what the Python
// realization issues. Per level two openings, each one flat u64 array (frame = 8-byte prefix + 11
// bytes of array header + 8 bytes per element); opened elements per row: one per pairwise
// comparison, one per payload DReLU, two per Beaver select.
static bool tournament_accounting_ok(const GadgetStat& st, size_t R, size_t n, int k, bool with_index) {
    auto t = tournament_material(n, k, with_index);
    if (!t.ok) return false;
    const uint64_t elements = R * (t.value.drelu_p1 + t.value.drelu_p2 + 2 * t.value.selects);
    const uint64_t bytes = 2 * (t.value.rounds * 19 + 8 * elements);
    const uint64_t material = R * (t.value.drelu_p1 * ll::drelu_bytes_per_element(32, false) +
                                   t.value.drelu_p2 * ll::drelu_bytes_per_element(32, true) + t.value.selects * 24);
    return st.rounds == t.value.rounds && st.bytes == bytes && st.material_bytes == material;
}

// Cross-language pins: (rounds, channel bytes, material bytes) of python Party._tournament measured
// on 2026-10-10 with the in-process harness of python/tests/test_party.py for R = 3 rows (max_rows
// when with_index is false, argmax_row(release=False) otherwise). Material is pinned where the
// Python harness served every comparison from its P = 1 pool (0 = not pinned).
struct TournamentPin {
    size_t n;
    int k;
    bool with_index;
    uint64_t rounds, bytes, material;
};
static const TournamentPin kPythonTournamentPins[] = {
    {5, 2, false, 6, 804, 20352},     {5, 2, true, 6, 996, 20496},     {5, 4, false, 4, 728, 54984},
    {5, 4, true, 4, 824, 55056},      {5, 8, false, 2, 748, 76560},    {5, 8, true, 2, 748, 76560},
    {17, 2, false, 10, 2684, 81408},  {17, 2, true, 10, 3452, 81984},  {17, 4, false, 6, 2532, 254568},
    {17, 4, true, 6, 2772, 274440},   {17, 8, false, 4, 3752, 401544}, {17, 8, true, 4, 3848, 414744},
    {64, 2, false, 12, 9528, 320544}, {64, 2, true, 12, 12504, 322776}, {64, 4, false, 6, 9300, 0},
    {64, 4, true, 6, 10020, 0},       {64, 8, false, 4, 15272, 0},     {64, 8, true, 4, 15608, 0}};

// 1 when a pin exists and matches, 0 when no pin exists, -1 on a mismatch.
static int python_pin_check(const GadgetStat& st, size_t n, int k, bool with_index) {
    for (const TournamentPin& pin : kPythonTournamentPins) {
        if (pin.n != n || pin.k != k || pin.with_index != with_index) continue;
        if (st.rounds != pin.rounds || st.bytes != pin.bytes) return -1;
        if (pin.material && st.material_bytes != pin.material) return -1;
        return 1;
    }
    return 0;
}

static Expected<Value> gadget_body(Party& p, int party) {
    (void)party;
    ll::Rng rng = ll::Rng::from_seed(7);  // identical plaintexts on both parties (public test inputs)
    Value res = dict({});
    const int f = p.f();
    const double ulp = std::ldexp(1.0, -f);
    // --- Beaver / mul_fxp
    {
        auto x = rand_vec(rng, 64, -50, 50), y = rand_vec(rng, 64, -50, 50);
        OC2_TRY(sx, share_plain(p, x, 8, 8));
        OC2_TRY(sy, share_plain(p, y, 8, 8));
        OC2_TRY(sz, p.mul_fxp(sx, sy));
        OC2_TRY(z, p.reveal(sz));
        auto zd = p.decode(z);
        std::vector<double> ref(64);
        // reference on the encoded (quantised) inputs: the gadget's own error is the truncation
        for (size_t i = 0; i < 64; ++i) ref[i] = p.dec(p.enc(x[i])) * p.dec(p.enc(y[i]));
        res.set("mul_fxp_err_ulp", max_abs_diff(zd, ref) / ulp);
        size_t before = p.all_stats.size();
        OC2_TRY(bz, p.beaver_mul(sx, sy));
        (void)bz;
        (void)before;
    }
    // --- faithful truncation (exact), including magnitudes far beyond the local gate's safe range
    {
        std::vector<uint64_t> vals;
        std::vector<int64_t> ref;
        for (int i = 0; i < 48; ++i) {
            int bits = 4 + (i * 56) / 48;  // magnitudes 2^4 .. 2^59
            uint64_t mag = (rng.next_u64() >> (64 - bits)) | (1ULL << (bits - 1));
            int64_t v = (i % 3 == 1) ? -static_cast<int64_t>(mag) : static_cast<int64_t>(mag);
            if (i % 3 == 1 && bits > 39) v = static_cast<int64_t>(mag);  // negatives stay inside [-2^39, 2^39)
            vals.push_back(static_cast<uint64_t>(v));
            ref.push_back(v >> f);  // arithmetic shift = floor division for negatives
        }
        vals.push_back(static_cast<uint64_t>(-13LL));
        ref.push_back(-1);
        vals.push_back(0);
        ref.push_back(0);
        Mat plain = Mat::from_vector(vals, 1, vals.size());
        Mat sx = p.party() == 1 ? plain : Mat(1, vals.size());
        OC2_TRY(sh, p.share_input(p.party() == 1 ? &sx : nullptr, 1, 1, vals.size()));
        OC2_TRY(st, p.trunc_faithful(sh));
        OC2_TRY(t, p.reveal(st));
        int mism = 0;
        for (size_t i = 0; i < vals.size(); ++i)
            if (static_cast<int64_t>(t.a[i]) != ref[i]) ++mism;
        res.set("trunc_faithful_mismatches", static_cast<int64_t>(mism));
        res.set("trunc_faithful_rounds", rounds_of(p, "trunc_faithful"));
    }
    // --- matmul_shared
    {
        auto X = rand_vec(rng, 24, -3, 3), Y = rand_vec(rng, 30, -3, 3);
        for (auto& v : X) v = p.dec(p.enc(v));
        for (auto& v : Y) v = p.dec(p.enc(v));
        OC2_TRY(sX, share_plain(p, X, 4, 6));
        OC2_TRY(sY, share_plain(p, Y, 6, 5));
        OC2_TRY(sZ, p.matmul_shared(sX, sY));
        OC2_TRY(Z, p.reveal(sZ));
        std::vector<double> ref(20, 0.0);
        for (size_t i = 0; i < 4; ++i)
            for (size_t k = 0; k < 6; ++k)
                for (size_t j = 0; j < 5; ++j) ref[i * 5 + j] += X[i * 6 + k] * Y[k * 5 + j];
        res.set("matmul_err", max_abs_diff(p.decode(Z), ref));
        res.set("matmul_rounds", rounds_of(p, "matmul_shared"));
    }
    // --- linear_generic and linear_wmask (with a real install), bias included
    {
        auto X = rand_vec(rng, 18, -2, 2), W = rand_vec(rng, 30, -1, 1), b = rand_vec(rng, 5, -1, 1);
        for (auto& v : X) v = p.dec(p.enc(v));
        for (auto& v : W) v = p.dec(p.enc(v));
        for (auto& v : b) v = p.dec(p.enc(v));
        std::vector<double> ref(15, 0.0);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 5; ++j) {
                for (size_t k = 0; k < 6; ++k) ref[i * 5 + j] += X[i * 6 + k] * W[k * 5 + j];
                ref[i * 5 + j] += b[j];
            }
        Mat Wenc, benc;
        if (p.party() == 0) {
            OC2_TRY(we, p.encode(W, 6, 5));
            OC2_TRY(be, p.encode(b, 1, 5));
            Wenc = we;
            benc = be;
        }
        OC2_TRY(Ws, p.share_input(p.party() == 0 ? &Wenc : nullptr, 0, 6, 5));
        OC2_TRY(bs, p.share_input(p.party() == 0 ? &benc : nullptr, 0, 1, 5));
        p.set_vector_share("lin", Ws);
        p.set_vector_share("lin.b", bs);
        OC2_TRY(sX, share_plain(p, X, 3, 6));
        OC2_TRY(yg, p.linear_generic(sX, "lin"));
        OC2_TRY(Yg, p.reveal(yg));
        res.set("linear_generic_err", max_abs_diff(p.decode(Yg), ref));
        // wmask: the mask comes from the dealer, E = W - B from the install protocol
        auto lw_before = p.linear_wmask(sX, "lin");
        if (lw_before.ok) return fail("linear_wmask before install must fail");
        res.set("wmask_before_install_code", lw_before.code.substr(0, lw_before.code.find(':')));
        OC2_TRY(inst, p.install_wmask("lin", p.party() == 0 ? &Wenc : nullptr));
        res.set("install_rounds", inst.get_int("rounds").value);
        // measured counters of the install gadget (material: the 8 d m bytes of the mask share)
        res.set("install_material", static_cast<int64_t>(p.stats.back().material_bytes));
        res.set("install_stat_kind", p.stats.back().kind);
        res.set("install_stat_rounds", static_cast<int64_t>(p.stats.back().rounds));
        res.set("install_stat_bytes", static_cast<int64_t>(p.stats.back().bytes));
        res.set("install_skipped_second", p.install_wmask("lin", p.party() == 0 ? &Wenc : nullptr).value.get("skipped")->as_bool().value);
        if (!p.installed("lin")) return fail("installed() false after install");
        OC2_TRY(yw, p.linear_wmask(sX, "lin"));
        OC2_TRY(Yw, p.reveal(yw));
        res.set("linear_wmask_err", max_abs_diff(p.decode(Yw), ref));
        res.set("linear_wmask_bytes", static_cast<int64_t>(p.stats.back().bytes));
    }
    // --- drelu / relu / compare_public / select (exact)
    {
        auto x = rand_vec(rng, 40, -100, 100);
        x[0] = 0.0;
        x[1] = -ulp;
        x[2] = ulp;
        OC2_TRY(sx, share_plain(p, x, 1, 40));
        OC2_TRY(b, p.drelu(sx));
        OC2_TRY(B, p.reveal(b));
        OC2_TRY(r, p.relu(sx));
        OC2_TRY(R, p.reveal(r));
        OC2_TRY(c, p.compare_public(sx, 10.0, "lt"));
        OC2_TRY(C, p.reveal(c));
        int mism = 0;
        auto rd = p.decode(R);
        for (size_t i = 0; i < 40; ++i) {
            double xe = p.dec(p.enc(x[i]));
            if (B.a[i] != (xe >= 0 ? 1u : 0u)) ++mism;
            if (std::fabs(rd[i] - std::max(xe, 0.0)) > 0) ++mism;
            if (C.a[i] != (xe < 10.0 ? 1u : 0u)) ++mism;
        }
        res.set("drelu_relu_mismatches", static_cast<int64_t>(mism));
        res.set("drelu_rounds", rounds_of(p, "drelu"));
        OC2_TRY(sa, share_plain(p, rand_vec(rng, 40, -5, 5), 1, 40));
        OC2_TRY(sel, p.select(b, sa, sx));
        OC2_TRY(S, p.reveal(sel));
        auto ad = p.decode(p.party() == 0 ? S : S);  // same on both
        (void)ad;
        res.set("select_ok", true);
    }
    // --- k-ary max / argmax: identical for k in {2, 4, 8}, lowest index on ties
    {
        std::vector<size_t> lengths = {1, 2, 3, 5, 8, 17, 32, 64};
        int64_t mism_k = 0, mism_ref = 0, mism_max = 0, acct_mism = 0, acct_calls = 0, pin_mism = 0, pin_hits = 0;
        for (size_t n : lengths) {
            // rows with deliberate ties: values quantised to a coarse grid
            auto raw = rand_vec(rng, 3 * n, -40, 40);
            for (auto& v : raw) v = std::round(v / 8.0) * 8.0;
            if (n >= 3) {
                raw[2] = raw[0];  // tie with the first element in row 0
                raw[n + n - 1] = 1000.0;  // large unique max at the end of row 1
                raw[n] = 1000.0;          // and a tie at index 0 of row 1 -> lowest index 0 wins
                raw[2 * n + 1] = 500.0;   // row 2: max at index 1
            }
            OC2_TRY(sx, share_plain(p, raw, 3, n));
            std::vector<uint64_t> idx_by_k[3];
            std::vector<double> max_by_k[3];
            int ks[3] = {2, 4, 8};
            for (int ki = 0; ki < 3; ++ki) {
                OC2_TRY(mx, p.max_rows(sx, nullptr, ks[ki]));
                if (!tournament_accounting_ok(p.stats.back(), 3, n, ks[ki], false)) ++acct_mism;
                int pin = python_pin_check(p.stats.back(), n, ks[ki], false);
                if (pin < 0) ++pin_mism;
                if (pin > 0) ++pin_hits;
                OC2_TRY(MX, p.reveal(mx));
                max_by_k[ki] = p.decode(MX);
                OC2_TRY(ix, p.argmax_row(sx, ks[ki], false));
                if (!tournament_accounting_ok(p.stats.back(), 3, n, ks[ki], true)) ++acct_mism;
                pin = python_pin_check(p.stats.back(), n, ks[ki], true);
                if (pin < 0) ++pin_mism;
                if (pin > 0) ++pin_hits;
                ++acct_calls;
                OC2_TRY(IX, p.reveal(ix));
                idx_by_k[ki] = IX.a;
            }
            for (size_t r = 0; r < 3; ++r) {
                std::vector<double> row(raw.begin() + static_cast<std::ptrdiff_t>(r * n), raw.begin() + static_cast<std::ptrdiff_t>((r + 1) * n));
                size_t ref_idx = argmax_ref(row);
                double ref_max = row[ref_idx];
                for (int ki = 0; ki < 3; ++ki) {
                    if (idx_by_k[ki][r] != idx_by_k[0][r]) ++mism_k;
                    if (idx_by_k[ki][r] != ref_idx) ++mism_ref;
                    if (std::fabs(max_by_k[ki][r] - ref_max) > 0) ++mism_max;
                }
            }
        }
        res.set("argmax_k_disagreements", mism_k);
        res.set("argmax_ref_mismatches", mism_ref);
        res.set("max_mismatches", mism_max);
        res.set("tournament_accounting_mismatches", acct_mism);
        res.set("tournament_accounting_calls", acct_calls);
        res.set("tournament_python_pin_mismatches", pin_mism);
        res.set("tournament_python_pin_hits", pin_hits);
        // masked max (softmax style causal mask)
        auto raw = rand_vec(rng, 2 * 6, -10, 10);
        raw[5] = 100.0;  // masked out in row 0 below
        std::vector<uint8_t> mask = {1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        OC2_TRY(sx, share_plain(p, raw, 2, 6));
        OC2_TRY(mx, p.max_rows(sx, &mask, 4));
        GadgetStat masked_stat = p.stats.back();
        OC2_TRY(MX, p.reveal(mx));
        auto md = p.decode(MX);
        double r0 = std::max({raw[0], raw[1], raw[2]}), r1 = *std::max_element(raw.begin() + 6, raw.end());
        res.set("masked_max_err", std::max(std::fabs(md[0] - r0), std::fabs(md[1] - r1)));
        // Public pairs cost no comparison: level 1 opens 10 live pairs, then 6 payload DReLUs and one
        // Beaver select (2 x 2 elements); level 2 opens 1 live pair, then one Beaver select.
        res.set("masked_max_rounds", static_cast<int64_t>(masked_stat.rounds));
        res.set("masked_max_bytes", static_cast<int64_t>(masked_stat.bytes));
        res.set("masked_max_material", static_cast<int64_t>(masked_stat.material_bytes));
        // --- magnitude bound of the fused nodes: differences just inside the bound of each k
        // are still exact (2^11 real units for nodes of 8, 2^12 for nodes of 4, 2^15 for binary nodes)
        int64_t edge_mism = 0;
        const int eks[3] = {8, 4, 2};
        const double amp[3] = {1023.75, 2047.75, 16383.75};
        for (int e = 0; e < 3; ++e) {
            const size_t n = 16;
            std::vector<double> v(2 * n);
            for (size_t j = 0; j < n; ++j) v[j] = (j % 3 == 1) ? amp[e] : -amp[e];          // ties at the maximum: index 1 wins
            for (size_t j = 0; j < n; ++j) v[n + j] = amp[e] * (2.0 * ll::rng_uniform(rng) - 1.0);
            v[n + 13] = amp[e];
            v[n + 2] = -amp[e];
            OC2_TRY(se, share_plain(p, v, 2, n));
            OC2_TRY(me, p.max_rows(se, nullptr, eks[e]));
            OC2_TRY(ME, p.reveal(me));
            OC2_TRY(ie, p.argmax_row(se, eks[e], false));
            OC2_TRY(IE, p.reveal(ie));
            auto med = p.decode(ME);
            for (size_t r = 0; r < 2; ++r) {
                std::vector<double> row(n);
                for (size_t j = 0; j < n; ++j) row[j] = p.dec(p.enc(v[r * n + j]));
                size_t ref_idx = argmax_ref(row);
                if (IE.a[r] != ref_idx) ++edge_mism;
                if (med[r] != row[ref_idx]) ++edge_mism;
            }
        }
        res.set("kary_bound_edge_mismatches", edge_mism);
        res.set("kary_bound_g8", static_cast<int64_t>(Party::kary_difference_bound(32, 8)));
        res.set("kary_bound_g4", static_cast<int64_t>(Party::kary_difference_bound(32, 3)));
        res.set("kary_bound_g2", static_cast<int64_t>(Party::kary_difference_bound(32, 2)));
    }
    // --- trunc_local: the explicit arithmetic shift equals the lowlevel SecureML gate on both parties
    {
        std::vector<uint64_t> v = {0, 1, 2, 7, 8, 65535, 65536, 65537, static_cast<uint64_t>(-1LL), static_cast<uint64_t>(-8LL),
                                   static_cast<uint64_t>(-13LL), static_cast<uint64_t>(-65536LL), static_cast<uint64_t>(-65537LL),
                                   1ULL << 63, (1ULL << 63) - 1, (1ULL << 63) + 1, 0xffffffffffffffffULL};
        for (int i = 0; i < 32; ++i) v.push_back(rng.next_u64());
        int64_t mism = 0;
        for (int f : {1, 3, 16, 31, 62, 63}) {
            Mat t = p.trunc_local(Mat::row_vector(v), f);
            auto ref = lowlevel::trunc::trunc_local(v, f, p.party());
            if (!ref.ok) { ++mism; continue; }
            for (size_t i = 0; i < v.size(); ++i)
                if (t.a[i] != ref.value[i]) ++mism;
        }
        res.set("trunc_local_mismatches", mism);
        // floor(-13 / 8) = -2 on P0; P1 computes -floor(13 / 8) = -1
        res.set("trunc_local_m13", static_cast<int64_t>(p.trunc_local(Mat(1, 1, static_cast<uint64_t>(-13LL)), 3).a[0]));
        res.set("trunc_local_f0_identity", p.trunc_local(Mat(1, 1, 12345), 0).a[0] == 12345);
    }
    // --- gelu / silu
    {
        auto x = rand_vec(rng, 200, -8, 8);
        OC2_TRY(sx, share_plain(p, x, 4, 50));
        OC2_TRY(g, p.gelu(sx));
        OC2_TRY(G, p.reveal(g));
        OC2_TRY(s, p.silu(sx));
        OC2_TRY(S, p.reveal(s));
        auto gd = p.decode(G), sd = p.decode(S);
        double ge = 0, se = 0;
        for (size_t i = 0; i < 200; ++i) {
            ge = std::max(ge, std::fabs(gd[i] - gelu_ref(x[i])));
            se = std::max(se, std::fabs(sd[i] - silu_ref(x[i])));
        }
        res.set("gelu_err", ge);
        res.set("silu_err", se);
        res.set("gelu_rounds", rounds_of(p, "gelu_clip12"));
    }
    // --- exp_neg
    {
        auto z = rand_vec(rng, 64, -16, 0);
        z[0] = 0.0;
        z[1] = -16.0;
        z[2] = -0.5;
        OC2_TRY(sz, share_plain(p, z, 1, 64));
        OC2_TRY(e, p.exp_neg(sz));
        OC2_TRY(E, p.reveal(e));
        auto ed = p.decode(E);
        double worst = 0.0;
        for (size_t i = 0; i < 64; ++i) {
            double ref = std::exp(z[i]);
            double tol = 1e-2 * ref + 3e-4;
            worst = std::max(worst, std::fabs(ed[i] - ref) / tol);
        }
        res.set("exp_neg_err_ratio", worst);
        res.set("exp_neg_rounds", rounds_of(p, "exp_neg"));
    }
    // --- recip and rsqrt
    {
        auto S = rand_vec(rng, 32, 1.0, 4000.0);
        S[0] = 1.0;
        S[1] = 4095.0;
        S[2] = 2.0;
        OC2_TRY(sS, share_plain(p, S, 1, 32));
        OC2_TRY(r, p.recip(sS, 10));
        OC2_TRY(R, p.reveal(r));
        double worst = 0.0;
        for (size_t i = 0; i < 32; ++i) {
            double got = static_cast<double>(static_cast<int64_t>(R.a[i])) / std::ldexp(1.0, f + 10);
            worst = std::max(worst, std::fabs(got * S[i] - 1.0));
        }
        res.set("recip_rel_err", worst);
        res.set("recip_rounds", rounds_of(p, "recip"));
        // domain: V in [4^-7, 2^15) (the DCF bound of the numeric contract caps compared values at 2^15)
        auto V = rand_vec(rng, 32, 1e-4, 3.0e4);
        V[0] = 1e-4;
        V[1] = 1.0;
        V[2] = 3.2e4;
        for (size_t i = 3; i < 16; ++i) V[i] = std::pow(10.0, -4.0 + 8.4 * static_cast<double>(i - 3) / 12.0);
        OC2_TRY(sV, share_plain(p, V, 1, 32));
        OC2_TRY(q, p.rsqrt(sV, 4));
        OC2_TRY(Q, p.reveal(q));
        worst = 0.0;
        for (size_t i = 0; i < 32; ++i) {
            double got = static_cast<double>(static_cast<int64_t>(Q.a[i])) / std::ldexp(1.0, f + 4);
            double ve = p.dec(p.enc(V[i]));
            worst = std::max(worst, std::fabs(got * std::sqrt(ve) - 1.0));
        }
        res.set("rsqrt_rel_err", worst);
        res.set("rsqrt_rounds", rounds_of(p, "rsqrt"));
        // The inverse square root product (paper, Appendix D, Disclosure): the normalisation
        // product V * s2 of rsqrt on 2^18 elements, through the real
        // mul_fxp. With s2 at scale 2^(f+e2), e2 = rsqrt_scale_bits, the product is below 2^34 and
        // the local truncation never wraps (with e2 = f it was a 2^48 product: about 2.6 gross
        // errors expected on this many elements, each off by 2^32).
        {
            const NumericContract& c = p.contract();
            const int e2 = rsqrt_scale_bits(c);
            const size_t N = static_cast<size_t>(1) << 18;
            ll::Rng vr = ll::Rng::from_seed(4711);   // public test operands, identical on both parties
            Mat sx(1, N), ss(1, N);
            std::vector<int64_t> exact(N);
            uint64_t max_prod = 0;
            for (size_t i = 0; i < N; ++i) {
                // V log-uniform over [4^(j0+1), 2^14], k = number of thresholds 4^(j0+i) that V passes
                const double lg = 2.0 * (c.rsqrt_j0 + 1) + (14.0 - 2.0 * (c.rsqrt_j0 + 1)) * (static_cast<double>(vr.next_u64() >> 11) / 9007199254740992.0);
                const uint64_t V = static_cast<uint64_t>(std::llround(std::pow(2.0, lg) * std::ldexp(1.0, f)));
                int k = 0;
                for (int t = 1; t <= c.rsqrt_m; ++t)
                    if (static_cast<double>(V) >= std::pow(4.0, c.rsqrt_j0 + t) * std::ldexp(1.0, f)) k = t;
                const uint64_t s2 = static_cast<uint64_t>(std::llround(std::pow(4.0, -(c.rsqrt_j0 + 1) - k) * std::ldexp(1.0, f + e2)));
                const uint64_t prod = V * s2;
                max_prod = std::max(max_prod, prod);
                exact[i] = static_cast<int64_t>(prod >> (f + e2));
                const uint64_t mx = vr.next_u64(), ms = vr.next_u64();
                sx.a[i] = p.party() == 0 ? mx : V - mx;
                ss.a[i] = p.party() == 0 ? ms : s2 - ms;
            }
            OC2_TRY(z, p.mul_fxp(sx, ss, f + e2));
            OC2_TRY(Z, p.reveal(z));
            int64_t gross = 0;
            for (size_t i = 0; i < N; ++i) {
                const int64_t d = static_cast<int64_t>(Z.a[i]) - exact[i];
                if (d > 2 || d < -2) ++gross;
            }
            res.set("rsqrt_norm_gross", gross);
            res.set("rsqrt_norm_max_product_bits", static_cast<int64_t>(64 - __builtin_clzll(max_prod | 1)));
            res.set("rsqrt_scale_bits", static_cast<int64_t>(e2));
            res.set("rsqrt_product_bits", static_cast<int64_t>(rsqrt_product_bits(c)));
        }
    }
    // --- softmax (with a causal mask) and layernorm
    {
        size_t R = 4, n = 32;
        auto x = rand_vec(rng, R * n, -6, 6);
        std::vector<uint8_t> mask(R * n, 1);
        for (size_t j = 20; j < n; ++j) mask[1 * n + j] = 0;  // row 1 attends 20 keys
        mask[2 * n + 0] = 0;
        OC2_TRY(sx, share_plain(p, x, R, n));
        OC2_TRY(sm, p.softmax_rows(sx, &mask, 4));
        OC2_TRY(SM, p.reveal(sm));
        auto smd = p.decode(SM);
        double err = 0.0, masked_leak = 0.0, sum_err = 0.0;
        for (size_t r = 0; r < R; ++r) {
            std::vector<double> row(x.begin() + static_cast<std::ptrdiff_t>(r * n), x.begin() + static_cast<std::ptrdiff_t>((r + 1) * n));
            std::vector<uint8_t> mrow(mask.begin() + static_cast<std::ptrdiff_t>(r * n), mask.begin() + static_cast<std::ptrdiff_t>((r + 1) * n));
            auto ref = softmax_ref(row, &mrow);
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                err = std::max(err, std::fabs(smd[r * n + j] - ref[j]));
                if (!mrow[j]) masked_leak = std::max(masked_leak, std::fabs(smd[r * n + j]));
                sum += smd[r * n + j];
            }
            sum_err = std::max(sum_err, std::fabs(sum - 1.0));
        }
        res.set("softmax_err", err);
        res.set("softmax_masked_leak", masked_leak);
        res.set("softmax_sum_err", sum_err);
        size_t d = 48;
        auto y = rand_vec(rng, 3 * d, -3, 3);
        for (size_t j = 0; j < d; ++j) y[j] = y[j] * 4.0 + 2.0;  // row 0: large variance and offset
        auto gamma = rand_vec(rng, d, 0.5, 1.5), beta = rand_vec(rng, d, -0.5, 0.5);
        Mat genc, benc;
        if (p.party() == 0) {
            OC2_TRY(ge, p.encode(gamma, 1, d));
            OC2_TRY(be, p.encode(beta, 1, d));
            genc = ge;
            benc = be;
        }
        OC2_TRY(gs, p.share_input(p.party() == 0 ? &genc : nullptr, 0, 1, d));
        OC2_TRY(bs, p.share_input(p.party() == 0 ? &benc : nullptr, 0, 1, d));
        OC2_TRY(sy, share_plain(p, y, 3, d));
        OC2_TRY(ln, p.layernorm_rows(sy, &gs, &bs, 1e-5, false));
        OC2_TRY(LN, p.reveal(ln));
        auto lnd = p.decode(LN);
        double lerr = 0.0;
        for (size_t r = 0; r < 3; ++r) {
            std::vector<double> row(y.begin() + static_cast<std::ptrdiff_t>(r * d), y.begin() + static_cast<std::ptrdiff_t>((r + 1) * d));
            auto ref = layernorm_ref(row, &gamma, &beta, 1e-5, false);
            for (size_t j = 0; j < d; ++j) lerr = std::max(lerr, std::fabs(lnd[r * d + j] - ref[j]));
        }
        res.set("layernorm_err", lerr);
        res.set("layernorm_rounds", rounds_of(p, "layernorm_rows"));
    }
    // --- ring_to_field
    {
        uint64_t t = 68720050177ULL;
        std::vector<uint64_t> u = {0, 1, static_cast<uint64_t>(-1LL), static_cast<uint64_t>(-12345LL), 1ULL << 63,
                                   (1ULL << 63) - 1, rng.next_u64(), rng.next_u64(), 255, static_cast<uint64_t>(-2LL)};
        Mat plain = Mat::from_vector(u, 1, u.size());
        OC2_TRY(su, p.share_input(p.party() == 1 ? &plain : nullptr, 1, 1, u.size()));
        OC2_TRY(z, p.ring_to_field(su, t));
        // reconstruct over Z_t: exchange shares (test-only reveal)
        OC2_TRY(other, p.channel().exchange(u64_array(z.a, {1, u.size()})));
        auto oz = other.as_array().value.as_u64();
        int mism = 0;
        for (size_t i = 0; i < u.size(); ++i) {
            uint64_t rec = (z.a[i] + oz[i]) % t;
            if (rec != lowlevel::r2f::signed_mod(u[i], t)) ++mism;
            if (z.a[i] >= t) ++mism;
        }
        res.set("r2f_mismatches", static_cast<int64_t>(mism));
    }
    // --- reveal_final through the ReleasePolicy (recipient P1, kinds token/class, max 4)
    {
        Mat sx = p.const_ring(17, 1, 1);
        OC2_TRY(r1, p.reveal_final(sx, "token"));
        res.set("release_p1_value", p.party() == 1 ? static_cast<int64_t>(r1.a[0]) : static_cast<int64_t>(-1));
        res.set("release_p0_empty", p.party() == 0 ? r1.empty() : true);
        auto bad = p.reveal_final(sx, "hidden");
        res.set("release_bad_kind_refused", !bad.ok && bad.code.rfind("UNAUTHORIZED_RELEASE", 0) == 0);
        OC2_TRY(r2, p.reveal_final(sx, "class"));
        OC2_TRY(r3, p.reveal_final(sx, "token"));
        OC2_TRY(r4, p.reveal_final(sx, "token"));
        (void)r2; (void)r3; (void)r4;
        auto exhausted = p.reveal_final(sx, "token");
        res.set("release_count_enforced", !exhausted.ok);
    }
    res.set("rounds_total", static_cast<int64_t>(p.channel().stats.rounds));
    res.set("bytes_sent", static_cast<int64_t>(p.channel().stats.bytes_sent));
    res.set("bytes_recv", static_cast<int64_t>(p.channel().stats.bytes_recv));
    return ok(std::move(res));
}

static int test_two_party_gadgets() {
    auto tpr = make_two_party("gadgets");
    CHECK_OK(tpr, "two party setup");
    TwoParty& tp = *tpr.value;
    auto out = run_both(tp, gadget_body);
    for (int p = 0; p < 2; ++p)
        if (!out[p].ok) std::fprintf(stderr, "party %d body failed: %s\n", p, out[p].code.c_str());
    CHECK_OK(out[0], "party 0 body");
    CHECK_OK(out[1], "party 1 body");
    const Value& r = out[1].value;
    const Value& r0 = out[0].value;
    std::printf("gadget metrics: %s\n", to_json(r).value.c_str());
    auto num = [&](const char* k) { return r.get_number(k).value; };
    auto inum = [&](const char* k) { return r.get_int(k).value; };
    CHECK(num("mul_fxp_err_ulp") <= 2.0, "mul_fxp within 2 ulp");
    CHECK(inum("trunc_faithful_mismatches") == 0, "faithful truncation exact across magnitudes 2^4..2^59");
    CHECK(inum("trunc_faithful_rounds") == 1, "faithful truncation is one round");
    CHECK(num("matmul_err") <= 2e-4, "matmul_shared accurate");
    CHECK(inum("matmul_rounds") == 1, "matmul_shared is one round");
    CHECK(num("linear_generic_err") <= 2e-4, "linear_generic accurate");
    CHECK(num("linear_wmask_err") <= 2e-4, "linear_wmask accurate");
    CHECK(r.get_string("wmask_before_install_code").value == "OBJECT_NOT_INSTALLED", "wmask before install refused");
    CHECK(inum("install_rounds") == 1, "install_wmask is one round");
    {
        // measured versus formula for install_wmask (the mask bundle used to be consumed before
        // the Track existed, so the gadget reported material 0)
        auto est = material_estimate("install_wmask", {6, 5}, 32, 16);
        CHECK(est.ok && inum("install_material") == est.value.get_int("bytes").value && inum("install_material") == 8 * 6 * 5,
              "install_wmask material equals the formula 8 d m");
        CHECK(r.get_string("install_stat_kind").value == "install_wmask" && inum("install_stat_rounds") == 1, "install stat: one round");
        CHECK(inum("install_stat_bytes") == 2 * (8 + 1 + 1 + 1 + 2 * 8 + 6 * 5 * 8), "install opens B1 one way and E the other (d m words each)");
    }
    CHECK(inum("rsqrt_scale_bits") == 2 && inum("rsqrt_product_bits") == 34, "rsqrt selector scale: e2 = 2, product below 2^34");
    CHECK(inum("rsqrt_norm_max_product_bits") <= 35, "the normalisation product stays inside the declared bound");
    CHECK(inum("rsqrt_norm_gross") == 0, "no gross truncation error in the rsqrt normalisation product (2^18 elements)");
    CHECK(r.get("install_skipped_second")->as_bool().value, "second install skipped");
    CHECK(inum("linear_wmask_bytes") == 2 * (8 + 1 + 1 + 1 + 2 * 8 + 3 * 6 * 8), "linear_wmask opens only X - A (3 x 6 words each way)");
    CHECK(inum("drelu_relu_mismatches") == 0, "drelu / relu / compare_public exact");
    CHECK(inum("drelu_rounds") == 1, "drelu is one round");
    CHECK(inum("argmax_k_disagreements") == 0, "argmax identical for k = 2, 4, 8");
    CHECK(inum("argmax_ref_mismatches") == 0, "argmax equals the lowest-index reference");
    CHECK(inum("max_mismatches") == 0, "max_rows exact");
    CHECK(inum("tournament_accounting_calls") == 24 && inum("tournament_accounting_mismatches") == 0,
          "fused tournament: rounds = 2 levels, opened bytes and material equal the Python realization for every (n, k)");
    CHECK(r0.get_int("tournament_accounting_mismatches").value == 0, "fused tournament accounting agrees on P0");
    CHECK(inum("tournament_python_pin_hits") == 18 && inum("tournament_python_pin_mismatches") == 0,
          "rounds, wire bytes and material equal the values measured on the Python realization (18 pinned shapes)");
    CHECK(num("masked_max_err") <= 1e-5, "masked max excludes masked positions (quantised inputs)");
    CHECK(inum("masked_max_rounds") == 4, "masked 4-ary max over 6: two levels, two rounds each");
    CHECK(inum("masked_max_bytes") == 2 * (4 * 19 + 8 * (10 + 10 + 1 + 4)), "masked max: publicly decided pairs open nothing");
    CHECK(inum("masked_max_material") == 11 * 1672 + 6 * 2200 + 4 * 24, "masked max material: 11 comparisons, 6 payload DReLUs, 4 select triples");
    CHECK(inum("kary_bound_edge_mismatches") == 0, "k-ary max / argmax exact just inside the magnitude bound of every k");
    CHECK(inum("kary_bound_g8") == (1LL << 27) && inum("kary_bound_g4") == (1LL << 28) && inum("kary_bound_g2") == (1LL << 31),
          "declared bounds: 2^(n-2-ceil(log2 g)) ring units for fused nodes, 2^(n-1) for binary nodes");
    CHECK(inum("trunc_local_mismatches") == 0 && r0.get_int("trunc_local_mismatches").value == 0,
          "trunc_local equals the lowlevel gate (explicit arithmetic shift) on both parties");
    CHECK(r0.get_int("trunc_local_m13").value == -2 && inum("trunc_local_m13") == -1, "trunc_local floors on P0 and negates the floor on P1");
    CHECK(r.get("trunc_local_f0_identity")->as_bool().value, "trunc_local by 0 bits is the identity");
    CHECK(num("gelu_err") <= 4.84e-3 + 3e-3, "gelu within contract error + fixed-point noise");
    CHECK(num("silu_err") <= 1.28e-2 + 3e-3, "silu within contract error + fixed-point noise");
    CHECK(inum("gelu_rounds") == 14, "gelu rounds 1 + 12 + 1");
    CHECK(num("exp_neg_err_ratio") <= 1.0, "exp_neg within 1% relative + 3e-4");
    CHECK(inum("exp_neg_rounds") == 10, "exp_neg rounds 3 + r");
    CHECK(num("recip_rel_err") <= 2e-3, "recip relative error");
    CHECK(inum("recip_rounds") == 12, "recip rounds 12");
    CHECK(num("rsqrt_rel_err") <= 3e-3, "rsqrt relative error");
    CHECK(inum("rsqrt_rounds") == 14, "rsqrt rounds 14");
    CHECK(num("softmax_err") <= 5e-3, "softmax absolute error");
    CHECK(num("softmax_masked_leak") == 0.0, "masked softmax entries are exactly 0");
    CHECK(num("softmax_sum_err") <= 1e-2, "softmax rows sum to 1");
    CHECK(num("layernorm_err") <= 2e-2, "layernorm absolute error");
    CHECK(inum("layernorm_rounds") == 18, "layernorm rounds 18");
    CHECK(inum("r2f_mismatches") == 0, "ring_to_field exact and reduced");
    CHECK(inum("release_p1_value") == 17 && r0.get("release_p0_empty")->as_bool().value, "reveal_final to P1 only");
    CHECK(r.get("release_bad_kind_refused")->as_bool().value, "undeclared release kind refused");
    CHECK(r.get("release_count_enforced")->as_bool().value, "release count enforced");
    CHECK(r.get_int("rounds_total").value == r0.get_int("rounds_total").value, "round counters agree on both parties");
    CHECK(r.get_int("bytes_sent").value == r0.get_int("bytes_recv").value, "byte counters agree on both parties");
    // ledger: every touched bundle is SPENT, nothing RESERVED
    bool spent_only = true;
    for (int p = 0; p < 2; ++p) {
        auto rec = tp.ledger[p]->recover();
        if (!rec.ok || rec.value.get_int("reserved_burned").value != 0) spent_only = false;
    }
    CHECK(spent_only, "no RESERVED material left after a clean run");
    // telemetry probe + verifier over the same channels
    {
        Bytes key(32, 0x5a);
        std::array<Expected<TelemetryFrame>, 2> fr;
        std::array<TelemetryVerifier, 2> ver = {TelemetryVerifier(key), TelemetryVerifier(key)};
        auto run = [&](int p) { fr[p] = sense_epoch(*tp.ch[p], p, 0, 0, 1.0, ver[p], key, 1e12, 2, 4096); };
        std::thread a(run, 0), b(run, 1);
        a.join();
        b.join();
        CHECK_OK(fr[0], "sense P0");
        CHECK_OK(fr[1], "sense P1");
        CHECK(fr[0].value.digest().value == fr[1].value.digest().value, "both parties hold the identical frame");
        CHECK(fr[1].value.bandwidth_bps > 0 && fr[1].value.rtt_s > 0, "probe measured positive values");
        CHECK(tp.ch[0]->stats.rounds == tp.ch[1]->stats.rounds, "probe rounds symmetric");
        TelemetryFrame f2 = fr[1].value;
        f2.seq = 1;
        f2.t = 2.0;
        auto tag2 = sign_frame(f2, key).value;
        CHECK_OK(ver[0].accept(f2, tag2), "next frame accepted");
        CHECK(!ver[0].accept(f2, tag2).ok, "replay rejected");
        TelemetryFrame f3 = f2;
        f3.seq = 2;
        f3.bandwidth_bps += 1.0;
        CHECK(ver[0].accept(f3, tag2).code.rfind("TELEMETRY_BAD_TAG", 0) == 0, "tampered frame rejected");
        TelemetryFrame f4 = f2;
        f4.seq = 5;
        CHECK(ver[0].accept(f4, sign_frame(f4, key).value).code.rfind("TELEMETRY_OUT_OF_ORDER", 0) == 0, "gap rejected");
        auto rt = TelemetryFrame::from_value(f2.to_value());
        CHECK(rt.ok && rt.value.digest().value == f2.digest().value, "frame dict round trip");
        // probe parameters are validated before any message (the unchecked code crashed on samples = 0)
        uint64_t sent_before = tp.ch[0]->stats.bytes_sent + tp.ch[1]->stats.bytes_sent;
        for (int p = 0; p < 2; ++p) {
            CHECK(probe_channel(*tp.ch[p], p, 0, 4096).code.rfind("TELEMETRY_PARAMS", 0) == 0, "probe_samples = 0 refused");
            CHECK(probe_channel(*tp.ch[p], p, -3, 4096).code.rfind("TELEMETRY_PARAMS", 0) == 0, "negative probe_samples refused");
            CHECK(probe_channel(*tp.ch[p], p, 3, 0).code.rfind("TELEMETRY_PARAMS", 0) == 0, "empty probe burst refused");
            CHECK(probe_channel(*tp.ch[p], p, PROBE_MAX_SAMPLES + 1, 4096).code.rfind("TELEMETRY_PARAMS", 0) == 0, "oversized sample count refused");
            TelemetryVerifier v2(key);
            CHECK(sense_epoch(*tp.ch[p], p, 0, 0, 1.0, v2, key, 1e12, 0, 4096).code.rfind("TELEMETRY_PARAMS", 0) == 0,
                  "sense_epoch propagates the parameter check");
        }
        CHECK(tp.ch[0]->stats.bytes_sent + tp.ch[1]->stats.bytes_sent == sent_before, "a refused probe sends nothing");
    }
    for (int p = 0; p < 2; ++p) std::remove(tp.paths[p].c_str());
    return 0;
}

int main() {
    if (test_codec()) return 1;
    if (test_shaper_and_channel()) return 1;
    if (test_ledger()) return 1;
    if (test_dealer()) return 1;
    if (test_dealer_spellings_and_burn()) return 1;
    if (test_poly()) return 1;
    if (test_two_party_gadgets()) return 1;
    std::printf("PASS %d checks\n", g_checks);
    return 0;
}
