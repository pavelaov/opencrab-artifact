// OpenCrab v2 C++17 mirror: MaterialLedger crash-boundary tests and decoder hardening
// (paper, Section 4.3, ledger).
//
// What this file checks.
//   1. Torn trailing journal line. With one write + fsync per record, a crash can leave exactly
//      one incomplete record at the end of the journal. recover() must drop it (durable
//      truncation), record a LEDGER_TORN_TAIL note and continue; any malformed line that is NOT
//      the unterminated tail must still fail closed (LEDGER_CORRUPT / LEDGER_BAD_STATE) and leave
//      the file untouched.
//   2. Crash-boundary injection. The party process is "killed" (the ledger object is dropped
//      without any cleanup call and a new one replays the journal) after each of: reserve, spend
//      before send, send before ack, commit. A second pass cuts the journal at every line
//      boundary and inside every line (kill in the middle of a write). After every restart the
//      four ledger invariants are asserted:
//        I1  a material_id enters SPENT at most once and a SPENT id is never returned;
//        I2  reserve is atomic, spend is journaled BEFORE the first message that uses the
//            material, RESERVED entries are burned at restart;
//        I3  a message for (gate_id, seq) may be retransmitted only byte-identical, and a new
//            input cannot reuse the mask of a prior seq;
//        I4  the binding tuple and the installed-object versions survive the restart and reject
//            every other tuple or version.
//      The gadget-level form of I2 (a real Party::beaver_mul over a loopback channel) reads the
//      journal from disk at the moment the first frame leaves and requires the SPENT line there.
//   3. wmask version binding. An installed object recorded for model version v0 is never
//      reported installed for v1, the v0 mask bundle cannot enter a v1 session, and a real
//      re-install under v1 draws a fresh mask, so E1 - E0 = W1 - W0 cannot be formed by P1.
//   4. Decoder limits: CODEC_DEPTH (wire codec, limit 64), JSON_DEPTH (parse_json, limit 64),
//      CHANNEL_FRAME_TOO_LARGE (default 2 GiB, configurable) and no up-front allocation of an
//      announced frame.
//
// What is NOT covered: a real process kill (the restart is simulated in one process; the journal
// bytes on disk are what a killed process leaves because every record is fsynced before the
// call returns), power loss that reorders or drops fsynced blocks, and a host that rolls the
// journal back (design limit stated in ledger.hpp).
//
// Prints "PASS <n> checks" and exits 0, or prints the failing check and exits 1.
#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/contract.hpp>
#include <opencrab2/highlevel/dealer.hpp>
#include <opencrab2/highlevel/json.hpp>
#include <opencrab2/highlevel/ledger.hpp>
#include <opencrab2/highlevel/party.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace opencrab2;
using namespace opencrab2::highlevel;

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
    return std::string(base) + "/oc2_ledger_" + name + "_" + std::to_string(static_cast<long long>(stamp)) + "_" +
           std::to_string(counter++) + ".jsonl";
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void append_file(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::vector<std::string> complete_lines(const std::string& data) {
    std::vector<std::string> lines;
    size_t pos = 0;
    for (;;) {
        const size_t nl = data.find('\n', pos);
        if (nl == std::string::npos) break;
        lines.push_back(data.substr(pos, nl - pos));
        pos = nl + 1;
    }
    return lines;
}

Value binding_for(const std::string& model_version) {
    return dict({{"session_id", "sess-1"}, {"setup_id", "setup-A"}, {"model_version", model_version},
                 {"numeric_contract_hash", "hash-1"}, {"profile_id", "SH2P-TrustedOffline-v1"}});
}

Value material_version(const std::string& model_version) {
    return dict({{"setup_id", "setup-A"}, {"model_version", model_version}, {"numeric_contract_hash", "hash-1"},
                 {"profile_id", "SH2P-TrustedOffline-v1"}});
}

Value object_version(const std::string& model_version) {
    return dict({{"model_version", model_version}, {"client_key_domain", "client-0"}});
}

// ---------------------------------------------------------------------------------------------
// Independent journal model (test code only): replays complete lines and audits the file.
// ---------------------------------------------------------------------------------------------
struct Model {
    std::map<std::string, std::string> state;      // material_id -> AVAILABLE | RESERVED | SPENT | BURNED
    std::map<std::string, std::string> send_log;   // "gate/seq" -> digest
    std::map<std::string, std::string> installed;  // object id -> canonical version
    std::map<std::string, int> spent_lines;        // material_id -> number of SPENT lines
    std::map<std::string, int> spent_index;        // material_id -> line index of its SPENT line
    std::map<std::string, int> sent_index;         // "gate/seq" -> line index of its SENT line
    int notes = 0;
    std::string violation;                         // first audit violation (empty when clean)
};

Model replay_model(const std::vector<std::string>& lines) {
    Model m;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].empty()) continue;
        const auto v = parse_json(lines[i]);
        if (!v.ok) {
            if (m.violation.empty()) m.violation = "unparsable line " + std::to_string(i + 1);
            continue;
        }
        const std::string id = v.value.get_string("material_id").value;
        const std::string st = v.value.get_string("state").value;
        if (st == "BINDING") continue;   // first line of every journal: the session binding
        if (st == "NOTE") {
            ++m.notes;
            continue;
        }
        if (st == "SENT") {
            m.send_log[id] = v.value.get_string("digest").value;
            if (!m.sent_index.count(id)) m.sent_index[id] = static_cast<int>(i);
            continue;
        }
        if (st == "INSTALLED") {
            m.installed[id] = canonical_json(*v.value.get("version")).value;
            continue;
        }
        const std::string prev = m.state.count(id) ? m.state[id] : std::string();
        // audit of invariant 1 on the raw file: terminal states are never left, SPENT at most once
        if ((prev == "SPENT" || prev == "BURNED") && st != prev && m.violation.empty())
            m.violation = id + " leaves " + prev + " at line " + std::to_string(i + 1);
        if (prev.empty() && st != "AVAILABLE" && m.violation.empty()) m.violation = id + " starts with " + st;
        if (st == "SPENT") {
            if (prev != "RESERVED" && m.violation.empty()) m.violation = id + " SPENT from " + prev;
            m.spent_lines[id] += 1;
            if (m.spent_lines[id] > 1 && m.violation.empty()) m.violation = id + " SPENT twice";
            m.spent_index[id] = static_cast<int>(i);
        }
        m.state[id] = st;
    }
    return m;
}

// What a restart must produce from a journal prefix: RESERVED becomes BURNED.
std::map<std::string, std::string> expected_after_restart(const Model& m) {
    std::map<std::string, std::string> out = m.state;
    for (auto& kv : out)
        if (kv.second == "RESERVED") kv.second = "BURNED";
    return out;
}

// ---------------------------------------------------------------------------------------------
// 1. Torn trailing line
// ---------------------------------------------------------------------------------------------
void test_torn_tail() {
    const std::string path = temp_path("torn");
    const Value ver = material_version("v0");
    {
        MaterialLedger L(path, binding_for("v0"));
        CHECK_OK(L.open(), "open");
        CHECK_OK(L.add_available("a", "beaver", 240, ver), "add a");
        CHECK_OK(L.add_available("b", "beaver", 240, ver), "add b");
        CHECK_OK(L.add_available("c", "drelu", 1000, ver), "add c");
        CHECK_OK(L.reserve({"a"}, "g", 0), "reserve a");
        CHECK_OK(L.spend({"a"}), "spend a");
        CHECK_OK(L.record_send("g", 0, "d0"), "record send");
    }
    const std::string clean = read_file(path);
    CHECK(!clean.empty() && clean.back() == '\n', "journal ends with a newline");

    // (a) a record cut in the middle of a string
    const std::string fragment = "{\"material_id\": \"c\", \"state\": \"RESER";
    append_file(path, fragment);
    {
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, "recover tolerates a torn trailing line");
        CHECK(rec.value.get_int("torn_tail").value == 1, "torn tail reported");
        CHECK(rec.value.get_int("torn_tail_bytes").value == static_cast<int64_t>(fragment.size()), "torn byte count");
        CHECK(L.notes().size() == 1 && L.notes()[0].rfind("LEDGER_TORN_TAIL", 0) == 0, "LEDGER_TORN_TAIL note recorded");
        CHECK(rec.value.get("notes")->as_list().value->size() == 1, "note returned by recover");
        CHECK(L.state("c").value == "AVAILABLE", "the id of the torn record keeps its earlier state");
        CHECK(L.state("a").value == "SPENT" && L.state("b").value == "AVAILABLE", "complete records replayed");
        CHECK(L.retransmit_allowed("g", 0, "d0"), "send log replayed");
        const std::string after = read_file(path);
        CHECK(after.compare(0, clean.size(), clean) == 0, "history before the tail is byte-identical");
        CHECK(after.find(fragment) == std::string::npos, "the torn fragment is gone from the file");
        CHECK(after.back() == '\n', "the journal ends at a line boundary again");
        const std::vector<std::string> lines = complete_lines(after);
        const auto note = parse_json(lines.back());
        CHECK_OK(note, "note line parses");
        CHECK(note.value.get_string("state").value == "NOTE" && note.value.get_string("material_id").value == "LEDGER_TORN_TAIL",
              "journaled NOTE line");
        CHECK(note.value.get_int("dropped_bytes").value == static_cast<int64_t>(fragment.size()), "note carries the byte count");
        CHECK(note.value.get_int("offset").value == static_cast<int64_t>(clean.size()), "note carries the offset");
        // the ledger is usable: the journal keeps growing in whole lines
        CHECK_OK(L.reserve({"c"}, "g2", 1), "reserve after a torn-tail recovery");
        CHECK_OK(L.spend({"c"}), "spend after a torn-tail recovery");
    }
    {
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, "second recover");
        CHECK(rec.value.get_int("torn_tail").value == 0, "nothing torn the second time");
        CHECK(L.notes().size() == 1, "the audit note survives restarts");
        CHECK(L.state("c").value == "SPENT", "post-recovery records are durable");
        CHECK(replay_model(complete_lines(read_file(path))).violation.empty(), "file audit clean");
    }

    // (b) a tail that is complete JSON but lacks its newline: the write was never acknowledged,
    //     so it is dropped as well; the id was RESERVED and is burned, never SPENT, never returned
    {
        write_file(path, clean);
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.recover(), "recover clean");
            CHECK_OK(L.reserve({"b"}, "g3", 2), "reserve b");
            CHECK_OK(L.spend({"b"}), "spend b");
        }
        std::string data = read_file(path);
        CHECK(data.back() == '\n', "ends with newline");
        data.pop_back();   // the SPENT record of b without its newline
        write_file(path, data);
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, "recover with an unterminated complete record");
        CHECK(rec.value.get_int("torn_tail").value == 1, "unterminated record counted as torn");
        CHECK(L.state("b").value == "BURNED", "RESERVED at restart is burned (the SPENT record never completed)");
        CHECK_CODE(L.reserve({"b"}, "g4", 3), "MATERIAL_REUSE", "burned id is never returned");
        CHECK(rec.value.get_int("reserved_burned").value == 1, "one entry burned by this recovery");
    }

    // (c) corruption in the middle still fails closed, the file is untouched, the ledger is dead
    {
        std::vector<std::string> lines = complete_lines(clean);
        std::string bad;
        for (size_t i = 0; i < lines.size(); ++i) bad += (i == 2 ? lines[i].substr(0, lines[i].size() / 2) : lines[i]) + "\n";
        write_file(path, bad);
        MaterialLedger L(path, binding_for("v0"));
        CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "corrupt middle line refused");
        CHECK(read_file(path) == bad, "a refused journal is left byte-identical");
        CHECK(L.corrupt(), "ledger marked corrupt");
        CHECK_CODE(L.open(), "LEDGER_CORRUPT", "open refused after a failed recovery");
        CHECK_CODE(L.add_available("z", "beaver", 8, ver), "LEDGER_NOT_OPEN", "no state change after a failed recovery");
        CHECK_CODE(L.reserve({"a"}, "g", 0), "LEDGER_NOT_OPEN", "no reserve after a failed recovery");

        // middle corruption AND a torn tail: still refused, and nothing is truncated
        const std::string both = bad + fragment;
        write_file(path, both);
        MaterialLedger L2(path, binding_for("v0"));
        CHECK_CODE(L2.recover(), "LEDGER_CORRUPT", "corrupt middle line refused even with a torn tail");
        CHECK(read_file(path) == both, "no truncation when recovery is refused");
    }

    // (d) an unparsable FINAL line that has its newline is corruption, not a torn tail
    {
        const std::string bad = clean + "{\"material_id\": \"c\", \"state\": \"RESER\n";
        write_file(path, bad);
        MaterialLedger L(path, binding_for("v0"));
        CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "terminated but unparsable final line refused");
        CHECK(read_file(path) == bad, "file untouched");
    }

    // (e) structurally impossible histories are refused (they would return a SPENT id)
    {
        const std::vector<std::string> lines = complete_lines(clean);
        std::string avail_a, spent_a;
        for (const auto& l : lines) {
            const auto v = parse_json(l);
            if (v.value.get_string("material_id").value == "a" && v.value.get_string("state").value == "AVAILABLE") avail_a = l;
            if (v.value.get_string("material_id").value == "a" && v.value.get_string("state").value == "SPENT") spent_a = l;
        }
        CHECK(!avail_a.empty() && !spent_a.empty(), "fixture lines found");
        write_file(path, clean + avail_a + "\n");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "AVAILABLE after SPENT refused");
        }
        write_file(path, clean + spent_a + "\n");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "second SPENT refused");
        }
        write_file(path, spent_a + "\n");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "state line without an AVAILABLE line refused");
        }
        std::string weird = avail_a;
        const size_t at = weird.find("AVAILABLE");
        weird.replace(at, 9, "HALFSPENT");
        write_file(path, weird + "\n");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_CODE(L.recover(), "LEDGER_BAD_STATE", "unknown state refused");
        }
    }

    // (f) a tampered line with 100000 nested arrays must not overflow the stack
    {
        const std::string deep = std::string(100000, '[') + std::string(100000, ']');
        write_file(path, clean + deep + "\n" + clean);
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_CODE(rec, "LEDGER_CORRUPT", "deeply nested line refused");
        CHECK(rec.code.find("JSON_DEPTH") != std::string::npos, "the refusal names JSON_DEPTH");
    }

    // (g) a journal that holds nothing but a fragment
    {
        write_file(path, fragment);
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, "recover from a lone fragment");
        CHECK(rec.value.get_int("torn_tail").value == 1 && rec.value.get_int("available").value == 0, "empty state");
        CHECK_OK(L.add_available("fresh", "beaver", 8, ver), "usable afterwards");
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------------------------
// 2. Crash boundaries
// ---------------------------------------------------------------------------------------------
enum class Boundary { RESERVE = 1, SPEND_BEFORE_SEND = 2, SEND_BEFORE_ACK = 3, COMMIT = 4 };

const char* boundary_name(Boundary b) {
    switch (b) {
        case Boundary::RESERVE: return "after reserve";
        case Boundary::SPEND_BEFORE_SEND: return "after spend, before send";
        case Boundary::SEND_BEFORE_ACK: return "after send, before ack";
        case Boundary::COMMIT: return "after commit";
    }
    return "?";
}

const char* kDigest = "digest-of-the-g1-message";
const char* kObject = "wmask_install(W)";

// One gate of a session. An earlier gate g0 is complete (m-old SPENT, its message logged). Gate g1
// uses two bundles (m-a, m-b); m-c is spare stock. The steps stop at `until` and the ledger object
// is then dropped by the caller without any cleanup call: that is the simulated kill.
void run_gate(MaterialLedger& L, Boundary until) {
    const Value ver = material_version("v0");
    CHECK_OK(L.open(), "open");
    CHECK_OK(L.add_available("m-old", "beaver", 100, ver), "add m-old");
    CHECK_OK(L.reserve({"m-old"}, "g0", 0), "reserve m-old");
    CHECK_OK(L.spend({"m-old"}), "spend m-old");
    CHECK_OK(L.record_send("g0", 0, "digest-old"), "send g0");
    CHECK_OK(L.add_available("m-a", "beaver", 240, ver), "add m-a");
    CHECK_OK(L.add_available("m-b", "trunc_faithful", 2064, ver), "add m-b");
    CHECK_OK(L.add_available("m-c", "beaver", 240, ver), "add m-c");
    CHECK_OK(L.reserve({"m-a", "m-b"}, "g1", 3), "reserve g1");                 // boundary RESERVE
    if (until == Boundary::RESERVE) return;
    CHECK_OK(L.spend({"m-a", "m-b"}), "spend g1");                              // boundary SPEND_BEFORE_SEND
    if (until == Boundary::SPEND_BEFORE_SEND) return;
    CHECK_OK(L.record_send("g1", 7, kDigest), "record send g1");                // the message leaves here
    if (until == Boundary::SEND_BEFORE_ACK) return;
    CHECK_OK(L.register_installed(kObject, object_version("v0")), "commit");    // boundary COMMIT
}

// Invariants that hold after EVERY restart, whatever the crash point.
void check_common_invariants(MaterialLedger& L, const std::string& path, const std::string& ctx) {
    // I1: the earlier gate's material is SPENT and is never returned
    CHECK(L.state("m-old").value == "SPENT", ctx + ": SPENT survives the restart");
    CHECK_CODE(L.reserve({"m-old"}, "g9", 9), "MATERIAL_REUSE", ctx + ": a SPENT id is never returned");
    CHECK_CODE(L.spend({"m-old"}), "LEDGER_NOT_RESERVED", ctx + ": a SPENT id cannot be spent again");
    CHECK_CODE(L.burn({"m-old"}, "x"), "LEDGER_ALREADY_SPENT", ctx + ": a SPENT id cannot be relabelled");
    // I2: nothing is left RESERVED by a restart
    for (const char* id : {"m-old", "m-a", "m-b", "m-c"}) {
        const auto st = L.state(id);
        if (st.ok) CHECK(st.value != "RESERVED", ctx + ": no RESERVED entry survives a restart");
    }
    // I2: reserve is atomic (one unknown id: nothing is reserved and nothing is journaled)
    const std::string before = read_file(path);
    if (L.state("m-c").ok && L.state("m-c").value == "AVAILABLE") {
        CHECK_CODE(L.reserve({"m-c", "no-such-id"}, "g9", 9), "LEDGER_UNKNOWN_ID", ctx + ": reserve with an unknown id fails");
        CHECK(L.state("m-c").value == "AVAILABLE", ctx + ": atomic reserve left m-c AVAILABLE");
        CHECK_CODE(L.reserve({"m-c", "m-old"}, "g9", 9), "MATERIAL_REUSE", ctx + ": reserve with a SPENT id fails");
        CHECK(L.state("m-c").value == "AVAILABLE", ctx + ": atomic reserve left m-c AVAILABLE (2)");
        CHECK(read_file(path) == before, ctx + ": a refused reserve writes nothing");
    }
    // I3: the earlier gate's message may be retransmitted only byte-identical
    CHECK(L.retransmit_allowed("g0", 0, "digest-old"), ctx + ": identical retransmission allowed");
    CHECK(!L.retransmit_allowed("g0", 0, "another-digest"), ctx + ": different payload under the same seq refused");
    CHECK_CODE(L.record_send("g0", 0, "another-digest"), "LEDGER_SEQ_REUSE", ctx + ": seq reuse with a new input refused");
    CHECK_OK(L.record_send("g0", 0, "digest-old"), ctx + ": identical record_send is idempotent");
    CHECK(read_file(path) == before, ctx + ": an identical retransmission writes nothing");
    // I4: the binding tuple rejects every other tuple
    CHECK_OK(L.check_binding(binding_for("v0")), ctx + ": own binding accepted");
    for (const char* key : {"setup_id", "model_version", "numeric_contract_hash", "profile_id"}) {
        Value foreign = binding_for("v0");
        foreign.set(key, Value("something-else"));
        CHECK_CODE(L.check_binding(foreign), "MATERIAL_VERSION_MISMATCH", ctx + ": foreign binding rejected");
    }
    CHECK(!L.installed(kObject, object_version("v1")), ctx + ": never installed for another model version");
    // the file itself: terminal states never left, SPENT at most once, SPENT before SENT
    const Model m = replay_model(complete_lines(read_file(path)));
    CHECK(m.violation.empty(), ctx + ": journal audit: " + m.violation);
    for (const auto& kv : m.spent_lines) CHECK(kv.second <= 1, ctx + ": at most one SPENT record per id");
}

void test_crash_boundaries() {
    for (Boundary b : {Boundary::RESERVE, Boundary::SPEND_BEFORE_SEND, Boundary::SEND_BEFORE_ACK, Boundary::COMMIT}) {
        const std::string path = temp_path("boundary");
        const std::string ctx = std::string("crash ") + boundary_name(b);
        {
            MaterialLedger L(path, binding_for("v0"));
            run_gate(L, b);
        }   // kill: no burn, no close protocol, only what fsync left on disk
        const std::string on_disk = read_file(path);
        const Model before = replay_model(complete_lines(on_disk));
        CHECK(before.violation.empty(), ctx + ": journal written by the gate is well formed");

        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, ctx + ": recover");
        CHECK(rec.value.get_int("torn_tail").value == 0, ctx + ": clean boundary has no torn tail");
        // append-only: the restart never rewrites history
        CHECK(read_file(path).compare(0, on_disk.size(), on_disk) == 0, ctx + ": history is append-only across the restart");

        if (b == Boundary::RESERVE) {
            // I2: RESERVED at restart is burned; it was never used, and it is never returned
            CHECK(rec.value.get_int("reserved_burned").value == 2, ctx + ": both reserved bundles burned");
            CHECK(L.state("m-a").value == "BURNED" && L.state("m-b").value == "BURNED", ctx + ": RESERVED became BURNED");
            CHECK_CODE(L.reserve({"m-a"}, "g1", 3), "MATERIAL_REUSE", ctx + ": burned id never returned");
            CHECK(!L.retransmit_allowed("g1", 7, kDigest), ctx + ": no message of g1 was ever sent");
            CHECK(L.bytes_burned() == 240 + 2064, ctx + ": burned bytes accounted");
        } else {
            // I1 + I2: SPENT was durable before any message could leave
            CHECK(rec.value.get_int("reserved_burned").value == 0, ctx + ": nothing left to burn");
            CHECK(L.state("m-a").value == "SPENT" && L.state("m-b").value == "SPENT", ctx + ": SPENT is durable");
            CHECK_CODE(L.reserve({"m-a", "m-b"}, "g1", 3), "MATERIAL_REUSE", ctx + ": SPENT ids never returned");
            CHECK(before.spent_index.count("m-a") && before.spent_index.count("m-b"), ctx + ": SPENT records on disk");
        }
        if (b == Boundary::SPEND_BEFORE_SEND) {
            // the kill hit between the durable spend and the send: no send record exists, so the
            // spent material never masks anything, and it cannot be used for a later message either
            CHECK(!before.sent_index.count("g1/7"), ctx + ": SPENT is on disk while no send record exists");
            CHECK(!L.retransmit_allowed("g1", 7, kDigest), ctx + ": nothing to retransmit");
        }
        if (b == Boundary::SEND_BEFORE_ACK || b == Boundary::COMMIT) {
            // I2 (order on disk): both SPENT records precede the send record of the gate
            CHECK(before.sent_index.count("g1/7") == 1, ctx + ": send record on disk");
            CHECK(before.spent_index.at("m-a") < before.sent_index.at("g1/7") && before.spent_index.at("m-b") < before.sent_index.at("g1/7"),
                  ctx + ": spend is journaled before the first message");
            // I3: byte-identical retransmission only; a new input needs a new seq AND new material
            CHECK(L.retransmit_allowed("g1", 7, kDigest), ctx + ": the unacknowledged message may be retransmitted identically");
            CHECK(!L.retransmit_allowed("g1", 7, "digest-of-a-new-input"), ctx + ": a new input under seq 7 is refused");
            CHECK_CODE(L.record_send("g1", 7, "digest-of-a-new-input"), "LEDGER_SEQ_REUSE", ctx + ": seq 7 cannot carry a new input");
            CHECK(!L.retransmit_allowed("g1", 8, kDigest), ctx + ": seq 8 has no logged message");
        }
        CHECK(L.installed(kObject, object_version("v0")) == (b == Boundary::COMMIT), ctx + ": installed exactly after the commit record");
        check_common_invariants(L, path, ctx);

        // A new input after the restart: only untouched stock can mask it.
        CHECK(L.state("m-c").value == "AVAILABLE", ctx + ": spare stock survives");
        CHECK_OK(L.reserve({"m-c"}, "g1", 4), ctx + ": new input reserves fresh material");
        CHECK_OK(L.spend({"m-c"}), ctx + ": new input spends fresh material");
        CHECK_OK(L.record_send("g1", 8, "digest-of-a-new-input"), ctx + ": new input under a new seq");
        CHECK(replay_model(complete_lines(read_file(path))).violation.empty(), ctx + ": journal audit after the new input");

        // A second restart changes nothing (recover is idempotent).
        const std::string s_a = L.state("m-a").value, s_b = L.state("m-b").value;
        MaterialLedger L3(path, binding_for("v0"));
        const auto rec3 = L3.recover();
        CHECK_OK(rec3, ctx + ": second recover");
        CHECK(rec3.value.get_int("reserved_burned").value == 0, ctx + ": second recover burns nothing");
        CHECK(L3.state("m-a").value == s_a && L3.state("m-b").value == s_b && L3.state("m-c").value == "SPENT" &&
                  L3.state("m-old").value == "SPENT",
              ctx + ": states stable across restarts");
        CHECK(L3.retransmit_allowed("g1", 8, "digest-of-a-new-input"), ctx + ": post-restart send log durable");
        std::remove(path.c_str());
    }
}

// Kill at EVERY record boundary and inside every record of the complete scenario (first byte,
// middle, last byte before the newline): the restart must equal the independent model of the
// complete lines, with RESERVED burned and the incomplete record dropped.
void test_crash_every_cut() {
    const std::string full_path = temp_path("cut_full");
    {
        MaterialLedger L(full_path, binding_for("v0"));
        run_gate(L, Boundary::COMMIT);
    }
    const std::string J = read_file(full_path);
    std::remove(full_path.c_str());
    std::vector<size_t> starts = {0};
    for (size_t i = 0; i < J.size(); ++i)
        if (J[i] == '\n') starts.push_back(i + 1);
    CHECK(starts.size() == 14 + 1, "the scenario writes 14 records");   // 1 binding, 4 + 3 adds, 2 reserve, 2 spend, 1 send, 1 commit
    std::vector<size_t> cuts;
    for (size_t li = 0; li + 1 < starts.size(); ++li) {
        const size_t lo = starts[li], hi = starts[li + 1];   // record occupies [lo, hi), newline at hi - 1
        cuts.push_back(lo);              // kill exactly between two records
        cuts.push_back(lo + 1);          // first byte of the record written
        cuts.push_back((lo + hi) / 2);   // middle of the record
        cuts.push_back(hi - 1);          // complete JSON, newline missing
    }
    cuts.push_back(J.size());
    size_t torn_seen = 0, burned_seen = 0;
    for (size_t cut : cuts) {
        const std::string path = temp_path("cut");
        const std::string prefix = J.substr(0, cut);
        write_file(path, prefix);
        const std::string ctx = "cut at byte " + std::to_string(cut);
        const std::string::size_type last_nl = prefix.rfind('\n');
        const size_t complete_len = last_nl == std::string::npos ? 0 : last_nl + 1;
        const bool torn = complete_len != prefix.size();
        const Model model = replay_model(complete_lines(prefix));
        CHECK(model.violation.empty(), ctx + ": prefix model clean");
        const std::map<std::string, std::string> want = expected_after_restart(model);
        {
            MaterialLedger L(path, binding_for("v0"));
            const auto rec = L.recover();
            CHECK_OK(rec, ctx + ": recover");
            CHECK((rec.value.get_int("torn_tail").value == 1) == torn, ctx + ": torn tail detected exactly when the cut is inside a record");
            if (torn) {
                ++torn_seen;
                CHECK(rec.value.get_int("torn_tail_bytes").value == static_cast<int64_t>(prefix.size() - complete_len), ctx + ": torn bytes");
                CHECK(!L.notes().empty(), ctx + ": LEDGER_TORN_TAIL note");
            }
            int64_t reserved = 0;
            for (const auto& kv : model.state)
                if (kv.second == "RESERVED") ++reserved;
            burned_seen += static_cast<size_t>(reserved);
            CHECK(rec.value.get_int("reserved_burned").value == reserved, ctx + ": exactly the RESERVED entries are burned");
            for (const char* id : {"m-old", "m-a", "m-b", "m-c"}) {
                const auto st = L.state(id);
                if (want.count(id)) {
                    CHECK(st.ok && st.value == want.at(id), ctx + ": state of " + id + " equals the model");
                } else {
                    CHECK(!st.ok, ctx + ": " + id + " unknown, as in the model");
                }
                // I1 at every cut: once SPENT in the complete records, SPENT after the restart
                if (model.state.count(id) && model.state.at(id) == "SPENT") {
                    CHECK(st.value == "SPENT", ctx + ": SPENT is never lost");
                    CHECK_CODE(L.reserve({id}, "g9", 9), "MATERIAL_REUSE", ctx + ": SPENT is never returned");
                }
            }
            CHECK(L.retransmit_allowed("g1", 7, kDigest) == (model.send_log.count("g1/7") == 1), ctx + ": send log equals the model");
            CHECK(L.retransmit_allowed("g0", 0, "digest-old") == (model.send_log.count("g0/0") == 1), ctx + ": earlier send log equals the model");
            CHECK(L.installed(kObject, object_version("v0")) == (model.installed.count(kObject) == 1), ctx + ": installed equals the model");
            CHECK(!L.installed(kObject, object_version("v1")), ctx + ": never installed for v1");
            // I2: a message is logged only when its material is durably SPENT in the same prefix
            if (model.send_log.count("g1/7"))
                CHECK(model.state.at("m-a") == "SPENT" && model.state.at("m-b") == "SPENT", ctx + ": a logged message implies durable SPENT");
        }
        const std::string after = read_file(path);
        CHECK(after.compare(0, complete_len, prefix, 0, complete_len) == 0, ctx + ": complete records preserved byte for byte");
        CHECK(after.empty() || after.back() == '\n', ctx + ": the journal ends at a record boundary");
        const Model audit = replay_model(complete_lines(after));
        CHECK(audit.violation.empty(), ctx + ": journal audit after recovery: " + audit.violation);
        CHECK(audit.notes == (torn ? 1 : 0), ctx + ": one NOTE record per torn tail");
        for (const auto& kv : audit.state) CHECK(kv.second != "RESERVED", ctx + ": no RESERVED record state after recovery");
        std::remove(path.c_str());
    }
    CHECK(torn_seen == 14 * 3, "every in-record cut was exercised");
    CHECK(burned_seen > 0, "some cuts left RESERVED entries to burn");
    std::fprintf(stderr, "[test_ledger] crash cuts: %zu restarts, %zu torn tails dropped, %zu reserved entries burned\n", cuts.size(), torn_seen,
                 burned_seen);
}

// ---------------------------------------------------------------------------------------------
// Two-party rig (loopback channel, in-process dealer link, real Party gadgets)
// ---------------------------------------------------------------------------------------------
// Records every frame a party sends and receives and calls a hook BEFORE a frame leaves.
class SpyChannel : public Channel {
public:
    explicit SpyChannel(Channel& inner) : Channel(inner.party), inner_(inner) {}
    std::function<void(const Value&)> before_send;
    std::vector<Value> sent;
    std::vector<Value> received;

    Expected<Unit> send(const Value& obj) override {
        if (before_send) before_send(obj);
        sent.push_back(obj);
        auto r = inner_.send(obj);
        inner_.clear_pending();   // rounds are accounted at this level
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
        if (before_send) before_send(obj);
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
    Value binding;
};

// journal paths may be passed in to restart a party over an existing journal (recover).
std::unique_ptr<Rig> make_rig(const std::string& model_version, uint64_t dealer_seed, const std::string* reuse_paths = nullptr) {
    auto rig = std::make_unique<Rig>();
    rig->dc.setup_id = "setup-" + model_version;
    rig->dc.model_version = model_version;
    rig->dc.numeric_contract_hash = rig->contract.contract_hash();
    rig->key[0] = Bytes(32, 0x11);
    rig->key[1] = Bytes(32, 0x22);
    rig->dealer = std::make_shared<Dealer>(rig->dc, ll::Rng::from_seed(dealer_seed), rig->key[0], rig->key[1]);
    rig->link = std::make_shared<InProcessDealerLink>(rig->dealer);
    auto pair = loopback_pair();
    CHECK_OK(pair, "loopback pair");
    rig->raw[0] = std::move(pair.value.first);
    rig->raw[1] = std::move(pair.value.second);
    rig->binding = dict({{"session_id", "sess-" + model_version}, {"setup_id", rig->dc.setup_id}, {"model_version", model_version},
                         {"numeric_contract_hash", rig->dc.numeric_contract_hash}, {"profile_id", rig->dc.profile_id}});
    for (int p = 0; p < 2; ++p) {
        rig->ch[p] = std::make_unique<SpyChannel>(*rig->raw[p]);
        rig->paths[p] = reuse_paths ? reuse_paths[p] : temp_path("rig_p" + std::to_string(p));
        rig->ledger[p] = std::make_unique<MaterialLedger>(rig->paths[p], rig->binding);
        if (reuse_paths) {
            CHECK_OK(rig->ledger[p]->recover(), "rig recover");
        } else {
            CHECK_OK(rig->ledger[p]->open(), "rig open");
        }
        rig->client[p] = std::make_unique<DealerClient>(p, *rig->ledger[p], rig->key[p], rig->dc.numeric_contract_hash);
        rig->link->attach(*rig->client[p]);
    }
    return rig;
}

using Body = std::function<Expected<Value>(Party&, int)>;

std::array<Expected<Value>, 2> run_both(Rig& rig, const Body& body, const std::string& client_key_domain = "client-0") {
    std::array<Expected<Value>, 2> out;
    auto run = [&](int p) {
        ReleasePolicy rel(1, {"token"}, 1);
        rig.client[p]->set_gate("gate", 0);
        Party party(p, *rig.ch[p], *rig.client[p], rig.contract, rel, ll::Rng::from_seed(500 + static_cast<uint64_t>(p)), rig.dc.model_version,
                    client_key_domain);
        out[p] = body(party, p);
        if (!out[p].ok) rig.ch[p]->close();   // unblock the peer (fail closed)
    };
    std::thread t0(run, 0);
    std::thread t1(run, 1);
    t0.join();
    t1.join();
    return out;
}

// I2 at the gadget level: when the first frame of a Beaver product leaves, the bundle is SPENT in
// memory AND its SPENT record is already in the journal file.
void test_spend_before_first_message() {
    auto rig = make_rig("v0", 77);
    const size_t n = 16;
    const auto bp = rig->dealer->make_beaver(n);
    CHECK_OK(bp, "make beaver");
    CHECK_OK(rig->client[0]->ingest(bp.value.for0), "ingest P0");
    CHECK_OK(rig->client[1]->ingest(bp.value.for1), "ingest P1");
    const std::string mid = bp.value.for0.header.get_string("material_id").value;
    CHECK(rig->ledger[0]->state(mid).value == "AVAILABLE", "bundle AVAILABLE before the gate");

    struct Seen {
        int frames = 0;
        std::string state_at_first_send;
        bool spent_on_disk = false;
        bool reserved_on_disk = false;
        int spent_line = -1, lines = 0;
    } seen[2];
    for (int p = 0; p < 2; ++p) {
        MaterialLedger* ledger = rig->ledger[p].get();
        const std::string path = rig->paths[p];
        Seen* s = &seen[p];
        rig->ch[p]->before_send = [ledger, path, mid, s](const Value&) {
            if (s->frames++ > 0) return;
            s->state_at_first_send = ledger->state(mid).value;
            const std::vector<std::string> lines = complete_lines(read_file(path));   // what a kill right now would leave
            s->lines = static_cast<int>(lines.size());
            for (size_t i = 0; i < lines.size(); ++i) {
                const auto v = parse_json(lines[i]);
                if (!v.ok || v.value.get_string("material_id").value != mid) continue;
                if (v.value.get_string("state").value == "RESERVED") s->reserved_on_disk = true;
                if (v.value.get_string("state").value == "SPENT") {
                    s->spent_on_disk = true;
                    s->spent_line = static_cast<int>(i);
                }
            }
        };
    }
    std::vector<uint64_t> xs(n), ys(n);
    for (size_t i = 0; i < n; ++i) {
        xs[i] = 3 * i + 1;
        ys[i] = 1000 + i;
    }
    const auto out = run_both(*rig, [&](Party& party, int p) -> Expected<Value> {
        // public test shares: P0 holds x and y, P1 holds zeros
        Mat sx = Mat::row_vector(p == 0 ? xs : std::vector<uint64_t>(n, 0));
        Mat sy = Mat::row_vector(p == 0 ? ys : std::vector<uint64_t>(n, 0));
        OC2_TRY(z, party.beaver_mul(sx, sy));
        return ok(u64_array(z.a, {n}));
    });
    CHECK_OK(out[0], "beaver_mul P0");
    CHECK_OK(out[1], "beaver_mul P1");
    const std::vector<uint64_t> z0 = out[0].value.as_array().value.as_u64(), z1 = out[1].value.as_array().value.as_u64();
    for (size_t i = 0; i < n; ++i) CHECK(z0[i] + z1[i] == xs[i] * ys[i], "beaver product reconstructs");
    for (int p = 0; p < 2; ++p) {
        CHECK(seen[p].frames == 1, "one outgoing frame per party for the opening");
        CHECK(seen[p].state_at_first_send == "SPENT", "bundle is SPENT in memory before the first frame leaves");
        CHECK(seen[p].reserved_on_disk && seen[p].spent_on_disk, "RESERVED and SPENT records are on disk before the first frame leaves");
        CHECK(seen[p].spent_line == seen[p].lines - 1, "the SPENT record is the last record at that moment");
        CHECK(rig->ledger[p]->state(mid).value == "SPENT", "bundle SPENT after the gate");
        CHECK_CODE(rig->ledger[p]->reserve({mid}, "again", 1), "MATERIAL_REUSE", "the bundle is never returned");
    }
    for (int p = 0; p < 2; ++p) {
        rig->ledger[p].reset();
        std::remove(rig->paths[p].c_str());
    }
}

// ---------------------------------------------------------------------------------------------
// 2b. Reopen, binding, strict transitions, repeated ids
// ---------------------------------------------------------------------------------------------
void test_reopen_binding_and_duplicates() {
    const Value ver = material_version("v0");
    // (a) open() replays an existing journal: a SPENT id is never accepted again
    {
        const std::string path = temp_path("reopen");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open");
            CHECK_OK(L.add_available("a", "beaver", 10, ver), "add a");
            CHECK_OK(L.add_available("b", "beaver", 10, ver), "add b");
            CHECK_OK(L.reserve({"a"}, "g", 0), "reserve a");
            CHECK_OK(L.spend({"a"}), "spend a");
            CHECK_OK(L.burn({"b"}, "x"), "burn b");
            CHECK_OK(L.record_send("g", 1, "aa"), "send");
        }
        const std::string first_line = complete_lines(read_file(path))[0];
        const auto bl = parse_json(first_line);
        CHECK(bl.ok && bl.value.get_string("state").value == "BINDING" && bl.value.get_string("session_id").value == "sess-1" &&
                  bl.value.get_string("setup_id").value == "setup-A" && bl.value.get_string("model_version").value == "v0" &&
                  bl.value.get_string("numeric_contract_hash").value == "hash-1" &&
                  bl.value.get_string("profile_id").value == "SH2P-TrustedOffline-v1",
              "the first journal line carries the five binding fields");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open of an existing journal (no recover call)");
            CHECK(!L.needs_recovery(), "nothing RESERVED");
            CHECK(L.state("a").value == "SPENT" && L.state("b").value == "BURNED", "open() replayed the journal");
            CHECK_CODE(L.add_available("a", "beaver", 10, ver), "MATERIAL_REUSE", "a SPENT id is refused after a reopen");
            CHECK_CODE(L.add_available("b", "beaver", 10, ver), "MATERIAL_REUSE", "a BURNED id is refused after a reopen");
            CHECK_CODE(L.reserve({"a"}, "g", 1), "MATERIAL_REUSE", "and never reserved again");
            CHECK(L.bytes_consumed() == 10 && L.bytes_burned() == 10, "counters rebuilt by open()");
            CHECK_CODE(L.record_send("g", 1, "bb"), "LEDGER_SEQ_REUSE", "the send log survives a reopen without recover");
            CHECK_OK(L.add_available("c", "beaver", 10, ver), "new ids are accepted");
            const auto rec = L.recover();
            CHECK_OK(rec, "a later recover() replays cleanly");
            CHECK(rec.value.get_int("spent").value == 1 && rec.value.get_int("available").value == 1, "consistent counts");
        }
        CHECK(replay_model(complete_lines(read_file(path))).violation.empty(), "file audit clean");
        // open() with a torn fragment at the end: dropped before the first append
        append_file(path, "{\"material_id\": \"c\", \"state\": \"RESER");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open with a torn tail");
            CHECK(L.notes().size() == 1, "torn tail noted by open()");
            CHECK_OK(L.add_available("d", "beaver", 10, ver), "append after the torn tail");
            CHECK_OK(L.add_available("e", "beaver", 10, ver), "second append");
            CHECK_OK(L.recover(), "recover after appends over a dropped tail");
        }
        CHECK(replay_model(complete_lines(read_file(path))).violation.empty(), "file audit clean after the torn tail");
        // RESERVED at open: every state change waits for recover()
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open");
            CHECK_OK(L.reserve({"c"}, "g", 2), "reserve c");
        }
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open with a RESERVED entry");
            CHECK(L.needs_recovery(), "needs recovery");
            CHECK_CODE(L.add_available("f", "beaver", 10, ver), "LEDGER_RECOVERY_REQUIRED", "add refused before recover");
            CHECK_CODE(L.spend({"c"}), "LEDGER_RECOVERY_REQUIRED", "spend refused before recover");
            CHECK_CODE(L.reserve({"d"}, "g", 3), "LEDGER_RECOVERY_REQUIRED", "reserve refused before recover");
            CHECK_CODE(L.record_send("g", 9, "zz"), "LEDGER_RECOVERY_REQUIRED", "record_send refused before recover");
            const auto rec = L.recover();
            CHECK(rec.ok && rec.value.get_int("reserved_burned").value == 1 && !L.needs_recovery(), "recover burns it");
            CHECK(L.state("c").value == "BURNED", "burned");
            CHECK_OK(L.add_available("f", "beaver", 10, ver), "usable after recover");
        }
        const auto audit = MaterialLedger::audit_file(path);
        CHECK(audit.ok && audit.value.get_int("reserved").value == 0 && audit.value.get_int("spent").value == 1, "read-only audit");
        std::remove(path.c_str());
    }
    // (b) binding: the journal of another session, setup, model version, contract or profile is refused
    {
        const std::string path = temp_path("binding");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open");
            CHECK_OK(L.add_available("a", "beaver", 10, ver), "add a");
            CHECK_OK(L.reserve({"a"}, "g", 0), "reserve a");
        }
        const std::string before = read_file(path);
        const char* keys[] = {"session_id", "setup_id", "model_version", "numeric_contract_hash", "profile_id"};
        for (const char* k : keys) {
            Value other = binding_for("v0");
            other.set(k, Value("OTHER"));
            MaterialLedger L(path, other);
            CHECK_CODE(L.recover(), "LEDGER_BINDING_MISMATCH", "recover refuses a foreign journal");
            CHECK_CODE(L.open(), "LEDGER_BINDING_MISMATCH", "open refuses a foreign journal");
            CHECK_CODE(L.state("a"), "LEDGER_UNKNOWN_ID", "nothing of the foreign journal is adopted");
        }
        CHECK(read_file(path) == before, "a refused journal is byte-identical (nothing burned, nothing appended)");
        // a journal with records but no binding line is not a journal of this code
        const std::vector<std::string> lines = complete_lines(before);
        std::string headless;
        for (size_t i = 1; i < lines.size(); ++i) headless += lines[i] + "\n";
        write_file(path, headless);
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "journal without a BINDING record refused");
        }
        write_file(path, before + lines[0] + "\n");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_CODE(L.recover(), "LEDGER_CORRUPT", "second BINDING record refused");
        }
        std::remove(path.c_str());
    }
    // (c) add_available checks the bundle version against the binding
    {
        const std::string path = temp_path("version");
        MaterialLedger L(path, binding_for("v0"));
        CHECK_OK(L.open(), "open");
        const char* keys[] = {"setup_id", "model_version", "numeric_contract_hash", "profile_id"};
        for (const char* k : keys) {
            Value v = material_version("v0");
            v.set(k, Value("OTHER"));
            CHECK_CODE(L.add_available("z", "beaver", 10, v), "MATERIAL_VERSION_MISMATCH", "foreign bundle version refused");
        }
        CHECK_CODE(L.add_available("z", "beaver", 10, dict({{"setup_id", "setup-A"}})), "MATERIAL_VERSION_MISMATCH",
                   "version with missing fields refused");
        CHECK_CODE(L.add_available("z", "beaver", 10, Value("v0")), "MATERIAL_VERSION_MISMATCH", "version must be a dict");
        CHECK_CODE(L.add_available("", "beaver", 10, ver), "LEDGER_PARAMS", "empty id refused");
        CHECK_CODE(L.state("z"), "LEDGER_UNKNOWN_ID", "a refused bundle leaves no entry");
        CHECK(complete_lines(read_file(path)).size() == 1, "and no journal line");
        CHECK_OK(L.add_available("z", "beaver", 10, ver), "the bound version is accepted");
        std::remove(path.c_str());
    }
    // (d) replay accepts only the transitions the code writes
    {
        const std::string path = temp_path("lax");
        std::string binding_line, avail, reserved, spent, burned;
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open");
            CHECK_OK(L.add_available("a", "beaver", 10, ver), "add");
            CHECK_OK(L.reserve({"a"}, "g", 0), "reserve");
            CHECK_OK(L.spend({"a"}), "spend");
        }
        {
            const std::vector<std::string> lines = complete_lines(read_file(path));
            CHECK(lines.size() == 4, "four lines");
            binding_line = lines[0]; avail = lines[1]; reserved = lines[2]; spent = lines[3];
            burned = spent;
            burned.replace(burned.find("SPENT"), 5, "BURNED");
        }
        struct Case { std::string body; const char* what; bool ok; };
        const Case cases[] = {
            {avail + "\n" + spent + "\n", "AVAILABLE to SPENT without RESERVED", false},
            {avail + "\n" + reserved + "\n" + avail + "\n", "RESERVED back to AVAILABLE", false},
            {avail + "\n" + avail + "\n", "duplicate AVAILABLE", false},
            {avail + "\n" + reserved + "\n" + reserved + "\n", "RESERVED twice", false},
            {avail + "\n" + burned + "\n" + burned + "\n", "BURNED twice", false},
            {avail + "\n" + burned + "\n" + reserved + "\n", "RESERVED after BURNED", false},
            {avail + "\n" + burned + "\n", "AVAILABLE to BURNED", true},
            {avail + "\n" + reserved + "\n" + burned + "\n", "RESERVED to BURNED", true},
            {avail + "\n" + reserved + "\n" + spent + "\n", "the full lifecycle", true},
        };
        for (const Case& c : cases) {
            write_file(path, binding_line + "\n" + c.body);
            MaterialLedger L(path, binding_for("v0"));
            const auto rec = L.recover();
            if (c.ok) CHECK_OK(rec, c.what);
            else CHECK_CODE(rec, "LEDGER_CORRUPT", c.what);
        }
        std::remove(path.c_str());
    }
    // (e) repeated ids: one state line per id
    {
        const std::string path = temp_path("dup");
        {
            MaterialLedger L(path, binding_for("v0"));
            CHECK_OK(L.open(), "open");
            CHECK_OK(L.add_available("a", "beaver", 10, ver), "add a");
            CHECK_OK(L.add_available("b", "beaver", 10, ver), "add b");
            CHECK_OK(L.add_available("c", "beaver", 10, ver), "add c");
            CHECK_CODE(L.reserve({"a", "a"}, "g", 0), "LEDGER_PARAMS", "reserve refuses a repeated id");
            CHECK(L.state("a").value == "AVAILABLE", "nothing reserved by the refused call");
            CHECK_OK(L.reserve({"a", "b"}, "g", 0), "reserve");
            CHECK_OK(L.spend({"a", "a"}), "spend with a repeated id");
            CHECK(L.bytes_consumed() == 10, "a 10 byte entry is charged once");
            CHECK_OK(L.burn({"b", "b", "c", "b", "c"}, "x"), "burn with repeated ids");
            CHECK(L.bytes_burned() == 20, "burned bytes charged once per id");
        }
        const Model m = replay_model(complete_lines(read_file(path)));
        CHECK(m.violation.empty() && m.spent_lines.at("a") == 1, "one SPENT line, audit clean");
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, "the journal replays after repeated-id calls");
        CHECK(rec.value.get_int("spent").value == 1 && rec.value.get_int("burned").value == 2, "counts");
        std::remove(path.c_str());
    }
}

// ---------------------------------------------------------------------------------------------
// 3. wmask version binding: E = W - B is never reused across model versions
// ---------------------------------------------------------------------------------------------
void test_wmask_version_binding() {
    // --- ledger level: versioned installed objects across a restart ---------------------------
    const std::string path = temp_path("wmask");
    {
        MaterialLedger L(path, binding_for("v0"));
        CHECK_OK(L.open(), "open");
        CHECK(!L.installed(kObject, object_version("v0")), "nothing installed yet");
        CHECK_OK(L.register_installed(kObject, object_version("v0")), "install recorded for v0");
        CHECK(L.installed(kObject, object_version("v0")), "installed for v0");
        CHECK(!L.installed(kObject, object_version("v1")), "an object recorded for v0 is not installed for v1");
        Value other_domain = dict({{"model_version", "v0"}, {"client_key_domain", "client-1"}});
        CHECK(!L.installed(kObject, other_domain), "nor for another client key domain");
        CHECK(!L.installed(kObject, dict({{"model_version", "v0"}})), "nor for a version dict with a missing field");
        CHECK(!L.installed("wmask_install(other)", object_version("v0")), "nor for another object id");
    }
    {
        // The upgraded session must not adopt the journal of the v0 session: the journal is bound
        // to (session, setup, model version, contract, profile) by its first line.
        const std::string before = read_file(path);
        MaterialLedger L(path, binding_for("v1"));
        CHECK_CODE(L.recover(), "LEDGER_BINDING_MISMATCH", "recover under the v1 binding refused");
        CHECK_CODE(L.open(), "LEDGER_BINDING_MISMATCH", "open under the v1 binding refused");
        CHECK_CODE(L.register_installed(kObject, object_version("v1")), "LEDGER_NOT_OPEN", "no state change on a foreign journal");
        CHECK(read_file(path) == before, "the foreign journal is left byte-identical");
    }
    {
        // a restart of the SAME session replays the install record; a re-install under a new
        // object version (another client key domain) replaces it
        MaterialLedger L(path, binding_for("v0"));
        const auto rec = L.recover();
        CHECK_OK(rec, "recover under the v0 binding");
        CHECK(rec.value.get_int("installed").value == 1, "the v0 install record is replayed");
        CHECK(L.installed(kObject, object_version("v0")), "installed for v0 after the restart");
        CHECK(!L.installed(kObject, object_version("v1")), "after the restart: still not installed for v1");
        // bundles of another model version cannot enter the v0 session
        CHECK_CODE(L.check_binding(binding_for("v1")), "MATERIAL_VERSION_MISMATCH", "v1 bundle header rejected by the v0 ledger");
        CHECK_OK(L.register_installed(kObject, object_version("v1")), "re-install recorded under another object version");
        CHECK(L.installed(kObject, object_version("v1")), "installed for the new object version");
        CHECK(!L.installed(kObject, object_version("v0")), "and no longer reported for the old one");
    }
    {
        MaterialLedger L(path, binding_for("v0"));
        CHECK_OK(L.recover(), "second restart");
        CHECK(L.installed(kObject, object_version("v1")) && !L.installed(kObject, object_version("v0")), "the latest version wins after a restart");
    }
    std::remove(path.c_str());

    // --- protocol level: two real installs of the same matrix id under v0 and v1 ---------------
    const size_t d = 6, m = 5;
    ll::Rng wrng = ll::Rng::from_seed(4242);
    Mat W0(d, m), W1(d, m);
    for (auto& v : W0.a) v = wrng.next_u64() >> 24;   // any ring values; the protocol is exact mod 2^64
    for (auto& v : W1.a) v = wrng.next_u64() >> 24;
    const std::string mat_id = "W";
    const std::string object_id = Party::install_object_id(mat_id);

    auto install = [&](Rig& rig, const Mat& W, Mat& E_seen_by_p1, bool expect_skip_second) {
        const auto out = run_both(rig, [&](Party& party, int p) -> Expected<Value> {
            if (p == 1) party.set_vector_share(mat_id, Mat(d, m));   // P1 knows the public shape only
            const bool before = party.installed(mat_id);
            OC2_TRY(rec, party.install_wmask(mat_id, p == 0 ? &W : nullptr));
            const bool after = party.installed(mat_id);
            OC2_TRY(again, party.install_wmask(mat_id, p == 0 ? &W : nullptr));   // same version: no second protocol run
            return ok(dict({{"before", before}, {"after", after}, {"first", rec}, {"second", again}}));
        });
        CHECK_OK(out[0], "install P0");
        CHECK_OK(out[1], "install P1");
        for (int p = 0; p < 2; ++p) {
            CHECK(!out[p].value.get("before")->as_bool().value, "not installed before the protocol");
            CHECK(out[p].value.get("after")->as_bool().value, "installed after the protocol");
            CHECK(!out[p].value.get("first")->get("skipped")->as_bool().value, "the install really ran");
            CHECK(out[p].value.get("second")->get("skipped")->as_bool().value == expect_skip_second, "same-version re-install is skipped");
        }
        // P1 receives exactly one frame in the install: E = W - B
        CHECK(rig.ch[1]->received.size() == 1, "P1 receives one frame (E)");
        const NdArray arr = rig.ch[1]->received[0].as_array().value;
        CHECK(arr.numel() == d * m, "E has the mask shape");
        E_seen_by_p1 = Mat::from_vector(arr.as_u64(), d, m);
    };

    auto rig0 = make_rig("v0", 1001);
    Mat E0;
    install(*rig0, W0, E0, true);
    const Value ver0 = object_version("v0"), ver1 = object_version("v1");
    for (int p = 0; p < 2; ++p) {
        CHECK(rig0->ledger[p]->installed(object_id, ver0), "v0 session: object installed for v0");
        CHECK(!rig0->ledger[p]->installed(object_id, ver1), "v0 session: object not installed for v1");
    }
    // the v0 mask bundle (the reusable B) is refused by a v1 dealer client: B0 cannot be carried over
    const auto b0_bundle = rig0->dealer->make_wmask_mask("W-probe", d, m);
    CHECK_OK(b0_bundle, "a v0 mask bundle");
    const std::string paths[2] = {rig0->paths[0], rig0->paths[1]};
    for (int p = 0; p < 2; ++p) rig0->ledger[p].reset();   // the v0 processes end

    // model upgrade: the journals of the v0 session are bound to (sess-v0, setup-v0, v0) by their
    // first line and cannot be adopted under a v1 binding; the v1 session starts new journals
    {
        Value b1 = dict({{"session_id", "sess-v1"}, {"setup_id", "setup-v1"}, {"model_version", "v1"},
                         {"numeric_contract_hash", rig0->dc.numeric_contract_hash}, {"profile_id", rig0->dc.profile_id}});
        for (int p = 0; p < 2; ++p) {
            MaterialLedger foreign(paths[p], b1);
            CHECK_CODE(foreign.recover(), "LEDGER_BINDING_MISMATCH", "the v0 journal is refused under the v1 binding");
            CHECK(!foreign.installed(object_id, ver0) && !foreign.installed(object_id, ver1), "nothing of it is adopted");
        }
    }
    auto rig1 = make_rig("v1", 2002);
    for (int p = 0; p < 2; ++p)
        CHECK(!rig1->ledger[p]->installed(object_id, ver1) && !rig1->ledger[p]->installed(object_id, ver0),
              "v1 session: nothing is installed before its own install");
    CHECK_CODE(rig1->client[0]->ingest(b0_bundle.value.for0), "MATERIAL_VERSION_MISMATCH", "v0 mask bundle refused by the v1 client (P0)");
    CHECK_CODE(rig1->client[1]->ingest(b0_bundle.value.for1), "MATERIAL_VERSION_MISMATCH", "v0 mask bundle refused by the v1 client (P1)");
    Mat E1;
    install(*rig1, W1, E1, true);
    for (int p = 0; p < 2; ++p) {
        CHECK(rig1->ledger[p]->installed(object_id, ver1), "v1 session: installed for v1 after the re-install");
        CHECK(!rig1->ledger[p]->installed(object_id, ver0), "v1 session: no longer reported for v0");
    }
    // The leak the version binding prevents: with a reused mask B, P1 could form
    // E1 - E0 = (W1 - B) - (W0 - B) = W1 - W0. With a fresh mask the difference is masked.
    size_t leaking = 0, same_mask = 0;
    for (size_t i = 0; i < d * m; ++i) {
        if (E1.a[i] - E0.a[i] == W1.a[i] - W0.a[i]) ++leaking;
        if (W0.a[i] - E0.a[i] == W1.a[i] - E1.a[i]) ++same_mask;   // B0 == B1 at this entry
    }
    CHECK(leaking == 0, "E1 - E0 equals W1 - W0 at no entry");
    CHECK(same_mask == 0, "the v1 install drew a fresh mask at every entry");
    // arithmetic control: reusing B0 would have leaked every entry
    size_t would_leak = 0;
    for (size_t i = 0; i < d * m; ++i) {
        const uint64_t b0 = W0.a[i] - E0.a[i];
        const uint64_t e1_reused = W1.a[i] - b0;
        if (e1_reused - E0.a[i] == W1.a[i] - W0.a[i]) ++would_leak;
    }
    CHECK(would_leak == d * m, "control: a reused mask leaks W1 - W0 at every entry");
    for (int p = 0; p < 2; ++p) {
        const std::string p1 = rig1->paths[p];
        rig1->client[p].reset();
        rig1->ledger[p].reset();
        std::remove(p1.c_str());
        std::remove(paths[p].c_str());
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Decoder limits
// ---------------------------------------------------------------------------------------------
Bytes nested_lists(int depth) {
    Bytes b;
    for (int i = 0; i < depth; ++i) {
        b.push_back(0x07);
        put_u32be(b, 1);
    }
    b.push_back(0x08);   // none
    return b;
}

Bytes nested_dicts(int depth) {
    Bytes b;
    for (int i = 0; i < depth; ++i) {
        b.push_back(0x03);
        put_u32be(b, 1);
        put_u16be(b, 1);
        b.push_back('k');
    }
    b.push_back(0x08);
    return b;
}

#ifdef _WIN32
using raw_socket_t = SOCKET;
void raw_close(raw_socket_t s) { closesocket(s); }
const int kRawSendFlags = 0;
#else
using raw_socket_t = int;
void raw_close(raw_socket_t s) { ::close(s); }
const int kRawSendFlags = MSG_NOSIGNAL;   // a closed peer is an error return, not SIGPIPE
#endif

// A raw TCP client connected to a listening Channel: the test plays a misbehaving peer.
raw_socket_t raw_connect(int port) {
    raw_socket_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) die("raw connect", __LINE__, "");
    return s;
}

void raw_send(raw_socket_t s, const Bytes& b) {
    size_t at = 0;
    while (at < b.size()) {
        const int r = static_cast<int>(::send(s, reinterpret_cast<const char*>(b.data() + at), static_cast<int>(b.size() - at), kRawSendFlags));
        if (r <= 0) die("raw send", __LINE__, "");
        at += static_cast<size_t>(r);
    }
}

struct RawPeer {
    std::unique_ptr<Channel> channel;   // the party under test
    raw_socket_t peer;                  // the misbehaving peer
};

RawPeer make_raw_peer() {
    auto l = listen_socket("127.0.0.1", 0);
    CHECK_OK(l, "listen");
    RawPeer rp;
    rp.peer = raw_connect(l.value.port);
    auto ch = accept_channel(l.value, 0);
    CHECK_OK(ch, "accept");
    rp.channel = std::move(ch.value);
    return rp;
}

void test_decoder_limits() {
    // --- wire codec nesting --------------------------------------------------------------------
    CHECK(CODEC_MAX_DEPTH == 64, "the codec depth limit is 64");
    CHECK_OK(decode_value(nested_lists(64)), "64 nested lists decode");
    CHECK_CODE(decode_value(nested_lists(65)), "CODEC_DEPTH", "65 nested lists refused");
    CHECK_OK(decode_value(nested_dicts(64)), "64 nested dicts decode");
    CHECK_CODE(decode_value(nested_dicts(65)), "CODEC_DEPTH", "65 nested dicts refused");
    CHECK(decode_value(nested_lists(65)).code_prefix() == "CODEC_DEPTH", "stable code prefix");
    {
        // 50000 nested one-element lists (250 KB) crashed the first version
        const Bytes deep = nested_lists(50000);
        CHECK(deep.size() == 250001, "probe payload size");
        CHECK_CODE(decode_value(deep), "CODEC_DEPTH", "50000 nested lists refused without a crash");
        CHECK_CODE(decode_value(nested_dicts(200000)), "CODEC_DEPTH", "200000 nested dicts refused without a crash");
        // the same payload decoded on a worker thread (small default stack)
        Expected<Value> r;
        std::thread th([&]() { r = decode_value(deep); });
        th.join();
        CHECK_CODE(r, "CODEC_DEPTH", "refused on a worker thread");
    }
    {
        // siblings do not count as depth: a wide value at depth 64 still decodes, and a value
        // encoded by this layer round trips at the limit
        Value v = Value(List{Value(static_cast<int64_t>(1)), Value("x")});
        for (int i = 0; i < 63; ++i) v = Value(List{v, Value(static_cast<int64_t>(i)), dict({{"k", Value(true)}})});
        const Bytes enc = encode_value(v);
        const auto dec = decode_value(enc);
        CHECK_OK(dec, "depth-64 value with siblings decodes");
        CHECK(encode_value(dec.value) == enc, "round trip at the depth limit");
        const Bytes over = encode_value(Value(List{v}));
        CHECK_CODE(decode_value(over), "CODEC_DEPTH", "one more level refused");
    }
    CHECK(decode_value(Bytes{0x42}).code == "CODEC_TAG", "unknown tag still refused");
    // malformed frames that the Python decoder refuses are refused here too
    {
        CHECK_OK(decode_value(Bytes{0x09, 0x00}), "bool 0");
        CHECK_OK(decode_value(Bytes{0x09, 0x01}), "bool 1");
        CHECK_CODE(decode_value(Bytes{0x09, 0x02}), "CODEC_MALFORMED", "bool byte 2 refused");
        CHECK_CODE(decode_value(Bytes{0x09, 0xff}), "CODEC_MALFORMED", "bool byte 255 refused");
        auto ndarray = [](uint8_t ndim, const std::vector<uint64_t>& dims) {
            Bytes b{0x02, 0x01, ndim};
            for (uint64_t d : dims)
                for (int s = 56; s >= 0; s -= 8) b.push_back(static_cast<uint8_t>(d >> s));
            return b;
        };
        CHECK_OK(decode_value(ndarray(16, std::vector<uint64_t>(16, 0))), "16 dimensions decode");
        CHECK_CODE(decode_value(ndarray(17, std::vector<uint64_t>(17, 0))), "CODEC_SHAPE", "17 dimensions refused");
        CHECK_CODE(decode_value(ndarray(255, std::vector<uint64_t>(255, 0))), "CODEC_SHAPE", "255 dimensions refused");
        CHECK_CODE(decode_value(ndarray(2, {0, static_cast<uint64_t>(1) << 63})), "CODEC_SHAPE", "shape (0, 2^63) refused");
        CHECK_CODE(decode_value(ndarray(2, {0, ~static_cast<uint64_t>(0)})), "CODEC_SHAPE", "shape (0, 2^64 - 1) refused");
        CHECK_OK(decode_value(ndarray(2, {0, static_cast<uint64_t>(1) << 40})), "an empty array with a 2^40 dimension decodes");
        CHECK_CODE(decode_value(Bytes{0x06, 0, 0, 0, 2, 0xff, 0xfe}), "CODEC_MALFORMED", "str FF FE refused");
        CHECK_CODE(decode_value(Bytes{0x06, 0, 0, 0, 2, 0xc0, 0x80}), "CODEC_MALFORMED", "overlong encoding refused");
        CHECK_CODE(decode_value(Bytes{0x06, 0, 0, 0, 3, 0xed, 0xa0, 0x80}), "CODEC_MALFORMED", "surrogate refused");
        CHECK_CODE(decode_value(Bytes{0x06, 0, 0, 0, 4, 0xf4, 0x90, 0x80, 0x80}), "CODEC_MALFORMED", "code point above U+10FFFF refused");
        CHECK_CODE(decode_value(Bytes{0x06, 0, 0, 0, 1, 0xe2}), "CODEC_MALFORMED", "truncated sequence refused");
        CHECK_CODE(decode_value(Bytes{0x03, 0, 0, 0, 1, 0, 1, 0xff, 0x08}), "CODEC_MALFORMED", "dict key FF refused");
        const Bytes good_str{0x06, 0, 0, 0, 9, 'a', 0xc3, 0xa9, 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98};   // truncated 4-byte tail
        CHECK_CODE(decode_value(good_str), "CODEC_MALFORMED", "a 4-byte sequence cut after 3 bytes refused");
        const Bytes ok_str{0x06, 0, 0, 0, 10, 'a', 0xc3, 0xa9, 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80};
        const auto s = decode_value(ok_str);
        CHECK_OK(s, "1, 2, 3 and 4 byte sequences decode");
        CHECK(encode_value(s.value) == ok_str, "and re-encode byte-identically");
        const auto k = decode_value(Bytes{0x03, 0, 0, 0, 1, 0, 2, 0xc3, 0xa9, 0x09, 0x01});
        CHECK_OK(k, "a UTF-8 dict key with a bool value decodes");
    }
    CHECK_CODE(decode_value(Bytes{0x07, 0, 0, 0, 2, 0x08}), "CODEC_TRUNCATED", "truncated list still refused");

    // --- parse_json nesting (journal lines, records) -----------------------------------------------
    CHECK(JSON_PARSE_MAX_DEPTH == 64, "the parse_json depth limit is 64");
    auto arrays = [](int depth) { return std::string(static_cast<size_t>(depth), '[') + std::string(static_cast<size_t>(depth), ']'); };
    auto objects = [](int depth) {
        std::string s;
        for (int i = 0; i < depth; ++i) s += "{\"k\":";
        s += "null";
        s += std::string(static_cast<size_t>(depth), '}');
        return s;
    };
    CHECK_OK(parse_json(arrays(64)), "64 nested arrays parse");
    CHECK_CODE(parse_json(arrays(65)), "JSON_DEPTH", "65 nested arrays refused");
    CHECK_OK(parse_json(objects(64)), "64 nested objects parse");
    CHECK_CODE(parse_json(objects(65)), "JSON_DEPTH", "65 nested objects refused");
    CHECK_CODE(parse_json(std::string(100000, '[')), "JSON_DEPTH", "100000 open brackets refused without a crash");
    CHECK_CODE(parse_json(arrays(100000)), "JSON_DEPTH", "100000 nested arrays refused without a crash");
    CHECK_CODE(parse_json(objects(100000)), "JSON_DEPTH", "100000 nested objects refused without a crash");
    {
        Expected<Value> r;
        const std::string deep = arrays(100000);
        std::thread th([&]() { r = parse_json(deep); });
        th.join();
        CHECK_CODE(r, "JSON_DEPTH", "refused on a worker thread");
    }
    CHECK_CODE(parse_json("{\"a\": [1, 2"), "JSON_PARSE", "malformed JSON keeps its code");
    CHECK(parse_json("{\"x\": [1, {\"y\": [2.5, null]}]}").ok, "ordinary nested JSON parses");
    // the decision layer's Json::parse keeps its own limit and code (also on a worker thread)
    {
        Expected<Json> r;
        const std::string deep = arrays(100000);
        std::thread th([&]() { r = Json::parse(deep); });
        th.join();
        CHECK_CODE(r, "JSON_DEPTH", "Json::parse refuses 100000 nested arrays on a worker thread");
        CHECK(Json::parse(arrays(64)).ok, "Json::parse accepts ordinary nesting");
    }

    // --- frame length cap ----------------------------------------------------------------------------
    CHECK(Channel::kDefaultMaxFrameBytes == (static_cast<uint64_t>(1) << 31), "the default frame cap is 2 GiB");
    {
        // (a) announced length above the default cap, no payload at all: refused from the 8 header
        //     bytes alone, and the channel is closed (the stream cannot be resynchronized)
        RawPeer rp = make_raw_peer();
        CHECK(rp.channel->max_frame_bytes() == Channel::kDefaultMaxFrameBytes, "default cap in force");
        Bytes hdr;
        put_u64be(hdr, (static_cast<uint64_t>(1) << 31) + 1);
        raw_send(rp.peer, hdr);
        CHECK_CODE(rp.channel->recv(), "CHANNEL_FRAME_TOO_LARGE", "2 GiB + 1 announced: refused");
        CHECK(!rp.channel->is_open(), "channel closed after an oversize announcement");
        CHECK_CODE(rp.channel->recv(), "CHANNEL_CLOSED", "no further frames are read");
        CHECK(rp.channel->stats.messages_recv == 0, "the refused frame is not counted as received");
        raw_close(rp.peer);
    }
    {
        // (b) a frame announcing 4 GiB
        RawPeer rp = make_raw_peer();
        raw_send(rp.peer, Bytes{0, 0, 0, 1, 0, 0, 0, 0});
        CHECK_CODE(rp.channel->recv(), "CHANNEL_FRAME_TOO_LARGE", "4 GiB announced: refused");
        raw_close(rp.peer);
    }
    {
        // (c) announced 1 GiB (under the cap), 100 bytes sent, then the peer goes away: the receiver
        //     must not commit 1 GiB up front; it fails with CHANNEL_CLOSED after the bytes that arrived
        RawPeer rp = make_raw_peer();
        Bytes msg;
        put_u64be(msg, static_cast<uint64_t>(1) << 30);
        msg.resize(msg.size() + 100, 0x08);
        raw_send(rp.peer, msg);
        raw_close(rp.peer);
        const double t0 = now_s();
        CHECK_CODE(rp.channel->recv(), "CHANNEL_CLOSED", "short frame: peer closed");
        CHECK(now_s() - t0 < 5.0, "no 1 GiB allocation or wait on an announced length");
    }
    {
        // (d) configurable cap: exactly at the cap passes, one byte more is refused
        RawPeer rp = make_raw_peer();
        CHECK_CODE(rp.channel->set_max_frame_bytes(0), "CHANNEL_FRAME_LIMIT", "cap 0 refused");
        CHECK_CODE(rp.channel->set_max_frame_bytes((static_cast<uint64_t>(1) << 40) + 1), "CHANNEL_FRAME_LIMIT", "cap above 2^40 refused");
        CHECK(rp.channel->max_frame_bytes() == Channel::kDefaultMaxFrameBytes, "a refused setter changes nothing");
        const Value small = Value(Bytes(100, 7));            // payload = 1 + 4 + 100 bytes
        const Bytes frame = encode_frame(small);
        CHECK(frame.size() == 8 + 105, "fixture frame size");
        CHECK_OK(rp.channel->set_max_frame_bytes(105), "cap set to the exact payload size");
        raw_send(rp.peer, frame);
        const auto got = rp.channel->recv();
        CHECK_OK(got, "frame exactly at the cap accepted");
        CHECK(got.value.as_bytes().value == Bytes(100, 7), "payload intact");
        CHECK_OK(rp.channel->set_max_frame_bytes(104), "cap lowered by one byte");
        raw_send(rp.peer, frame);
        CHECK_CODE(rp.channel->recv(), "CHANNEL_FRAME_TOO_LARGE", "one byte over the cap refused");
        CHECK(!rp.channel->is_open(), "closed after the refusal");
        raw_close(rp.peer);
    }
    {
        // (e) outgoing frames obey the same cap, and a refused send writes nothing
        auto pair = loopback_pair();
        CHECK_OK(pair, "loopback");
        Channel& a = *pair.value.first;
        Channel& b = *pair.value.second;
        CHECK_OK(a.set_max_frame_bytes(64), "tight cap on the sender");
        CHECK_CODE(a.send(Value(Bytes(1000, 1))), "CHANNEL_FRAME_TOO_LARGE", "outgoing frame over the cap refused");
        CHECK(a.stats.bytes_sent == 0 && a.stats.messages_sent == 0, "nothing was written");
        CHECK_OK(a.send(Value(static_cast<int64_t>(7))), "a small frame still goes out");
        const auto got = b.recv();
        CHECK_OK(got, "the peer reads the next frame in sync");
        CHECK(got.value.as_int().value == 7, "stream in sync after a refused send");
        // a deep frame from a peer is refused by the codec, the stream stays in sync
        CHECK_OK(a.set_max_frame_bytes(Channel::kDefaultMaxFrameBytes), "cap restored to the default");
        Value deep = Value(None{});
        for (int i = 0; i < 70; ++i) deep = Value(List{deep});
        CHECK_OK(b.send(deep), "peer sends a 70-deep value");
        CHECK_CODE(a.recv(), "CODEC_DEPTH", "deep frame refused by the receiver");
        CHECK_OK(b.send(Value("after")), "next frame");
        const auto next = a.recv();
        CHECK(next.ok && next.value.as_string().value == "after", "the frame after a refused deep frame is read correctly");
    }
}

}  // namespace

int main() {
    CHECK_OK(ensure_sockets(), "sockets");
    test_torn_tail();
    std::fprintf(stderr, "[test_ledger] torn tail ok (%ld checks so far)\n", g_checks);
    test_crash_boundaries();
    std::fprintf(stderr, "[test_ledger] crash boundaries ok (%ld checks so far)\n", g_checks);
    test_crash_every_cut();
    test_spend_before_first_message();
    std::fprintf(stderr, "[test_ledger] spend before first message ok (%ld checks so far)\n", g_checks);
    test_reopen_binding_and_duplicates();
    std::fprintf(stderr, "[test_ledger] reopen, binding, transitions, repeated ids ok (%ld checks so far)\n", g_checks);
    test_wmask_version_binding();
    std::fprintf(stderr, "[test_ledger] wmask version binding ok (%ld checks so far)\n", g_checks);
    test_decoder_limits();
    std::fprintf(stderr, "[test_ledger] decoder limits ok (%ld checks so far)\n", g_checks);
    std::printf("PASS %ld checks\n", g_checks);
    return 0;
}
