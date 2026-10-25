// Implementation of ledger.hpp (MaterialLedger with a durable JSONL journal).
#include <opencrab2/highlevel/ledger.hpp>

#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace opencrab2 {
namespace highlevel {

const char* material_state_name(MaterialState s) {
    switch (s) {
        case MaterialState::AVAILABLE: return "AVAILABLE";
        case MaterialState::RESERVED: return "RESERVED";
        case MaterialState::SPENT: return "SPENT";
        case MaterialState::BURNED: return "BURNED";
    }
    return "UNKNOWN";
}

bool version_equal(const Value& a, const Value& b) {
    auto ca = canonical_json(a);
    auto cb = canonical_json(b);
    if (!ca.ok || !cb.ok) return false;
    return ca.value == cb.value;
}

MaterialLedger::MaterialLedger(std::string journal_path, Value session_binding)
    : path_(std::move(journal_path)), binding_(std::move(session_binding)) {}

MaterialLedger::~MaterialLedger() {
    if (fp_) std::fclose(fp_);
}

Expected<Unit> MaterialLedger::open() {
    if (fp_) return ok_unit();
    // A journal that failed recovery must not be appended to: the in-memory state would be empty
    // while the file still holds SPENT ids (fail closed until a recover() succeeds).
    if (corrupt_) return fail("LEDGER_CORRUPT: journal failed recovery, refusing to open " + path_);
    // An existing journal is replayed BEFORE anything is appended to it: the in-memory state must
    // hold every SPENT and BURNED id of the file (invariant 1), the binding line must match this
    // session, and a torn tail must be gone. RESERVED entries are not burned here: they block
    // every state change with LEDGER_RECOVERY_REQUIRED until recover() ran (the restart rule).
    bool has_content = false;
    {
        std::ifstream in(path_, std::ios::binary | std::ios::ate);
        if (in) has_content = in.tellg() > 0;
    }
    if (has_content) {
        OC2_TRY(counts, replay(false));
        (void)counts;
        return ok_unit();
    }
    fp_ = std::fopen(path_.c_str(), "ab");
    if (!fp_) return fail("LEDGER_OPEN: cannot open journal " + path_);
    return write_binding_line();
}

namespace {
const char* const kBindingKeys[] = {"session_id", "setup_id", "model_version", "numeric_contract_hash", "profile_id"};
const char* const kVersionKeys[] = {"setup_id", "model_version", "numeric_contract_hash", "profile_id"};
}  // namespace

// First line of every journal: the five binding fields of the session.
Expected<Unit> MaterialLedger::write_binding_line() {
    Value line = base_line("LEDGER_BINDING", "BINDING", "", -1, seq_++);
    for (const char* k : kBindingKeys) {
        if (const Value* v = binding_.get(k)) line.set(k, *v);
    }
    return append_line(line);
}

Expected<Unit> MaterialLedger::guard() const {
    if (!fp_) return fail("LEDGER_NOT_OPEN");
    if (needs_recovery_)
        return fail("LEDGER_RECOVERY_REQUIRED: journal holds RESERVED entries; call recover() first");
    return ok_unit();
}

std::vector<std::string> MaterialLedger::unique_ids(const std::vector<std::string>& ids) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& id : ids)
        if (seen.insert(id).second) out.push_back(id);
    return out;
}

namespace {

// Durable truncation of the journal to `size` bytes (drops a torn trailing line).
Expected<Unit> truncate_file(const std::string& path, uint64_t size) {
    std::FILE* f = std::fopen(path.c_str(), "rb+");
    if (!f) return fail("LEDGER_TRUNCATE: cannot reopen journal " + path);
    bool good = true;
#ifdef _WIN32
    if (_chsize_s(_fileno(f), static_cast<long long>(size)) != 0) good = false;
    if (good && _commit(_fileno(f)) != 0) good = false;
#else
    if (ftruncate(fileno(f), static_cast<off_t>(size)) != 0) good = false;
    if (good && fsync(fileno(f)) != 0) good = false;
#endif
    if (std::fclose(f) != 0) good = false;
    if (!good) return fail("LEDGER_TRUNCATE: cannot truncate journal " + path);
    return ok_unit();
}

}  // namespace

Expected<Unit> MaterialLedger::append_line(const Value& line) {
    if (!fp_) return fail("LEDGER_NOT_OPEN");
    OC2_TRY(text, to_json(line));
    text.push_back('\n');
    if (std::fwrite(text.data(), 1, text.size(), fp_) != text.size()) return fail("LEDGER_WRITE");
    if (std::fflush(fp_) != 0) return fail("LEDGER_FLUSH");
#ifdef _WIN32
    if (_commit(_fileno(fp_)) != 0) return fail("LEDGER_FSYNC");
#else
    if (fsync(fileno(fp_)) != 0) return fail("LEDGER_FSYNC");
#endif
    return ok_unit();
}

Value MaterialLedger::base_line(const std::string& material_id, const std::string& state, const std::string& gate_id,
                                int64_t epoch, int64_t seq) const {
    std::string session_id;
    if (const Value* s = binding_.get("session_id")) {
        auto r = s->as_string();
        if (r.ok) session_id = r.value;
    }
    return dict({{"material_id", material_id},
                 {"state", state},
                 {"session_id", session_id},
                 {"epoch", epoch},
                 {"gate_id", gate_id},
                 {"seq", seq},
                 {"ts", unix_time_s()}});
}

Expected<Unit> MaterialLedger::journal_state(const std::string& material_id, MaterialState st,
                                             const std::string& gate_id, int64_t epoch, int64_t seq,
                                             const Value* extra) {
    Value line = base_line(material_id, material_state_name(st), gate_id, epoch, seq);
    if (extra && extra->is_dict()) {
        for (const auto& e : *std::get_if<Dict>(&extra->v)) line.set(e.key, e.val);
    }
    return append_line(line);
}

Expected<Unit> MaterialLedger::add_available(const std::string& material_id, const std::string& material_type,
                                             uint64_t nbytes, const Value& version) {
    OC2_CHECK(guard());
    if (material_id.empty()) return fail("LEDGER_PARAMS: material_id must be non-empty");
    if (entries_.count(material_id)) return fail("MATERIAL_REUSE: material_id seen before: " + material_id);
    // version binding: a bundle of another setup, model version, contract or
    // profile never enters the stock of this session
    if (!version.is_dict()) return fail("MATERIAL_VERSION_MISMATCH: version must be a dict");
    OC2_CHECK(check_binding(version));
    LedgerEntry e;
    e.material_id = material_id;
    e.material_type = material_type;
    e.nbytes = nbytes;
    e.state = MaterialState::AVAILABLE;
    e.version = version;
    Value extra = dict({{"material_type", material_type}, {"nbytes", static_cast<int64_t>(nbytes)}, {"version", version}});
    OC2_CHECK(journal_state(material_id, MaterialState::AVAILABLE, "", -1, seq_++, &extra));
    entries_[material_id] = std::move(e);
    bytes_added_ += nbytes;
    return ok_unit();
}

Expected<Unit> MaterialLedger::reserve(const std::vector<std::string>& ids, const std::string& gate_id,
                                       int64_t epoch) {
    OC2_CHECK(guard());
    if (unique_ids(ids).size() != ids.size()) return fail("LEDGER_PARAMS: duplicate material_id in reserve request");
    // atomic: validate everything first
    for (const auto& id : ids) {
        auto it = entries_.find(id);
        if (it == entries_.end()) return fail("LEDGER_UNKNOWN_ID: " + id);
        if (it->second.state == MaterialState::SPENT || it->second.state == MaterialState::BURNED)
            return fail("MATERIAL_REUSE: " + id + " is " + material_state_name(it->second.state));
        if (it->second.state == MaterialState::RESERVED) return fail("LEDGER_ALREADY_RESERVED: " + id);
    }
    for (const auto& id : ids) {
        OC2_CHECK(journal_state(id, MaterialState::RESERVED, gate_id, epoch, seq_++, nullptr));
        auto& e = entries_[id];
        e.state = MaterialState::RESERVED;
        e.gate_id = gate_id;
        e.epoch = epoch;
    }
    return ok_unit();
}

Expected<Unit> MaterialLedger::spend(const std::vector<std::string>& ids_in) {
    OC2_CHECK(guard());
    // a repeated id counts once: a second SPENT line would break invariant 1 and the replay
    const std::vector<std::string> ids = unique_ids(ids_in);
    for (const auto& id : ids) {
        auto it = entries_.find(id);
        if (it == entries_.end()) return fail("LEDGER_UNKNOWN_ID: " + id);
        if (it->second.state != MaterialState::RESERVED)
            return fail(std::string("LEDGER_NOT_RESERVED: ") + id + " is " + material_state_name(it->second.state));
    }
    for (const auto& id : ids) {
        auto& e = entries_[id];
        OC2_CHECK(journal_state(id, MaterialState::SPENT, e.gate_id, e.epoch, seq_++, nullptr));
        e.state = MaterialState::SPENT;
        bytes_spent_ += e.nbytes;
    }
    return ok_unit();
}

Expected<Unit> MaterialLedger::burn(const std::vector<std::string>& ids_in, const std::string& reason) {
    if (!fp_) return fail("LEDGER_NOT_OPEN");
    const std::vector<std::string> ids = unique_ids(ids_in);
    for (const auto& id : ids) {
        auto it = entries_.find(id);
        if (it == entries_.end()) return fail("LEDGER_UNKNOWN_ID: " + id);
        if (it->second.state == MaterialState::SPENT) return fail("LEDGER_ALREADY_SPENT: " + id);
        if (it->second.state == MaterialState::BURNED) continue;
        Value extra = dict({{"reason", reason}});
        OC2_CHECK(journal_state(id, MaterialState::BURNED, it->second.gate_id, it->second.epoch, seq_++, &extra));
        it->second.state = MaterialState::BURNED;
        bytes_burned_ += it->second.nbytes;
    }
    return ok_unit();
}

Expected<std::string> MaterialLedger::state(const std::string& material_id) const {
    auto it = entries_.find(material_id);
    if (it == entries_.end()) return fail("LEDGER_UNKNOWN_ID: " + material_id);
    return ok(std::string(material_state_name(it->second.state)));
}

std::pair<uint64_t, uint64_t> MaterialLedger::stock(const std::string& material_type) const {
    uint64_t count = 0, bytes = 0;
    for (const auto& kv : entries_) {
        if (kv.second.state == MaterialState::AVAILABLE && kv.second.material_type == material_type) {
            count += 1;
            bytes += kv.second.nbytes;
        }
    }
    return {count, bytes};
}

Expected<Value> MaterialLedger::recover() { return replay(true); }

Expected<Value> MaterialLedger::replay(bool burn_reserved) {
    // Replay the journal from disk (independent of the in-memory map). Nothing on disk and
    // nothing in memory changes until every complete line has been validated.
    auto refuse = [this](const std::string& code) -> Failure {
        // Fail closed: a ledger whose journal cannot be replayed accepts no state change.
        if (fp_) {
            std::fclose(fp_);
            fp_ = nullptr;
        }
        corrupt_ = true;
        return fail(code);
    };
    std::string data;
    {
        std::ifstream in(path_, std::ios::binary);
        if (!in) return fail("LEDGER_OPEN: cannot read journal " + path_);
        std::ostringstream ss;
        ss << in.rdbuf();
        data = ss.str();
    }
    // append_line writes one line plus '\n' and fsyncs before the state change is visible to the
    // caller. A crash can therefore leave at most ONE incomplete record, at the very end and
    // without its newline. Such a tail was never acknowledged to the caller (no message that
    // depends on it was sent), so it is dropped: the affected id stays in its earlier state,
    // which is the safe direction (AVAILABLE or RESERVED, the latter burned below). Any other
    // malformed line is corruption and recovery refuses.
    const std::string::size_type last_nl = data.rfind('\n');
    const size_t complete_len = (last_nl == std::string::npos) ? 0 : last_nl + 1;
    const size_t torn_bytes = data.size() - complete_len;

    std::map<std::string, LedgerEntry> entries;
    std::map<std::string, std::string> send_log;
    std::map<std::string, Value> installed;
    std::vector<std::string> notes;
    int64_t max_seq = -1;
    size_t pos = 0;
    int64_t line_no = 0;
    int64_t records = 0;
    bool binding_seen = false;
    while (pos < complete_len) {
        const size_t nl = data.find('\n', pos);   // exists: pos < complete_len
        const std::string line = data.substr(pos, nl - pos);
        pos = nl + 1;
        ++line_no;
        if (line.empty()) continue;
        const std::string where = "LEDGER_CORRUPT: line " + std::to_string(line_no) + ": ";
        auto parsed = parse_json(line);
        if (!parsed.ok) return refuse(where + parsed.code);
        const Value& v = parsed.value;
        auto mid_r = v.get_string("material_id");
        if (!mid_r.ok) return refuse(where + mid_r.code);
        auto st_r = v.get_string("state");
        if (!st_r.ok) return refuse(where + st_r.code);
        const std::string& mid = mid_r.value;
        const std::string& st = st_r.value;
        if (const Value* sq = v.get("seq")) {
            auto s = sq->as_int();
            if (s.ok && s.value > max_seq) max_seq = s.value;
        }
        ++records;
        if (st == "BINDING") {
            if (binding_seen) return refuse(where + "second BINDING record");
            binding_seen = true;
            for (const char* k : kBindingKeys) {
                const Value* mine = binding_.get(k);
                const Value* theirs = v.get(k);
                const bool same = (!mine && !theirs) || (mine && theirs && version_equal(*mine, *theirs));
                if (!same) {
                    // not corruption: this journal belongs to another session, setup or model version
                    if (fp_) {
                        std::fclose(fp_);
                        fp_ = nullptr;
                    }
                    return fail(std::string("LEDGER_BINDING_MISMATCH: journal ") + k + " differs from the session binding");
                }
            }
            continue;
        }
        if (st == "NOTE") {
            // audit notes of earlier recoveries (LEDGER_TORN_TAIL); no state
            auto d = v.get_string("detail");
            notes.push_back(mid + (d.ok ? ": " + d.value : std::string()));
            continue;
        }
        if (st == "SENT") {
            auto dg = v.get_string("digest");
            if (!dg.ok) return refuse(where + "SENT line without digest");
            send_log[mid] = dg.value;
            continue;
        }
        if (st == "INSTALLED") {
            const Value* ver = v.get("version");
            if (!ver) return refuse(where + "INSTALLED line without version");
            installed[mid] = *ver;
            continue;
        }
        MaterialState next;
        if (st == "AVAILABLE") next = MaterialState::AVAILABLE;
        else if (st == "RESERVED") next = MaterialState::RESERVED;
        else if (st == "SPENT") next = MaterialState::SPENT;
        else if (st == "BURNED") next = MaterialState::BURNED;
        else return refuse("LEDGER_BAD_STATE: line " + std::to_string(line_no) + ": " + st);
        auto known = entries.find(mid);
        if (known != entries.end()) {
            // Only the transitions this code writes are accepted: AVAILABLE -> RESERVED,
            // RESERVED -> SPENT, AVAILABLE or RESERVED -> BURNED. SPENT and BURNED are terminal;
            // anything else (a second SPENT, SPENT without RESERVED, RESERVED back to AVAILABLE,
            // a duplicate AVAILABLE) contradicts invariant 1 and cannot come from this code.
            const MaterialState prev = known->second.state;
            const bool terminal = (prev == MaterialState::SPENT || prev == MaterialState::BURNED);
            if (prev == MaterialState::SPENT && next == MaterialState::SPENT) return refuse(where + mid + " is SPENT twice");
            if (terminal) return refuse(where + mid + " leaves terminal state " + material_state_name(prev));
            bool allowed = false;
            if (next == MaterialState::RESERVED) allowed = (prev == MaterialState::AVAILABLE);
            else if (next == MaterialState::SPENT) allowed = (prev == MaterialState::RESERVED);
            else if (next == MaterialState::BURNED) allowed = true;   // from AVAILABLE or RESERVED
            if (!allowed)
                return refuse(where + mid + " goes from " + material_state_name(prev) + " to " + material_state_name(next));
        } else if (next != MaterialState::AVAILABLE) {
            return refuse(where + mid + " has no AVAILABLE line");
        }
        LedgerEntry& e = entries[mid];
        e.material_id = mid;
        if (const Value* mt = v.get("material_type")) {
            auto s = mt->as_string();
            if (s.ok) e.material_type = s.value;
        }
        if (const Value* nb = v.get("nbytes")) {
            auto n = nb->as_int();
            if (n.ok) e.nbytes = static_cast<uint64_t>(n.value);
        }
        if (const Value* ver = v.get("version")) e.version = *ver;
        auto g = v.get_string("gate_id");
        if (g.ok) e.gate_id = g.value;
        auto ep = v.get_int("epoch");
        if (ep.ok) e.epoch = ep.value;
        e.state = next;
    }
    if (records > 0 && !binding_seen) return refuse("LEDGER_CORRUPT: journal has no BINDING record");
    // Every complete line is valid. Drop the torn tail durably before the journal is appended to.
    if (torn_bytes > 0) {
        if (fp_) {
            std::fclose(fp_);
            fp_ = nullptr;
        }
        auto tr = truncate_file(path_, complete_len);
        if (!tr.ok) return refuse(tr.code);
    }
    corrupt_ = false;
    entries_ = std::move(entries);
    send_log_ = std::move(send_log);
    installed_ = std::move(installed);
    notes_ = std::move(notes);
    seq_ = max_seq + 1;
    if (!fp_) {
        fp_ = std::fopen(path_.c_str(), "ab");
        if (!fp_) return fail("LEDGER_OPEN: cannot open journal " + path_);
    }
    if (!binding_seen) OC2_CHECK(write_binding_line());   // a journal that held nothing but a fragment
    if (torn_bytes > 0) {
        // Sizes and offsets only: the fragment itself is not copied anywhere.
        const std::string detail = "truncated " + std::to_string(torn_bytes) + " bytes of an incomplete trailing line at offset " +
                                   std::to_string(complete_len);
        Value note = base_line("LEDGER_TORN_TAIL", "NOTE", "", -1, seq_++);
        note.set("detail", detail);
        note.set("dropped_bytes", static_cast<int64_t>(torn_bytes));
        note.set("offset", static_cast<int64_t>(complete_len));
        OC2_CHECK(append_line(note));
        notes_.push_back("LEDGER_TORN_TAIL: " + detail);
    }
    int64_t n_avail = 0, n_burned_now = 0, n_spent = 0, n_burned = 0;
    bytes_spent_ = 0;
    bytes_burned_ = 0;
    bytes_added_ = 0;
    int64_t n_reserved = 0;
    needs_recovery_ = false;
    for (auto& kv : entries_) {
        bytes_added_ += kv.second.nbytes;
        if (kv.second.state == MaterialState::RESERVED && !burn_reserved) {
            n_reserved++;
            needs_recovery_ = true;
        } else if (kv.second.state == MaterialState::RESERVED) {
            Value extra = dict({{"reason", "recover: RESERVED at restart"}});
            OC2_CHECK(journal_state(kv.first, MaterialState::BURNED, kv.second.gate_id, kv.second.epoch, seq_++, &extra));
            kv.second.state = MaterialState::BURNED;
            n_burned_now++;
        }
        switch (kv.second.state) {
            case MaterialState::AVAILABLE: n_avail++; break;
            case MaterialState::SPENT: n_spent++; bytes_spent_ += kv.second.nbytes; break;
            case MaterialState::BURNED: n_burned++; bytes_burned_ += kv.second.nbytes; break;
            default: break;
        }
    }
    List note_list;
    for (const auto& n : notes_) note_list.push_back(Value(n));
    return ok(dict({{"available", n_avail},
                    {"reserved_burned", n_burned_now},
                    {"reserved", n_reserved},
                    {"spent", n_spent},
                    {"burned", n_burned},
                    {"installed", static_cast<int64_t>(installed_.size())},
                    {"sent", static_cast<int64_t>(send_log_.size())},
                    {"torn_tail", static_cast<int64_t>(torn_bytes > 0 ? 1 : 0)},
                    {"torn_tail_bytes", static_cast<int64_t>(torn_bytes)},
                    {"notes", Value(note_list)}}));
}

Expected<Unit> MaterialLedger::record_send(const std::string& gate_id, int64_t seq, const std::string& payload_digest) {
    OC2_CHECK(guard());
    std::string key = gate_id + "/" + std::to_string(seq);
    auto it = send_log_.find(key);
    if (it != send_log_.end()) {
        if (it->second != payload_digest) return fail("LEDGER_SEQ_REUSE: different payload for " + key);
        return ok_unit();  // identical retransmission
    }
    Value line = base_line(key, "SENT", gate_id, -1, seq);
    line.set("digest", payload_digest);
    OC2_CHECK(append_line(line));
    send_log_[key] = payload_digest;
    return ok_unit();
}

bool MaterialLedger::retransmit_allowed(const std::string& gate_id, int64_t seq, const std::string& payload_digest) const {
    auto it = send_log_.find(gate_id + "/" + std::to_string(seq));
    return it != send_log_.end() && it->second == payload_digest;
}

Expected<Unit> MaterialLedger::register_installed(const std::string& object_id, const Value& version) {
    OC2_CHECK(guard());
    Value line = base_line(object_id, "INSTALLED", "", -1, seq_++);
    line.set("version", version);
    OC2_CHECK(append_line(line));
    installed_[object_id] = version;
    return ok_unit();
}

bool MaterialLedger::installed(const std::string& object_id, const Value& version) const {
    auto it = installed_.find(object_id);
    if (it == installed_.end()) return false;
    return version_equal(it->second, version);
}

std::vector<std::string> MaterialLedger::installed_objects() const {
    std::vector<std::string> out;
    for (const auto& kv : installed_) out.push_back(kv.first);
    return out;
}

// Read-only audit of a journal file: no binding check, no truncation, nothing written.
Expected<Value> MaterialLedger::audit_file(const std::string& path) {
    std::string data;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) return fail("LEDGER_OPEN: cannot read journal " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        data = ss.str();
    }
    const std::string::size_type last_nl = data.rfind('\n');
    const size_t complete_len = (last_nl == std::string::npos) ? 0 : last_nl + 1;
    std::map<std::string, std::string> state;
    int64_t lines = 0, sent = 0, installed = 0;
    size_t pos = 0;
    while (pos < complete_len) {
        const size_t nl = data.find('\n', pos);
        const std::string line = data.substr(pos, nl - pos);
        pos = nl + 1;
        if (line.empty()) continue;
        ++lines;
        auto parsed = parse_json(line);
        if (!parsed.ok) return fail("LEDGER_CORRUPT: line " + std::to_string(lines) + ": " + parsed.code);
        auto mid = parsed.value.get_string("material_id");
        auto st = parsed.value.get_string("state");
        if (!mid.ok || !st.ok) return fail("LEDGER_CORRUPT: line " + std::to_string(lines) + ": missing fields");
        if (st.value == "BINDING" || st.value == "NOTE") continue;
        if (st.value == "SENT") { ++sent; continue; }
        if (st.value == "INSTALLED") { ++installed; continue; }
        state[mid.value] = st.value;
    }
    int64_t n_avail = 0, n_res = 0, n_spent = 0, n_burned = 0;
    for (const auto& kv : state) {
        if (kv.second == "AVAILABLE") n_avail++;
        else if (kv.second == "RESERVED") n_res++;
        else if (kv.second == "SPENT") n_spent++;
        else if (kv.second == "BURNED") n_burned++;
    }
    return ok(dict({{"available", n_avail}, {"reserved", n_res}, {"spent", n_spent}, {"burned", n_burned},
                    {"sent", sent}, {"installed", installed}, {"lines", lines},
                    {"torn_tail_bytes", static_cast<int64_t>(data.size() - complete_len)}}));
}

Expected<Unit> MaterialLedger::check_binding(const Value& header) const {
    for (const char* k : kVersionKeys) {
        const Value* mine = binding_.get(k);
        const Value* theirs = header.get(k);
        if (!mine) continue;  // unbound field in the session binding
        if (!theirs) return fail(std::string("MATERIAL_VERSION_MISMATCH: header lacks ") + k);
        if (!version_equal(*mine, *theirs))
            return fail(std::string("MATERIAL_VERSION_MISMATCH: ") + k + " differs");
    }
    return ok_unit();
}

}  // namespace highlevel
}  // namespace opencrab2
