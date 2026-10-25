// OpenCrab v2 C++17 mirror: MaterialLedger (paper, Section 4.3, ledger and MaterialCommit).
//
// States: AVAILABLE -> RESERVED -> SPENT; BURNED is a terminal variant of SPENT
// reached on uncertainty (disconnect, timeout, cancellation, restart with
// RESERVED entries). The journal is an append-only JSONL file; every state
// change is written, flushed and fsynced (FlushFileBuffers on Windows, fsync on
// POSIX) before the call returns, so that spend() is durable BEFORE the first
// secret-dependent message that uses the material.
//
// Journal line fields: {material_id, state, session_id, epoch, gate_id, seq, ts}
// plus, for AVAILABLE lines, material_type, nbytes and version so that
// recover() can rebuild the stock; install and send-log entries use the same
// line shape with state "INSTALLED" / "SENT" and the object id or
// "gate_id/seq" in material_id. The FIRST line is the binding record: state
// "BINDING", material_id "LEDGER_BINDING", plus session_id, setup_id,
// model_version, numeric_contract_hash and profile_id as bound by the session.
// A journal with records but without that line is LEDGER_CORRUPT; a journal whose
// binding line differs from the session binding is LEDGER_BINDING_MISMATCH.
//
// Replay accepts only the transitions this code writes (AVAILABLE -> RESERVED,
// RESERVED -> SPENT, AVAILABLE or RESERVED -> BURNED). reserve() refuses a
// repeated id (LEDGER_PARAMS); spend() and burn() count a repeated id once.
// add_available() checks the bundle version against the binding
// (MATERIAL_VERSION_MISMATCH).
//
// Restart (recover): the journal is replayed line by line and RESERVED entries become BURNED.
// Because every record is one write of "line + newline" followed by fsync, a crash can leave
// at most one incomplete record: the final one, without its newline. recover() drops exactly
// that torn tail (durable truncation to the last complete line), journals a NOTE line with the
// id LEDGER_TORN_TAIL (byte count and offset only), reports it in the returned counts
// (torn_tail, torn_tail_bytes, notes) and continues. The dropped record was never acknowledged
// to the caller, so no message depended on it and the affected id keeps its earlier state.
// Everything else fails closed with LEDGER_CORRUPT (unparsable or incomplete line that is NOT
// the unterminated tail, a line that leaves SPENT or BURNED, a second SPENT, a state line
// without an AVAILABLE line) or LEDGER_BAD_STATE; the file is left untouched, the in-memory
// state is not replaced, and open() refuses until a later recover() succeeds.
//
// Invariants (tested in tests/test_engine.cpp and, with crash-boundary injection at reserve,
// spend before send, send before ack and commit, in tests/test_ledger.cpp):
//   1. a material_id enters SPENT at most once; a SPENT or BURNED id is never
//      returned to AVAILABLE and reserve() on it fails with MATERIAL_REUSE;
//   2. reserve(ids) is atomic (all or nothing); spend(ids) requires RESERVED;
//   3. retransmit_allowed(gate_id, seq, digest) is true only for the digest
//      recorded by record_send for that (gate_id, seq): a new input never reuses
//      the mask of a prior seq;
//   4. version binding: bundles whose (session_id, setup_id, model_version,
//      numeric_contract_hash, profile_id) differs are rejected by
//      check_binding(); installed objects are versioned by
//      (model_version, client_key_domain) and installed() returns false on any
//      version mismatch.
// Design limit: a local journal does not resist a maliciously rolled-back host.
#pragma once

#include <opencrab2/highlevel/codec.hpp>

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace opencrab2 {
namespace highlevel {

enum class MaterialState { AVAILABLE, RESERVED, SPENT, BURNED };
const char* material_state_name(MaterialState s);

struct LedgerEntry {
    std::string material_id;
    std::string material_type;
    uint64_t nbytes = 0;
    MaterialState state = MaterialState::AVAILABLE;
    std::string gate_id;
    int64_t epoch = -1;
    Value version;  // dict
};

class MaterialLedger {
public:
    // binding keys: session_id, setup_id, model_version, numeric_contract_hash, profile_id
    MaterialLedger(std::string journal_path, Value session_binding);
    ~MaterialLedger();
    MaterialLedger(const MaterialLedger&) = delete;
    MaterialLedger& operator=(const MaterialLedger&) = delete;

    // Opens the journal. Must be called before any state change. A new (or empty) journal gets
    // its BINDING line: the five binding fields of the session, first line of the file. An
    // existing journal is REPLAYED before anything is appended: its BINDING line must equal the
    // session binding (LEDGER_BINDING_MISMATCH otherwise), a torn tail is dropped, and every
    // SPENT or BURNED id of the file is known again (add_available on it is MATERIAL_REUSE).
    // RESERVED entries found by open() are not burned; they block add_available, reserve, spend,
    // record_send and register_installed with LEDGER_RECOVERY_REQUIRED until recover() ran.
    Expected<Unit> open();
    // True when open() found RESERVED entries and recover() has not burned them yet.
    bool needs_recovery() const { return needs_recovery_; }
    // Read-only audit of a journal file (no binding check, nothing written): counts
    // {available, reserved, spent, burned, sent, installed, lines, torn_tail_bytes}.
    static Expected<Value> audit_file(const std::string& path);

    Expected<Unit> add_available(const std::string& material_id, const std::string& material_type, uint64_t nbytes,
                                 const Value& version);
    Expected<Unit> reserve(const std::vector<std::string>& ids, const std::string& gate_id, int64_t epoch);
    Expected<Unit> spend(const std::vector<std::string>& ids);
    Expected<Unit> burn(const std::vector<std::string>& ids, const std::string& reason);
    Expected<std::string> state(const std::string& material_id) const;
    // (count, bytes) of AVAILABLE entries of a type.
    std::pair<uint64_t, uint64_t> stock(const std::string& material_type) const;
    // Replays the journal, drops a torn trailing line (LEDGER_TORN_TAIL note), turns RESERVED
    // into BURNED (journaled), returns counts {available, reserved_burned, spent, burned,
    // installed, sent, torn_tail (0 or 1), torn_tail_bytes, notes (list of str)}. Fails closed
    // with LEDGER_CORRUPT / LEDGER_BAD_STATE on any other malformed line (see the header).
    Expected<Value> recover();
    // Audit notes of this and earlier recoveries ("LEDGER_TORN_TAIL: ...").
    const std::vector<std::string>& notes() const { return notes_; }
    // True after a failed recover(): open() and every state change are refused.
    bool corrupt() const { return corrupt_; }

    // Retransmission log.
    Expected<Unit> record_send(const std::string& gate_id, int64_t seq, const std::string& payload_digest);
    bool retransmit_allowed(const std::string& gate_id, int64_t seq, const std::string& payload_digest) const;

    // Installed objects, versioned by (model_version, client_key_domain, ...).
    Expected<Unit> register_installed(const std::string& object_id, const Value& version);
    bool installed(const std::string& object_id, const Value& version) const;
    std::vector<std::string> installed_objects() const;

    // Version binding of a bundle (the session tuple of its header).
    Expected<Unit> check_binding(const Value& header) const;
    const Value& binding() const { return binding_; }
    bool seen(const std::string& material_id) const { return entries_.count(material_id) > 0; }

    // Counters for the evidence record.
    uint64_t bytes_consumed() const { return bytes_spent_; }
    uint64_t bytes_burned() const { return bytes_burned_; }
    uint64_t bytes_added() const { return bytes_added_; }
    const std::string& journal_path() const { return path_; }

private:
    Expected<Unit> append_line(const Value& line);
    Expected<Unit> write_binding_line();
    Expected<Unit> guard() const;
    Expected<Value> replay(bool burn_reserved);
    static std::vector<std::string> unique_ids(const std::vector<std::string>& ids);
    Expected<Unit> journal_state(const std::string& material_id, MaterialState st, const std::string& gate_id,
                                 int64_t epoch, int64_t seq, const Value* extra);
    Value base_line(const std::string& material_id, const std::string& state, const std::string& gate_id,
                    int64_t epoch, int64_t seq) const;

    std::string path_;
    Value binding_;
    std::FILE* fp_ = nullptr;
    std::map<std::string, LedgerEntry> entries_;
    std::map<std::string, std::string> send_log_;  // "gate_id/seq" -> digest
    std::map<std::string, Value> installed_;
    std::vector<std::string> notes_;
    bool corrupt_ = false;
    bool needs_recovery_ = false;
    uint64_t bytes_spent_ = 0;
    uint64_t bytes_burned_ = 0;
    uint64_t bytes_added_ = 0;
    int64_t seq_ = 0;
};

// Canonical comparison of two version dicts (sorted canonical JSON equality).
bool version_equal(const Value& a, const Value& b);

}  // namespace highlevel
}  // namespace opencrab2
