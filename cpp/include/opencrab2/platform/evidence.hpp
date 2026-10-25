// opencrab2/platform/evidence.hpp
//
// The evidence record schema "opencrab-native-evidence-v2" of
// python/opencrab2/experiments/schema.py, as the C++
// apps write it. Every line an app appends to an evidence file goes through
// validate() first, so a C++ record is accepted by schema.validate and by
// make_tables.py exactly like a Python record.
//
// Vocabulary (kept in lock step with schema.py; tests/smoke_three_process.py
// checks the two lists and validates C++ lines with the Python validator):
//   record_fields()    RECORD_FIELDS, the required flat fields
//   optional_fields()  OPTIONAL_FIELDS; anything else must live under "extra"
//   statuses, workload scopes, dealer modes, clock bases, record kinds
//
// validate() mirrors schema.validate and fails closed with the same codes:
// SCHEMA_MISSING_FIELD, SCHEMA_UNKNOWN_FIELD, SCHEMA_BAD_VALUE,
// SCHEMA_BAD_STATUS, SCHEMA_BAD_SCOPE, SCHEMA_BAD_TYPE, SCHEMA_ELIGIBILITY.
// Eligibility rule (paper, Section 3, evidence classes): security_eligible may be true only with
// dealer_mode ISOLATED_RECIPIENT_BUNDLES, diagnostic_reveals == 0, status
// MEASURED_NATIVE and aborts == 0.
//
// conform() turns a record of the highlevel session loop (flat, with analysis
// fields next to the schema fields and the token list in "tokens") into the
// schema shape: schema fields stay flat, tokens becomes the count, every other
// field moves under extra, extra.language is "cpp". enforce() then applies the
// rules that never depend on the caller: a shared-seed dealer makes the record
// DIAGNOSTIC and not eligible, and an eligibility claim
// that the record's own fields do not support is withdrawn.
//
// append() validates, then appends one JSON line (sorted keys) under the
// cross-process lock file "<path>.lock" of schema.locked_append_lines
// (O_CREAT | O_EXCL, stale after 120 s, RECORD_LOCK_TIMEOUT after 30 s), so
// the two party processes of one session and a Python runner can append to
// the same file.
//
// Code identity: build.py defines OPENCRAB2_CODE_COMMIT for the translation
// unit of every app as cpp-tree-<sha256 over the sorted "<file sha256>  <path>"
// lines of include/, src/ and apps/>; OPENCRAB2_CODE_COMMIT_LITERAL is that
// value as a string literal ("cpp-tree-unknown" in a build without it). It is
// a macro on purpose: only app sources use it, so no library object can carry
// a stale value.
//
// Tested (tests/test_platform.cpp): defaults validate, every rejection code,
// conform of a session-shaped record, enforce of a shared-seed record, lock
// file append from several threads, the field lists against the Python lists
// (smoke script).
#pragma once

#include <opencrab2/platform/support.hpp>

#include <string>
#include <vector>

#ifndef OPENCRAB2_CODE_COMMIT
#define OPENCRAB2_CODE_COMMIT cpp-tree-unknown
#endif
#define OPENCRAB2_EVIDENCE_STR2(x) #x
#define OPENCRAB2_EVIDENCE_STR(x) OPENCRAB2_EVIDENCE_STR2(x)
#define OPENCRAB2_CODE_COMMIT_LITERAL OPENCRAB2_EVIDENCE_STR(OPENCRAB2_CODE_COMMIT)

namespace opencrab2 {
namespace platform {
namespace evidence {

extern const char* const kSchemaId;   // "opencrab-native-evidence-v2"
extern const char* const kProfileId;  // "SH2P-TrustedOffline-v1"

const std::vector<std::string>& record_fields();
const std::vector<std::string>& optional_fields();
const std::vector<std::string>& statuses();
const std::vector<std::string>& workload_scopes();
const std::vector<std::string>& dealer_modes();
const std::vector<std::string>& clock_bases();
const std::vector<std::string>& record_kinds();

// A complete record with the safe defaults of schema.new_record: status
// MODELED, security_eligible false, dealer_mode NONE, clock_basis
// modeled-epoch, zero counters, null timings, extra {language: "cpp"}.
highlevel::Value new_record(const std::string& code_commit);

// schema.validate. The failure code is "<SCHEMA_CODE>: <detail>".
Expected<bool> validate(const highlevel::Value& record);

// See the header comment. record_kind is one of record_kinds().
highlevel::Value conform(const highlevel::Value& record, const std::string& record_kind, const std::string& code_commit);

// Applies the caller-independent rules in place; returns the reasons for
// every change (empty when nothing changed). Each reason also goes to
// extra.eligibility_notes.
std::vector<std::string> enforce(highlevel::Value& record);

// Withdraws an eligibility claim for a stated reason (status becomes
// DIAGNOSTIC when it was MEASURED_NATIVE and `diagnostic` is true).
void withdraw_eligibility(highlevel::Value& record, const std::string& reason, bool diagnostic);

// Sets extra[key] = value (creates extra when absent).
void set_extra(highlevel::Value& record, const std::string& key, highlevel::Value value);

// validate + locked append of one JSON line. Codes: SCHEMA_*, RECORD_IO,
// RECORD_LOCK_TIMEOUT.
Expected<bool> append(const std::string& path, const highlevel::Value& record);

// The value a record stores in raw_measurement_path for the evidence file
// `path`: relative to the data root, with forward slashes, so that a record
// does not name the host or the place of the tree (the rule of the Python
// schema.relative_raw_path). A relative path is kept as given. A rooted path
// is made relative to $OPENCRAB2_RESULTS_DIR when it lies inside that
// directory; otherwise it is cut at its deepest directory named raw,
// calibration, processed or tables (<tree>/results/raw/cpp_bench.jsonl and
// <tree>/runs/full/raw/cpp_bench.jsonl are both stored as
// "raw/cpp_bench.jsonl"); a path without such a directory keeps its file name
// only. "" stays "".
std::string record_path(const std::string& path);

// "2026-10-10T19:05:45Z"
std::string utc_now();
// {os, hardware_threads, compiler, pointer_bits}: public host facts for extra.host.
highlevel::Value host_info();

}  // namespace evidence
}  // namespace platform
}  // namespace opencrab2
