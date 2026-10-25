// OpenCrab v2: small JSON value type with parse / canonical serialization.
//
// Used by every other module for configs, evidence records, dealer bundle headers and the
// cost-model coefficient files. The canonical serialization (sorted keys, no spaces, ASCII
// only, Python float repr) is byte-identical to Python's
//     json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
// for the value kinds both sides use (null, bool, int64, double, str, list, dict), so a
// SHA-256 over the canonical text (contract_hash, bundle header tags, decision digests)
// agrees across the two languages.
//
// Numbers: a JSON number without fraction or exponent that fits in int64 is an Int, any
// other number is a Double. Doubles print with the shortest round-tripping repr in the
// Python 3 style ("1e-05", "100.0", "1e+16", "0.0001"). NaN and infinities print as the
// Python tokens NaN / Infinity / -Infinity (they never appear in canonical records).
//
// Objects are kept as a vector of (key, value) pairs sorted by byte-wise key order (unique
// keys), so iteration order is the canonical order and the type stays complete for the
// standard containers in C++17.
//
// Failure model: parse / load_file / save_file return Expected<...> with codes JSON_PARSE,
// JSON_DEPTH, JSON_IO. Accessors never throw: a wrong-type accessor returns a neutral value
// (false, 0, 0.0, empty string, empty container) and the caller checks the type first when
// it matters (the typed helpers get_int / get_double / get_string do this and report).
//
// What is tested (tests/test_decision.cpp): round trip of a nested document, canonical key
// order, Python float repr samples, unicode escaping, parse error codes, and the contract
// hash against the Python hashlib value.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/lowlevel/sha256.hpp"

namespace opencrab2 {
namespace highlevel {

class Json {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;   // sorted by key, unique keys

    Json();                                  // null
    Json(std::nullptr_t);                    // null
    Json(bool b);
    Json(int v);
    Json(long v);
    Json(long long v);
    Json(unsigned v);
    Json(unsigned long v);
    Json(unsigned long long v);              // values above INT64_MAX become Double
    Json(double d);
    Json(const char* s);
    Json(std::string s);
    Json(Array a);
    Json(Object o);                          // the pairs are sorted and de-duplicated (last wins)

    static Json object();                    // empty object
    static Json array();                     // empty array

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_int() const { return type_ == Type::Int; }
    bool is_double() const { return type_ == Type::Double; }
    bool is_number() const { return type_ == Type::Int || type_ == Type::Double; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool() const;                    // false unless Bool
    std::int64_t as_int() const;             // Int, or a Double truncated, or 0
    double as_double() const;                // Int or Double, or 0.0
    const std::string& as_string() const;    // empty unless String
    const Array& as_array() const;           // empty unless Array
    Array& as_array();                       // converts a null into an empty array
    const Object& as_object() const;         // empty unless Object
    Object& as_object();                     // converts a null into an empty object

    // Object access.
    bool has(const std::string& key) const;
    const Json* find(const std::string& key) const;     // nullptr when absent or not an object
    const Json& get(const std::string& key) const;      // a shared null when absent
    Json& operator[](const std::string& key);           // inserts null (keeps the order sorted)
    void set(const std::string& key, Json v);
    bool erase(const std::string& key);

    // Typed getters that report a missing or mistyped field (codes JSON_MISSING, JSON_TYPE).
    Expected<std::int64_t> get_int(const std::string& key) const;
    Expected<double> get_double(const std::string& key) const;
    Expected<std::string> get_string(const std::string& key) const;
    Expected<bool> get_bool(const std::string& key) const;

    // Array access. Both int and size_t overloads exist (const and non-const) so that `value[0]` is an
    // exact match and never resolves to the string-key operator through a null `const char*`.
    std::size_t size() const;                           // elements (array), members (object), else 0
    const Json& operator[](std::size_t i) const;        // a shared null when out of range
    const Json& operator[](int i) const { return (*this)[static_cast<std::size_t>(i < 0 ? ~std::size_t(0) : i)]; }
    Json& operator[](std::size_t i);                    // out of range: a scratch null whose writes are discarded
    Json& operator[](int i) { return (*this)[static_cast<std::size_t>(i < 0 ? ~std::size_t(0) : i)]; }
    void push_back(Json v);

    // Serialization.
    std::string dump() const;                           // canonical: sorted keys, no spaces
    std::string dump_pretty(int indent = 2) const;      // sorted keys, indented, for humans
    static Expected<Json> parse(const std::string& text);
    static Expected<Json> load_file(const std::string& path);
    static Status save_file(const std::string& path, const Json& value, bool pretty = true);

    bool operator==(const Json& other) const;
    bool operator!=(const Json& other) const { return !(*this == other); }

private:
    Type type_;
    bool b_;
    std::int64_t i_;
    double d_;
    std::string s_;
    Array a_;
    Object o_;
};

// Python-style shortest repr of a double ("0.1", "1e-05", "100.0", "1e+16").
std::string format_double_repr(double d);

// Lower-case hex SHA-256 of a byte string (delegates to opencrab2::lowlevel::Sha256).
std::string sha256_hex(const std::string& bytes);

// SHA-256 hex of the canonical serialization of a value.
std::string json_sha256_hex(const Json& value);

}  // namespace highlevel
}  // namespace opencrab2
