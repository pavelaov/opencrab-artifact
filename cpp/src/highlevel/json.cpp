// OpenCrab v2: JSON value implementation (see include/opencrab2/highlevel/json.hpp).
#include "opencrab2/highlevel/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace opencrab2 {
namespace highlevel {

namespace {

const Json& shared_null() {
    static const Json n;
    return n;
}

const std::string& shared_empty_string() {
    static const std::string s;
    return s;
}

const Json::Array& shared_empty_array() {
    static const Json::Array a;
    return a;
}

const Json::Object& shared_empty_object() {
    static const Json::Object o;
    return o;
}

// Binary search in a sorted object; returns the insertion point.
Json::Object::const_iterator lower_bound_key(const Json::Object& o, const std::string& key) {
    return std::lower_bound(o.begin(), o.end(), key,
                            [](const std::pair<std::string, Json>& p, const std::string& k) { return p.first < k; });
}

Json::Object::iterator lower_bound_key(Json::Object& o, const std::string& key) {
    return std::lower_bound(o.begin(), o.end(), key,
                            [](const std::pair<std::string, Json>& p, const std::string& k) { return p.first < k; });
}

void append_hex4(std::string& out, unsigned v) {
    static const char* digits = "0123456789abcdef";
    out.push_back('\\');
    out.push_back('u');
    out.push_back(digits[(v >> 12) & 0xF]);
    out.push_back(digits[(v >> 8) & 0xF]);
    out.push_back(digits[(v >> 4) & 0xF]);
    out.push_back(digits[v & 0xF]);
}

// Decode one UTF-8 code point starting at s[i]; returns the code point and advances i.
// Invalid sequences yield U+FFFD and consume one byte.
unsigned decode_utf8(const std::string& s, std::size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        ++i;
        return c;
    }
    int extra = 0;
    unsigned cp = 0;
    if ((c & 0xE0) == 0xC0) { extra = 1; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { extra = 2; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { extra = 3; cp = c & 0x07; }
    else { ++i; return 0xFFFD; }
    for (int k = 1; k <= extra; ++k) {
        if (i + k >= s.size()) { ++i; return 0xFFFD; }
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) { ++i; return 0xFFFD; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += static_cast<std::size_t>(extra) + 1;
    if (cp > 0x10FFFF) return 0xFFFD;
    return cp;
}

void encode_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Python json.dumps(ensure_ascii=True) string escaping.
void dump_string(std::string& out, const std::string& s) {
    out.push_back('"');
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned cp = decode_utf8(s, i);
        switch (cp) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (cp < 0x20) {
                    append_hex4(out, cp);
                } else if (cp < 0x80) {
                    out.push_back(static_cast<char>(cp));
                } else if (cp < 0x10000) {
                    append_hex4(out, cp);
                } else {
                    const unsigned v = cp - 0x10000;
                    append_hex4(out, 0xD800 + (v >> 10));
                    append_hex4(out, 0xDC00 + (v & 0x3FF));
                }
        }
    }
    out.push_back('"');
}

void dump_value(std::string& out, const Json& v, int indent, int depth);

void dump_indent(std::string& out, int indent, int depth) {
    out.push_back('\n');
    out.append(static_cast<std::size_t>(indent) * static_cast<std::size_t>(depth), ' ');
}

void dump_value(std::string& out, const Json& v, int indent, int depth) {
    switch (v.type()) {
        case Json::Type::Null: out += "null"; return;
        case Json::Type::Bool: out += v.as_bool() ? "true" : "false"; return;
        case Json::Type::Int: out += std::to_string(v.as_int()); return;
        case Json::Type::Double: out += format_double_repr(v.as_double()); return;
        case Json::Type::String: dump_string(out, v.as_string()); return;
        case Json::Type::Array: {
            const Json::Array& a = v.as_array();
            if (a.empty()) { out += "[]"; return; }
            out.push_back('[');
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (i) out.push_back(',');
                if (indent > 0) dump_indent(out, indent, depth + 1);
                dump_value(out, a[i], indent, depth + 1);
            }
            if (indent > 0) dump_indent(out, indent, depth);
            out.push_back(']');
            return;
        }
        case Json::Type::Object: {
            const Json::Object& o = v.as_object();
            if (o.empty()) { out += "{}"; return; }
            out.push_back('{');
            for (std::size_t i = 0; i < o.size(); ++i) {
                if (i) out.push_back(',');
                if (indent > 0) dump_indent(out, indent, depth + 1);
                dump_string(out, o[i].first);
                out.push_back(':');
                if (indent > 0) out.push_back(' ');
                dump_value(out, o[i].second, indent, depth + 1);
            }
            if (indent > 0) dump_indent(out, indent, depth);
            out.push_back('}');
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------
class Parser {
public:
    explicit Parser(const std::string& t) : text_(t), pos_(0) {}

    Expected<Json> run() {
        skip_ws();
        Json v;
        std::string err = parse_value(v, 0);
        if (!err.empty()) return Expected<Json>::failure(err);
        skip_ws();
        if (pos_ != text_.size()) return Expected<Json>::failure("JSON_PARSE: trailing characters at " + std::to_string(pos_));
        return Expected<Json>::success(std::move(v));
    }

private:
    static constexpr int kMaxDepth = 512;

    void skip_ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    bool starts_with(const char* lit) const {
        const std::size_t n = std::strlen(lit);
        return text_.compare(pos_, n, lit) == 0;
    }

    std::string fail(const char* what) const {
        return std::string("JSON_PARSE: ") + what + " at " + std::to_string(pos_);
    }

    std::string parse_value(Json& out, int depth) {
        if (depth > kMaxDepth) return "JSON_DEPTH: nesting deeper than 512";
        if (pos_ >= text_.size()) return fail("unexpected end of input");
        const char c = text_[pos_];
        if (c == '{') return parse_object(out, depth);
        if (c == '[') return parse_array(out, depth);
        if (c == '"') {
            std::string s;
            std::string err = parse_string(s);
            if (!err.empty()) return err;
            out = Json(std::move(s));
            return "";
        }
        if (starts_with("true")) { pos_ += 4; out = Json(true); return ""; }
        if (starts_with("false")) { pos_ += 5; out = Json(false); return ""; }
        if (starts_with("null")) { pos_ += 4; out = Json(); return ""; }
        if (starts_with("NaN")) { pos_ += 3; out = Json(std::numeric_limits<double>::quiet_NaN()); return ""; }
        if (starts_with("Infinity")) { pos_ += 8; out = Json(std::numeric_limits<double>::infinity()); return ""; }
        if (starts_with("-Infinity")) { pos_ += 9; out = Json(-std::numeric_limits<double>::infinity()); return ""; }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
        return fail("unexpected character");
    }

    std::string parse_number(Json& out) {
        const std::size_t start = pos_;
        bool is_int = true;
        if (text_[pos_] == '-') ++pos_;
        if (pos_ >= text_.size() || !(text_[pos_] >= '0' && text_[pos_] <= '9')) return fail("bad number");
        if (text_[pos_] == '0') {
            ++pos_;
        } else {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            is_int = false;
            ++pos_;
            if (pos_ >= text_.size() || !(text_[pos_] >= '0' && text_[pos_] <= '9')) return fail("bad fraction");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            is_int = false;
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= text_.size() || !(text_[pos_] >= '0' && text_[pos_] <= '9')) return fail("bad exponent");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        const std::string tok = text_.substr(start, pos_ - start);
        if (is_int) {
            // Accumulate with overflow detection; fall back to double on overflow.
            const bool neg = tok[0] == '-';
            std::uint64_t acc = 0;
            bool overflow = false;
            for (std::size_t i = neg ? 1 : 0; i < tok.size(); ++i) {
                const unsigned d = static_cast<unsigned>(tok[i] - '0');
                if (acc > (std::numeric_limits<std::uint64_t>::max() - d) / 10) { overflow = true; break; }
                acc = acc * 10 + d;
            }
            if (!overflow) {
                if (!neg && acc <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                    out = Json(static_cast<long long>(acc));
                    return "";
                }
                if (neg && acc <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1ULL) {
                    const std::int64_t v = (acc == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1ULL)
                                               ? std::numeric_limits<std::int64_t>::min()
                                               : -static_cast<std::int64_t>(acc);
                    out = Json(static_cast<long long>(v));
                    return "";
                }
            }
        }
        errno = 0;
        const double d = std::strtod(tok.c_str(), nullptr);
        out = Json(d);
        return "";
    }

    static int hexval(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    std::string parse_hex4(unsigned& v) {
        if (pos_ + 4 > text_.size()) return fail("truncated \\u escape");
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const int h = hexval(text_[pos_ + k]);
            if (h < 0) return fail("bad \\u escape");
            v = (v << 4) | static_cast<unsigned>(h);
        }
        pos_ += 4;
        return "";
    }

    std::string parse_string(std::string& out) {
        ++pos_;  // opening quote
        while (true) {
            if (pos_ >= text_.size()) return fail("unterminated string");
            const char c = text_[pos_];
            if (c == '"') { ++pos_; return ""; }
            if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
            if (c != '\\') { out.push_back(c); ++pos_; continue; }
            ++pos_;
            if (pos_ >= text_.size()) return fail("truncated escape");
            const char e = text_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned cp = 0;
                    std::string err = parse_hex4(cp);
                    if (!err.empty()) return err;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // high surrogate: expect a low surrogate escape next
                        if (pos_ + 2 <= text_.size() && text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            unsigned lo = 0;
                            err = parse_hex4(lo);
                            if (!err.empty()) return err;
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                return fail("invalid low surrogate");
                            }
                        } else {
                            return fail("lone high surrogate");
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("lone low surrogate");
                    }
                    encode_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
    }

    std::string parse_array(Json& out, int depth) {
        ++pos_;  // [
        Json::Array items;
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; out = Json(std::move(items)); return ""; }
        while (true) {
            skip_ws();
            Json v;
            std::string err = parse_value(v, depth + 1);
            if (!err.empty()) return err;
            items.push_back(std::move(v));
            skip_ws();
            if (pos_ >= text_.size()) return fail("unterminated array");
            if (text_[pos_] == ',') { ++pos_; continue; }
            if (text_[pos_] == ']') { ++pos_; break; }
            return fail("expected , or ]");
        }
        out = Json(std::move(items));
        return "";
    }

    std::string parse_object(Json& out, int depth) {
        ++pos_;  // {
        Json obj = Json::object();
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; out = std::move(obj); return ""; }
        while (true) {
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != '"') return fail("expected object key");
            std::string key;
            std::string err = parse_string(key);
            if (!err.empty()) return err;
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != ':') return fail("expected :");
            ++pos_;
            skip_ws();
            Json v;
            err = parse_value(v, depth + 1);
            if (!err.empty()) return err;
            obj.set(key, std::move(v));
            skip_ws();
            if (pos_ >= text_.size()) return fail("unterminated object");
            if (text_[pos_] == ',') { ++pos_; continue; }
            if (text_[pos_] == '}') { ++pos_; break; }
            return fail("expected , or }");
        }
        out = std::move(obj);
        return "";
    }

    const std::string& text_;
    std::size_t pos_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Json
// ---------------------------------------------------------------------------
Json::Json() : type_(Type::Null), b_(false), i_(0), d_(0.0) {}
Json::Json(std::nullptr_t) : Json() {}
Json::Json(bool b) : type_(Type::Bool), b_(b), i_(0), d_(0.0) {}
Json::Json(int v) : type_(Type::Int), b_(false), i_(v), d_(0.0) {}
Json::Json(long v) : type_(Type::Int), b_(false), i_(static_cast<std::int64_t>(v)), d_(0.0) {}
Json::Json(long long v) : type_(Type::Int), b_(false), i_(static_cast<std::int64_t>(v)), d_(0.0) {}
Json::Json(unsigned v) : type_(Type::Int), b_(false), i_(static_cast<std::int64_t>(v)), d_(0.0) {}
Json::Json(unsigned long v) : Json(static_cast<unsigned long long>(v)) {}
Json::Json(unsigned long long v) : type_(Type::Int), b_(false), i_(0), d_(0.0) {
    if (v <= static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max())) {
        i_ = static_cast<std::int64_t>(v);
    } else {
        type_ = Type::Double;
        d_ = static_cast<double>(v);
    }
}
Json::Json(double d) : type_(Type::Double), b_(false), i_(0), d_(d) {}
Json::Json(const char* s) : type_(Type::String), b_(false), i_(0), d_(0.0), s_(s ? s : "") {}
Json::Json(std::string s) : type_(Type::String), b_(false), i_(0), d_(0.0), s_(std::move(s)) {}
Json::Json(Array a) : type_(Type::Array), b_(false), i_(0), d_(0.0), a_(std::move(a)) {}
Json::Json(Object o) : type_(Type::Object), b_(false), i_(0), d_(0.0) {
    for (auto& kv : o) set(kv.first, std::move(kv.second));
}

Json Json::object() { Json j; j.type_ = Type::Object; return j; }
Json Json::array() { Json j; j.type_ = Type::Array; return j; }

bool Json::as_bool() const { return type_ == Type::Bool ? b_ : false; }

std::int64_t Json::as_int() const {
    if (type_ == Type::Int) return i_;
    if (type_ == Type::Double && std::isfinite(d_)) {
        if (d_ >= 9.2233720368547758e18) return std::numeric_limits<std::int64_t>::max();
        if (d_ <= -9.2233720368547758e18) return std::numeric_limits<std::int64_t>::min();
        return static_cast<std::int64_t>(d_);
    }
    return 0;
}

double Json::as_double() const {
    if (type_ == Type::Int) return static_cast<double>(i_);
    if (type_ == Type::Double) return d_;
    return 0.0;
}

const std::string& Json::as_string() const { return type_ == Type::String ? s_ : shared_empty_string(); }
const Json::Array& Json::as_array() const { return type_ == Type::Array ? a_ : shared_empty_array(); }
Json::Array& Json::as_array() {
    if (type_ == Type::Null) type_ = Type::Array;
    return a_;
}
const Json::Object& Json::as_object() const { return type_ == Type::Object ? o_ : shared_empty_object(); }
Json::Object& Json::as_object() {
    if (type_ == Type::Null) type_ = Type::Object;
    return o_;
}

bool Json::has(const std::string& key) const { return find(key) != nullptr; }

const Json* Json::find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    auto it = lower_bound_key(o_, key);
    if (it == o_.end() || it->first != key) return nullptr;
    return &it->second;
}

const Json& Json::get(const std::string& key) const {
    const Json* p = find(key);
    return p ? *p : shared_null();
}

Json& Json::operator[](const std::string& key) {
    if (type_ == Type::Null) type_ = Type::Object;
    auto it = lower_bound_key(o_, key);
    if (it == o_.end() || it->first != key) it = o_.insert(it, std::make_pair(key, Json()));
    return it->second;
}

void Json::set(const std::string& key, Json v) {
    if (type_ == Type::Null) type_ = Type::Object;
    auto it = lower_bound_key(o_, key);
    if (it == o_.end() || it->first != key) o_.insert(it, std::make_pair(key, std::move(v)));
    else it->second = std::move(v);
}

bool Json::erase(const std::string& key) {
    if (type_ != Type::Object) return false;
    auto it = lower_bound_key(o_, key);
    if (it == o_.end() || it->first != key) return false;
    o_.erase(it);
    return true;
}

Expected<std::int64_t> Json::get_int(const std::string& key) const {
    const Json* p = find(key);
    if (!p) return Expected<std::int64_t>::failure("JSON_MISSING: " + key);
    if (p->is_int()) return Expected<std::int64_t>::success(p->i_);
    if (p->is_double() && std::isfinite(p->d_) && std::floor(p->d_) == p->d_) return Expected<std::int64_t>::success(p->as_int());
    return Expected<std::int64_t>::failure("JSON_TYPE: " + key + " is not an integer");
}

Expected<double> Json::get_double(const std::string& key) const {
    const Json* p = find(key);
    if (!p) return Expected<double>::failure("JSON_MISSING: " + key);
    if (!p->is_number()) return Expected<double>::failure("JSON_TYPE: " + key + " is not a number");
    return Expected<double>::success(p->as_double());
}

Expected<std::string> Json::get_string(const std::string& key) const {
    const Json* p = find(key);
    if (!p) return Expected<std::string>::failure("JSON_MISSING: " + key);
    if (!p->is_string()) return Expected<std::string>::failure("JSON_TYPE: " + key + " is not a string");
    return Expected<std::string>::success(p->s_);
}

Expected<bool> Json::get_bool(const std::string& key) const {
    const Json* p = find(key);
    if (!p) return Expected<bool>::failure("JSON_MISSING: " + key);
    if (!p->is_bool()) return Expected<bool>::failure("JSON_TYPE: " + key + " is not a bool");
    return Expected<bool>::success(p->b_);
}

std::size_t Json::size() const {
    if (type_ == Type::Array) return a_.size();
    if (type_ == Type::Object) return o_.size();
    return 0;
}

const Json& Json::operator[](std::size_t i) const {
    if (type_ != Type::Array || i >= a_.size()) return shared_null();
    return a_[i];
}

Json& Json::operator[](std::size_t i) {
    if (type_ == Type::Array && i < a_.size()) return a_[i];
    static Json scratch;   // out-of-range writes land here and are discarded (not for concurrent misuse)
    scratch = Json();
    return scratch;
}

void Json::push_back(Json v) {
    if (type_ == Type::Null) type_ = Type::Array;
    if (type_ == Type::Array) a_.push_back(std::move(v));
}

std::string Json::dump() const {
    std::string out;
    dump_value(out, *this, 0, 0);
    return out;
}

std::string Json::dump_pretty(int indent) const {
    std::string out;
    dump_value(out, *this, indent > 0 ? indent : 2, 0);
    return out;
}

Expected<Json> Json::parse(const std::string& text) {
    Parser p(text);
    return p.run();
}

Expected<Json> Json::load_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Expected<Json>::failure("JSON_IO: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse(ss.str());
}

Status Json::save_file(const std::string& path, const Json& value, bool pretty) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return fail_status("JSON_IO: cannot write " + path);
    out << (pretty ? value.dump_pretty(2) : value.dump());
    out << '\n';
    if (!out) return fail_status("JSON_IO: write failed for " + path);
    return ok_status();
}

bool Json::operator==(const Json& other) const {
    if (is_number() && other.is_number()) {
        if (is_int() && other.is_int()) return i_ == other.i_;
        return as_double() == other.as_double();
    }
    if (type_ != other.type_) return false;
    switch (type_) {
        case Type::Null: return true;
        case Type::Bool: return b_ == other.b_;
        case Type::Int: return i_ == other.i_;
        case Type::Double: return d_ == other.d_;
        case Type::String: return s_ == other.s_;
        case Type::Array: return a_ == other.a_;
        case Type::Object: return o_ == other.o_;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Python-style float repr
// ---------------------------------------------------------------------------
std::string format_double_repr(double d) {
    if (std::isnan(d)) return "NaN";
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    if (d == 0.0) return std::signbit(d) ? "-0.0" : "0.0";
    // Shortest digit string that round-trips.
    char buf[64];
    int prec = 1;
    for (; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*e", prec - 1, d);
        if (std::strtod(buf, nullptr) == d) break;
    }
    if (prec > 17) {
        prec = 17;
        std::snprintf(buf, sizeof(buf), "%.*e", 16, d);
    }
    // buf looks like "-d.ddddde+XX"
    std::string s(buf);
    bool neg = false;
    std::size_t p = 0;
    if (s[p] == '-') { neg = true; ++p; }
    std::string digits;
    digits.push_back(s[p++]);
    if (p < s.size() && s[p] == '.') {
        ++p;
        while (p < s.size() && s[p] != 'e' && s[p] != 'E') digits.push_back(s[p++]);
    }
    int exp10 = 0;
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) exp10 = std::atoi(s.c_str() + p + 1);
    // strip trailing zeros of the digit string (keep at least one digit)
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out;
    if (neg) out.push_back('-');
    if (exp10 >= -4 && exp10 < 16) {
        if (exp10 >= 0) {
            const std::size_t int_len = static_cast<std::size_t>(exp10) + 1;
            if (digits.size() <= int_len) {
                out += digits;
                out.append(int_len - digits.size(), '0');
                out += ".0";
            } else {
                out += digits.substr(0, int_len);
                out.push_back('.');
                out += digits.substr(int_len);
            }
        } else {
            out += "0.";
            out.append(static_cast<std::size_t>(-exp10 - 1), '0');
            out += digits;
        }
    } else {
        out.push_back(digits[0]);
        if (digits.size() > 1) {
            out.push_back('.');
            out += digits.substr(1);
        }
        out.push_back('e');
        out.push_back(exp10 < 0 ? '-' : '+');
        const int ae = exp10 < 0 ? -exp10 : exp10;
        if (ae < 10) out.push_back('0');
        out += std::to_string(ae);
    }
    return out;
}

std::string sha256_hex(const std::string& bytes) {
    const lowlevel::Sha256::Digest dg = lowlevel::Sha256::digest(bytes);
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::size_t i = 0; i < dg.size(); ++i) {
        out.push_back(digits[dg[i] >> 4]);
        out.push_back(digits[dg[i] & 0xF]);
    }
    return out;
}

std::string json_sha256_hex(const Json& value) { return sha256_hex(value.dump()); }

}  // namespace highlevel
}  // namespace opencrab2
