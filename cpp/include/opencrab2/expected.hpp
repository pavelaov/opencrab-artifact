// OpenCrab v2: fail-closed result type shared by every C++ layer.
//
// Every function that can fail returns Expected<T>: either ok == true with a value, or
// ok == false with a stable upper-case code such as "CONTRACT_VIOLATION" or "DCF_WIRE".
// The struct is an aggregate on purpose so that callers may also write
// Expected<int>{true, 5, ""}; the static helpers below are the preferred spelling.
#pragma once

#include <string>
#include <utility>

namespace opencrab2 {

// Marker type for functions that return nothing but may still fail.
struct Unit {};

template <class T>
struct Expected {
    bool ok = false;
    T value{};
    std::string code;   // empty when ok; otherwise a stable upper-case code, optionally ": detail"

    static Expected success(T v) {
        Expected e;
        e.ok = true;
        e.value = std::move(v);
        return e;
    }
    static Expected failure(std::string c) {
        Expected e;
        e.ok = false;
        e.code = std::move(c);
        return e;
    }
    explicit operator bool() const noexcept { return ok; }
    // The bare code without the ": detail" suffix.
    std::string code_prefix() const {
        const std::string::size_type p = code.find(':');
        return p == std::string::npos ? code : code.substr(0, p);
    }
};

using Status = Expected<Unit>;

inline Status ok_status() { return Status::success(Unit{}); }
inline Status fail_status(std::string code) { return Status::failure(std::move(code)); }

}  // namespace opencrab2
