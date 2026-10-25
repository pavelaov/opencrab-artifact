// Implementation of channel.hpp. See the header for the contracts.
#include <opencrab2/highlevel/channel.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#endif

namespace opencrab2 {
namespace highlevel {

namespace {

#ifdef _WIN32
const socket_t kInvalidSocket = static_cast<socket_t>(INVALID_SOCKET);
inline void close_socket(socket_t s) { closesocket(static_cast<SOCKET>(s)); }
inline int last_socket_error() { return WSAGetLastError(); }
inline bool would_block(int e) { return e == WSAEWOULDBLOCK; }
inline bool interrupted(int) { return false; }
const int kSendFlags = 0;
#else
const socket_t kInvalidSocket = -1;
inline void close_socket(socket_t s) { ::close(s); }
inline int last_socket_error() { return errno; }
inline bool would_block(int e) { return e == EWOULDBLOCK || e == EAGAIN; }
// A signal handler that ran during the call: nothing was transferred, the call is repeated.
inline bool interrupted(int e) { return e == EINTR; }
// A send to a peer that has closed reports EPIPE (CHANNEL_SEND) instead of raising SIGPIPE,
// whose default action would end the process before it can fail closed.
const int kSendFlags = MSG_NOSIGNAL;
#endif

bool set_nodelay(socket_t s) {
    int one = 1;
    return setsockopt(static_cast<
#ifdef _WIN32
                          SOCKET
#else
                          int
#endif
                          >(s),
                      IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one)) == 0;
}

}  // namespace

// Precise sleep for the shaper: the OS timer granularity (about 15 ms on
// Windows by default) would turn a 0.5 ms one-way delay into 15 ms, so the bulk
// of the wait is slept and the last two milliseconds are spun on the clock.
void sleep_s(double seconds) {
    if (seconds <= 0) return;
    double target = now_s() + seconds;
    if (seconds > 0.004) std::this_thread::sleep_for(std::chrono::duration<double>(seconds - 0.002));
    while (now_s() < target) std::this_thread::yield();
}

Value ChannelStats::snapshot() const {
    return dict({{"bytes_sent", static_cast<int64_t>(bytes_sent)},
                 {"bytes_recv", static_cast<int64_t>(bytes_recv)},
                 {"bytes_total", static_cast<int64_t>(bytes_sent + bytes_recv)},
                 {"messages_sent", static_cast<int64_t>(messages_sent)},
                 {"messages_recv", static_cast<int64_t>(messages_recv)},
                 {"rounds", static_cast<int64_t>(rounds)},
                 {"shaping_delay_s", shaping_delay_s}});
}

// ---------------------------------------------------------------------------
// WallClockShaper
// ---------------------------------------------------------------------------
Expected<Unit> WallClockShaper::validate_trace(const std::vector<TracePoint>& trace) {
    if (trace.empty()) return fail("CHANNEL_SHAPER_PARAMS: trace must contain at least one segment");
    for (const auto& tp : trace) {
        if (!std::isfinite(tp.elapsed_s)) return fail("CHANNEL_SHAPER_PARAMS: elapsed_s must be finite");
        if (!std::isfinite(tp.state.bandwidth_bps) || tp.state.bandwidth_bps < 0.0)
            return fail("CHANNEL_SHAPER_PARAMS: bandwidth_bps must be finite and >= 0 (0 = unshaped)");
        if (!std::isfinite(tp.state.one_way_delay_s) || tp.state.one_way_delay_s < 0.0)
            return fail("CHANNEL_SHAPER_PARAMS: one_way_delay_s must be finite and >= 0");
    }
    return ok_unit();
}

Expected<std::shared_ptr<WallClockShaper>> WallClockShaper::make(std::vector<TracePoint> trace, double origin_wall_s) {
    OC2_CHECK(validate_trace(trace));
    if (!std::isfinite(origin_wall_s)) return fail("CHANNEL_SHAPER_PARAMS: origin must be finite");
    return ok(std::make_shared<WallClockShaper>(std::move(trace), origin_wall_s));
}

WallClockShaper::WallClockShaper(std::vector<TracePoint> trace, double origin_wall_s)
    : trace_(std::move(trace)), origin_(origin_wall_s), link_busy_until_(-std::numeric_limits<double>::infinity()) {
    // An empty trace is the documented "no shaping" default of the constructor (one unshaped
    // segment); every other defect (NaN, negative bandwidth or delay) makes the shaper invalid:
    // delay_for returns NaN and Channel::send fails closed with the stored code.
    if (trace_.empty()) trace_.push_back(TracePoint{0.0, LinkState{}});
    auto st = validate_trace(trace_);
    if (st.ok && !std::isfinite(origin_)) st = fail("CHANNEL_SHAPER_PARAMS: origin must be finite");
    if (!st.ok) error_ = st.code;
    std::stable_sort(trace_.begin(), trace_.end(),
                     [](const TracePoint& a, const TracePoint& b) { return a.elapsed_s < b.elapsed_s; });
}

LinkState WallClockShaper::state_at(double wall_s) const {
    double el = wall_s - origin_;
    LinkState st = trace_.front().state;
    for (const auto& tp : trace_) {
        if (tp.elapsed_s <= el) st = tp.state;
        else break;
    }
    return st;
}

// End of serialization of nbytes that start draining at wall time `start` (channel.hpp, one rule):
// the smallest t with integral_start^t bandwidth = 8 nbytes. A segment with bandwidth 0 is
// unshaped: whatever is left when the transfer reaches it leaves at once.
double WallClockShaper::drain_end(size_t nbytes, double start) const {
    double t = start;
    double remaining = static_cast<double>(nbytes);
    size_t idx = 0;
    for (size_t i = 0; i < trace_.size(); ++i) {
        if (origin_ + trace_[i].elapsed_s <= t) idx = i;
        else break;
    }
    while (remaining > 0.0) {
        const LinkState& st = trace_[idx].state;
        double seg_end = (idx + 1 < trace_.size()) ? origin_ + trace_[idx + 1].elapsed_s
                                                   : std::numeric_limits<double>::infinity();
        if (st.bandwidth_bps <= 0.0) break;   // unshaped segment: the rest leaves immediately
        double rate_Bps = st.bandwidth_bps / 8.0;
        double capacity = (seg_end - t) * rate_Bps;
        if (capacity < 0.0) capacity = 0.0;
        if (remaining <= capacity) {
            t += remaining / rate_Bps;
            remaining = 0.0;
        } else {
            remaining -= capacity;
            t = seg_end;
            ++idx;
            if (idx >= trace_.size()) idx = trace_.size() - 1;
        }
    }
    return t;
}

double WallClockShaper::serialization_time(size_t nbytes, double wall_s) const {
    if (!valid()) return std::numeric_limits<double>::quiet_NaN();
    return drain_end(nbytes, wall_s) - wall_s;
}

double WallClockShaper::delay_for(size_t nbytes, double wall_s) {
    if (!valid()) return std::numeric_limits<double>::quiet_NaN();
    std::lock_guard<std::mutex> g(mu_);
    double start = std::max(wall_s, link_busy_until_);
    double t = drain_end(nbytes, start);
    link_busy_until_ = t;
    // One rule for both languages: the one-way delay of the state in force when the LAST byte
    // leaves the bucket (the Python shaper: state_at(wall_s + serialization)).
    double owd = state_at(t).one_way_delay_s;
    return (t - wall_s) + owd;
}

// ---------------------------------------------------------------------------
// sockets
// ---------------------------------------------------------------------------
Expected<Unit> ensure_sockets() {
#ifdef _WIN32
    static bool done = false;
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (!done) {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return fail("CHANNEL_WSASTARTUP");
        done = true;
    }
#endif
    return ok_unit();
}

Channel::Channel(socket_t sock, int party_, std::shared_ptr<WallClockShaper> shaper)
    : party(party_), sock_(sock), open_(true), shaper_(std::move(shaper)) {
    set_nodelay(sock_);
}

Channel::Channel(int party_) : party(party_), sock_(0), open_(false) {}

Channel::~Channel() { close(); }

void Channel::close() {
    if (open_) {
#ifdef _WIN32
        shutdown(static_cast<SOCKET>(sock_), SD_BOTH);
#else
        shutdown(sock_, SHUT_RDWR);
#endif
        close_socket(sock_);
        open_ = false;
    }
}

void Channel::set_shaper(std::shared_ptr<WallClockShaper> shaper) { shaper_ = std::move(shaper); }

Expected<Unit> Channel::send_all(const uint8_t* p, size_t n) {
    while (n > 0) {
        int chunk = static_cast<int>(std::min<size_t>(n, 1u << 20));
#ifdef _WIN32
        int r = ::send(static_cast<SOCKET>(sock_), reinterpret_cast<const char*>(p), chunk, kSendFlags);
#else
        ssize_t r = ::send(sock_, p, static_cast<size_t>(chunk), kSendFlags);
#endif
        if (r <= 0) {
            int e = last_socket_error();
            if (would_block(e) || interrupted(e)) continue;
            return fail("CHANNEL_SEND: errno " + std::to_string(e));
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return ok_unit();
}

Expected<Unit> Channel::recv_all(uint8_t* p, size_t n) {
    while (n > 0) {
        int chunk = static_cast<int>(std::min<size_t>(n, 1u << 20));
#ifdef _WIN32
        int r = ::recv(static_cast<SOCKET>(sock_), reinterpret_cast<char*>(p), chunk, 0);
#else
        ssize_t r = ::recv(sock_, p, static_cast<size_t>(chunk), 0);
#endif
        if (r == 0) return fail("CHANNEL_CLOSED: peer closed during recv");
        if (r < 0) {
            int e = last_socket_error();
            if (would_block(e) || interrupted(e)) continue;
            return fail("CHANNEL_RECV: errno " + std::to_string(e));
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return ok_unit();
}

Expected<Unit> Channel::set_max_frame_bytes(uint64_t n) {
    if (n == 0 || n > kHardMaxFrameBytes) return fail("CHANNEL_FRAME_LIMIT: max frame bytes must be in [1, 2^40]");
    max_frame_bytes_ = n;
    return ok_unit();
}

Expected<Unit> Channel::send_frame(const Value& obj) {
    if (!open_) return fail("CHANNEL_CLOSED");
    Bytes frame = encode_frame(obj);
    // Refuse before the first byte leaves: the peer would refuse the frame anyway and a partial
    // write would desynchronize the stream. The size depends on public shapes only.
    if (frame.size() - 8 > max_frame_bytes_) return fail("CHANNEL_FRAME_TOO_LARGE: outgoing frame exceeds the cap");
    if (shaper_) {
        if (!shaper_->valid()) return fail(shaper_->error());   // fail closed: no unshaped send on a bad trace
        double d = shaper_->delay_for(frame.size(), now_s());
        if (d > 0) {
            stats.shaping_delay_s += d;
            sleep_s(d);
        }
    }
    OC2_CHECK(send_all(frame.data(), frame.size()));
    stats.bytes_sent += frame.size();
    stats.messages_sent += 1;
    return ok_unit();
}

Expected<Value> Channel::recv_frame() {
    if (!open_) return fail("CHANNEL_CLOSED");
    uint8_t hdr[8];
    OC2_CHECK(recv_all(hdr, 8));
    uint64_t len = get_u64be(hdr);
    // The length is chosen by the peer: refuse before reading or allocating anything.
    if (len > max_frame_bytes_) {
        // The unread payload would be misparsed as the next frame: the channel is dead from here.
        close();
        return fail("CHANNEL_FRAME_TOO_LARGE: announced " + std::to_string(len) + " bytes");
    }
    // Grow the buffer only as bytes actually arrive, so an announced length alone commits no
    // memory; an allocation failure fails closed instead of terminating the process.
    Bytes payload;
    const uint64_t kStep = static_cast<uint64_t>(1) << 24;   // 16 MiB
    uint64_t got = 0;
    while (got < len) {
        const size_t step = static_cast<size_t>(std::min<uint64_t>(kStep, len - got));
        try {
            payload.resize(static_cast<size_t>(got) + step);
        } catch (const std::exception&) {
            close();
            return fail("CHANNEL_FRAME_ALLOC: cannot allocate the announced frame");
        }
        OC2_CHECK(recv_all(payload.data() + static_cast<size_t>(got), step));
        got += step;
    }
    stats.bytes_recv += 8 + len;
    stats.messages_recv += 1;
    return decode_value(payload);
}

Expected<Unit> Channel::send(const Value& obj) {
    OC2_CHECK(send_frame(obj));
    pending_send_ = true;
    return ok_unit();
}

Expected<Value> Channel::recv() {
    OC2_TRY(v, recv_frame());
    if (pending_send_) {
        stats.rounds += 1;
        pending_send_ = false;
    }
    return ok(std::move(v));
}

Expected<Value> Channel::exchange(const Value& obj) {
    stats.rounds += 1;
    pending_send_ = false;
    if (party == 0) {
        OC2_CHECK(send_frame(obj));
        return recv_frame();
    }
    OC2_TRY(other, recv_frame());
    OC2_CHECK(send_frame(obj));
    return ok(std::move(other));
}

// ---------------------------------------------------------------------------
// listen / dial
// ---------------------------------------------------------------------------
Expected<Listener> listen_socket(const std::string& host, int port) {
    OC2_CHECK(ensure_sockets());
    socket_t ls = static_cast<socket_t>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (ls == kInvalidSocket) return fail("CHANNEL_SOCKET");
    int one = 1;
#ifdef _WIN32
    setsockopt(static_cast<SOCKET>(ls), SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
#else
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close_socket(ls);
        return fail("CHANNEL_ADDRESS");
    }
    if (::bind(static_cast<
#ifdef _WIN32
                   SOCKET
#else
                   int
#endif
                   >(ls),
               reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_socket(ls);
        return fail("CHANNEL_BIND: errno " + std::to_string(last_socket_error()));
    }
    if (::listen(static_cast<
#ifdef _WIN32
                     SOCKET
#else
                     int
#endif
                     >(ls),
                 1) != 0) {
        close_socket(ls);
        return fail("CHANNEL_LISTEN");
    }
    sockaddr_in bound;
    std::memset(&bound, 0, sizeof(bound));
#ifdef _WIN32
    int blen = sizeof(bound);
#else
    socklen_t blen = sizeof(bound);
#endif
    getsockname(static_cast<
#ifdef _WIN32
                    SOCKET
#else
                    int
#endif
                    >(ls),
                reinterpret_cast<sockaddr*>(&bound), &blen);
    Listener l;
    l.sock = ls;
    l.port = ntohs(bound.sin_port);
    l.valid = true;
    return ok(l);
}

Expected<std::unique_ptr<Channel>> accept_channel(Listener& l, int party, std::shared_ptr<WallClockShaper> shaper) {
    if (!l.valid) return fail("CHANNEL_LISTENER_INVALID");
    socket_t conn = kInvalidSocket;
    do {
        conn = static_cast<socket_t>(::accept(static_cast<
#ifdef _WIN32
                                                  SOCKET
#else
                                                  int
#endif
                                                  >(l.sock),
                                              nullptr, nullptr));
    } while (conn == kInvalidSocket && interrupted(last_socket_error()));
    close_socket(l.sock);
    l.valid = false;
    if (conn == kInvalidSocket) return fail("CHANNEL_ACCEPT");
    return ok(std::make_unique<Channel>(conn, party, std::move(shaper)));
}

Expected<std::pair<std::unique_ptr<Channel>, int>> listen_channel(const std::string& host, int port, int party,
                                                                  std::shared_ptr<WallClockShaper> shaper) {
    OC2_TRY(l, listen_socket(host, port));
    int bound = l.port;
    OC2_TRY(ch, accept_channel(l, party, std::move(shaper)));
    return ok(std::make_pair(std::move(ch), bound));
}

Expected<std::unique_ptr<Channel>> dial_channel(const std::string& host, int port, int party,
                                                std::shared_ptr<WallClockShaper> shaper, int retries) {
    OC2_CHECK(ensure_sockets());
    std::string last;
    for (int attempt = 0; attempt < retries; ++attempt) {
        socket_t s = static_cast<socket_t>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
        if (s == kInvalidSocket) return fail("CHANNEL_SOCKET");
        sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            close_socket(s);
            return fail("CHANNEL_ADDRESS");
        }
        if (::connect(static_cast<
#ifdef _WIN32
                          SOCKET
#else
                          int
#endif
                          >(s),
                      reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            return ok(std::make_unique<Channel>(s, party, std::move(shaper)));
        }
        last = std::to_string(last_socket_error());
        close_socket(s);
        sleep_s(0.02);
    }
    return fail("CHANNEL_CONNECT: " + host + ":" + std::to_string(port) + " errno " + last);
}

Expected<std::pair<std::unique_ptr<Channel>, std::unique_ptr<Channel>>> loopback_pair(
    std::shared_ptr<WallClockShaper> shaper0, std::shared_ptr<WallClockShaper> shaper1) {
    OC2_TRY(l, listen_socket("127.0.0.1", 0));
    int port = l.port;
    Expected<std::unique_ptr<Channel>> dialed;
    std::thread th([&]() { dialed = dial_channel("127.0.0.1", port, 1, shaper1, 200); });
    auto accepted = accept_channel(l, 0, shaper0);
    th.join();
    if (!accepted.ok) return fail(accepted.code);
    if (!dialed.ok) return fail(dialed.code);
    return ok(std::make_pair(std::move(accepted.value), std::move(dialed.value)));
}

}  // namespace highlevel
}  // namespace opencrab2
