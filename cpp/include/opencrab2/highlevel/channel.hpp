// OpenCrab v2 C++17 mirror: TCP channel with the binding wire codec
// (codec.hpp, the bytes of the Python opencrab2.highlevel.channel), exact byte
// and round counters, and the WallClockShaper (paper, Section 7.1).
//
// Frame = 8-byte big-endian length + typed payload (codec.hpp). stats.bytes_*
// count payload plus the 8-byte prefix. Round accounting mirrors the Python
// Channel: exchange() counts one round on both sides; a bare send() followed
// later by a recv() on the same party counts one round when the pair closes.
//
// WallClockShaper: the bandwidth and one-way delay are read from an exogenous
// trace e(wall_time) indexed by elapsed wall time since a shared origin. A
// message of n bytes started at wall time w is serialized by a token bucket
// whose rate follows the trace across segment boundaries (a change applies
// mid-transfer); the sender sleeps for serialization + one-way delay before
// writing to the socket, so the receiver observes the shaped arrival time.
// One rule for both languages: the one-way delay added to a message is
// that of the state in force when its LAST byte leaves the bucket; a bandwidth of 0 means
// "unshaped" for that segment (whatever is left of a transfer when it reaches the segment leaves
// at once; a trace cannot express an outage). Bandwidth and delay must be finite and >= 0 and
// times finite: anything else is CHANNEL_SHAPER_PARAMS (WallClockShaper::make returns the code;
// a shaper built with the plain constructor from such a trace is invalid and every send on its
// channel fails closed with the same code).
//
// Frame length cap: the 8-byte prefix is chosen by the peer, so recv refuses a
// length above max_frame_bytes() with CHANNEL_FRAME_TOO_LARGE before reading or
// allocating anything (default 2 GiB, set_max_frame_bytes to tighten it to the
// largest public message shape of a session). The payload buffer then grows
// only as bytes actually arrive (16 MiB steps), so an announced length alone
// never commits memory; an allocation failure returns CHANNEL_FRAME_ALLOC.
// send refuses an over-long frame with the same code before writing a byte, so
// the stream stays in sync. Nesting depth of a decoded payload is limited by
// the codec (CODEC_DEPTH, codec.hpp).
//
// Sockets: winsock2 on Windows (link -lws2_32), POSIX sockets elsewhere.
// Every failing operation returns Expected with a CHANNEL_* code.
#pragma once

#include <opencrab2/highlevel/codec.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace highlevel {

#ifdef _WIN32
using socket_t = uintptr_t;
#else
using socket_t = int;
#endif

struct ChannelStats {
    uint64_t bytes_sent = 0;
    uint64_t bytes_recv = 0;
    uint64_t messages_sent = 0;
    uint64_t messages_recv = 0;
    uint64_t rounds = 0;
    double shaping_delay_s = 0.0;
    Value snapshot() const;
};

struct LinkState {
    double bandwidth_bps = 0.0;    // 0: unshaped; negative or non-finite: CHANNEL_SHAPER_PARAMS
    double one_way_delay_s = 0.0;  // per message
};

struct TracePoint {
    double elapsed_s = 0.0;  // since the shared origin
    LinkState state;
};

class WallClockShaper {
public:
    // trace sorted by elapsed_s (sorted here defensively); origin_wall_s is a
    // now_s() value shared by both parties (exchanged out of band).
    WallClockShaper(std::vector<TracePoint> trace, double origin_wall_s);
    // Validating factory: CHANNEL_SHAPER_PARAMS for an empty trace, a non-finite time or origin,
    // a negative or non-finite bandwidth or delay.
    static Expected<std::shared_ptr<WallClockShaper>> make(std::vector<TracePoint> trace, double origin_wall_s);
    static Expected<Unit> validate_trace(const std::vector<TracePoint>& trace);
    // False when the constructor was given a trace that validate_trace refuses; error() is the code.
    bool valid() const { return error_.empty(); }
    const std::string& error() const { return error_; }
    LinkState state_at(double wall_s) const;
    // Seconds to drain nbytes through an IDLE link starting at wall_s (pure, no queueing).
    double serialization_time(size_t nbytes, double wall_s) const;
    // Total delay for a message of nbytes handed over at wall_s: queueing behind earlier messages,
    // serialization across segments, plus the one-way delay of the state in force when the last
    // byte leaves. Advances the internal link-busy time. NaN on an invalid shaper.
    double delay_for(size_t nbytes, double wall_s);
    double origin_wall_s() const { return origin_; }
    const std::vector<TracePoint>& trace() const { return trace_; }

private:
    double drain_end(size_t nbytes, double start) const;
    std::vector<TracePoint> trace_;
    double origin_;
    double link_busy_until_;
    std::string error_;
    mutable std::mutex mu_;
};

// The socket channel is also the base class of transport adapters (the
// platform layer's WebSocket session bridge derives from it): send, recv,
// exchange and close are virtual, and the protected constructor builds a
// channel without a socket. Round accounting stays in the base class.
class Channel {
public:
    Channel(socket_t sock, int party, std::shared_ptr<WallClockShaper> shaper = nullptr);
    virtual ~Channel();
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    virtual Expected<Unit> send(const Value& obj);
    virtual Expected<Value> recv();
    // P0 sends then receives; P1 receives then sends; counts one round.
    virtual Expected<Value> exchange(const Value& obj);
    // One-directional message that still forms a protocol round (reveal_to).
    void count_round() { stats.rounds += 1; }
    // Forget a pending bare send (callers that account rounds explicitly).
    void clear_pending() { pending_send_ = false; }
    void set_shaper(std::shared_ptr<WallClockShaper> shaper);
    std::shared_ptr<WallClockShaper> shaper() const { return shaper_; }
    virtual void close();
    bool is_open() const { return open_; }

    // Largest accepted (and sent) frame payload in bytes, excluding the 8-byte prefix.
    static constexpr uint64_t kDefaultMaxFrameBytes = static_cast<uint64_t>(1) << 31;   // 2 GiB
    static constexpr uint64_t kHardMaxFrameBytes = static_cast<uint64_t>(1) << 40;      // ceiling of the setter
    // Fails with CHANNEL_FRAME_LIMIT (and changes nothing) unless 1 <= n <= kHardMaxFrameBytes.
    Expected<Unit> set_max_frame_bytes(uint64_t n);
    uint64_t max_frame_bytes() const { return max_frame_bytes_; }

    ChannelStats stats;
    int party;

protected:
    // Socket-less channel for derived transports (open_ = false, no socket is ever closed).
    explicit Channel(int party);
    bool pending_send_ = false;
    // Derived transports enforce the same cap on their own framing.
    uint64_t max_frame_bytes_ = kDefaultMaxFrameBytes;

private:
    Expected<Unit> send_all(const uint8_t* p, size_t n);
    Expected<Unit> recv_all(uint8_t* p, size_t n);
    Expected<Unit> send_frame(const Value& obj);
    Expected<Value> recv_frame();

    socket_t sock_;
    bool open_;
    std::shared_ptr<WallClockShaper> shaper_;
};

// Initializes the socket subsystem (WSAStartup on Windows); idempotent.
Expected<Unit> ensure_sockets();

// Binds host:port (port 0 = ephemeral), accepts one connection, returns the
// channel and the bound port.
Expected<std::pair<std::unique_ptr<Channel>, int>> listen_channel(const std::string& host, int port, int party,
                                                                  std::shared_ptr<WallClockShaper> shaper = nullptr);
// Bind-only variant for callers that must publish the port before accepting.
struct Listener {
    socket_t sock = 0;
    int port = 0;
    bool valid = false;
};
Expected<Listener> listen_socket(const std::string& host, int port);
Expected<std::unique_ptr<Channel>> accept_channel(Listener& l, int party,
                                                  std::shared_ptr<WallClockShaper> shaper = nullptr);
Expected<std::unique_ptr<Channel>> dial_channel(const std::string& host, int port, int party,
                                                std::shared_ptr<WallClockShaper> shaper = nullptr, int retries = 500);
// Two connected loopback channels (party 0 first) for in-process tests.
Expected<std::pair<std::unique_ptr<Channel>, std::unique_ptr<Channel>>> loopback_pair(
    std::shared_ptr<WallClockShaper> shaper0 = nullptr, std::shared_ptr<WallClockShaper> shaper1 = nullptr);

void sleep_s(double seconds);

}  // namespace highlevel
}  // namespace opencrab2
