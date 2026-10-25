// opencrab2/platform/tcp.hpp
//
// Blocking TCP server and client wrappers with timeouts (winsock2 on Windows,
// POSIX sockets elsewhere). Every operation that can fail returns
// opencrab2::Expected<T> with one of the stable codes listed below; nothing
// throws. The wrappers are deliberately small: one socket, one thread, no
// event loop. The 8-byte big-endian length-prefixed frame helpers implement
// the frame layer of the wire codec (codec.hpp).
//
// Codes: TCP_INIT_FAILED, BAD_HOST_PORT, TCP_RESOLVE_FAILED, TCP_REFUSED
// (the peer answered the connection attempt with a reset: nothing listens on
// that port), TCP_CONNECT_FAILED (any other connect error), TCP_BIND_FAILED,
// TCP_LISTEN_FAILED, TCP_ACCEPT_FAILED, TCP_TIMEOUT, TCP_CLOSED,
// TCP_SEND_FAILED, TCP_RECV_FAILED, TCP_INVALID_SOCKET,
// TCP_SETSOCKOPT_FAILED, FRAME_TOO_LARGE.
//
// connect() classifies a refused connection as TCP_REFUSED on every host and
// on both the timed (non-blocking) and the blocking path. Windows reports a
// failed non-blocking connect in the exception set of select(), which is
// watched together with the write set, and it retransmits the SYN after a
// reset for about two seconds; on loopback destinations that retransmission
// is switched off (no SYN can be lost there), so a refused local port is
// reported at once. A refusal that has not been reported when the caller's
// timeout expires (short timeout, remote Windows-style retransmission) is
// still TCP_TIMEOUT: the code states what was observed.
//
// What is tested (tests/test_platform.cpp): loopback listen/connect/send/recv,
// frame round trip, accept timeout, recv timeout, parse_host_port, and
// TCP_REFUSED (not TCP_TIMEOUT) for a closed loopback port on the timed and
// the blocking path, well inside the timeout.
// Design limit: IPv4 and IPv6 literals and names resolve through getaddrinfo;
// the first address that connects wins.
#pragma once

#include <opencrab2/platform/support.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace platform {

#ifdef _WIN32
using socket_handle_t = uintptr_t;
#else
using socket_handle_t = int;
#endif
constexpr socket_handle_t kInvalidSocket = static_cast<socket_handle_t>(-1);

// Idempotent process-wide initialisation (WSAStartup on Windows; no-op on
// POSIX). Called automatically by connect/bind; exposed for apps.
Expected<bool> tcp_global_init();

// Text of the last socket error (for logs; never contains payload data).
std::string tcp_last_error_string();

// "host:port" -> (host, port). IPv6 literals use "[::1]:port". A bare port
// ("8787") yields host 127.0.0.1. Fails closed with BAD_HOST_PORT.
Expected<std::pair<std::string, uint16_t>> parse_host_port(const std::string& s);

double monotonic_s();   // steady clock seconds
double wall_s();        // seconds since the Unix epoch (shared origin for shaping)
void sleep_s(double seconds);

class TcpSocket {
 public:
  TcpSocket() = default;
  explicit TcpSocket(socket_handle_t h) : h_(h) {}
  ~TcpSocket();
  TcpSocket(TcpSocket&& o) noexcept;
  TcpSocket& operator=(TcpSocket&& o) noexcept;
  TcpSocket(const TcpSocket&) = delete;
  TcpSocket& operator=(const TcpSocket&) = delete;

  bool valid() const { return h_ != kInvalidSocket; }
  socket_handle_t native() const { return h_; }
  socket_handle_t release();

  // Receive and send timeout in seconds; 0 restores fully blocking mode.
  Expected<bool> set_timeout(double seconds);
  Expected<bool> set_nodelay(bool on);

  Expected<size_t> send_all(const uint8_t* p, size_t n);
  Expected<size_t> send_all(const std::vector<uint8_t>& v) { return send_all(v.data(), v.size()); }
  Expected<size_t> send_all(const std::string& s) {
    return send_all(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  }

  // Exactly n bytes or a failure (TCP_CLOSED on EOF, TCP_TIMEOUT, TCP_RECV_FAILED).
  Expected<size_t> recv_exact(uint8_t* p, size_t n);
  // Up to n bytes; value 0 means orderly EOF.
  Expected<size_t> recv_some(uint8_t* p, size_t n);

  // 8-byte big-endian length prefix + payload (wire codec frame layer).
  Expected<bool> send_frame(const uint8_t* payload, size_t n);
  Expected<bool> send_frame(const std::vector<uint8_t>& payload) { return send_frame(payload.data(), payload.size()); }
  Expected<std::vector<uint8_t>> recv_frame(size_t max_len = (static_cast<size_t>(1) << 31));

  // Half close: queued data is delivered, then FIN. Use before close() on a
  // socket whose peer may still be sending (an abortive close would reset the
  // connection and discard data the peer has not read yet).
  void shutdown_send();
  void shutdown_both();
  void close();
  std::string peer() const;  // "ip:port" or "" when unknown

  static Expected<TcpSocket> connect(const std::string& host, uint16_t port, double timeout_s = 10.0,
                                     int retries = 1, double retry_delay_s = 0.05);

 private:
  socket_handle_t h_ = kInvalidSocket;
};

class TcpListener {
 public:
  TcpListener() = default;
  ~TcpListener();
  TcpListener(TcpListener&& o) noexcept;
  TcpListener& operator=(TcpListener&& o) noexcept;
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;

  // port 0 selects an ephemeral port; port() reports the bound one.
  static Expected<TcpListener> bind(const std::string& host, uint16_t port, int backlog = 16);

  bool valid() const { return h_ != kInvalidSocket; }
  uint16_t port() const { return port_; }
  std::string host() const { return host_; }

  // timeout_s <= 0 blocks; otherwise TCP_TIMEOUT after the wait.
  Expected<TcpSocket> accept(double timeout_s = 0.0);
  void close();

 private:
  socket_handle_t h_ = kInvalidSocket;
  uint16_t port_ = 0;
  std::string host_;
};

}  // namespace platform
}  // namespace opencrab2
