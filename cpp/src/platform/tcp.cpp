// opencrab2/platform/tcp.cpp (see tcp.hpp)
#include <opencrab2/platform/tcp.hpp>

#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace opencrab2 {
namespace platform {

namespace {

#ifdef _WIN32
inline int last_err() { return WSAGetLastError(); }
inline bool err_is_timeout(int e) { return e == WSAETIMEDOUT || e == WSAEWOULDBLOCK; }
inline bool err_is_interrupt(int e) { return e == WSAEINTR; }
inline void close_handle(socket_handle_t h) { closesocket(static_cast<SOCKET>(h)); }
inline bool set_nonblocking(socket_handle_t h, bool on) {
  u_long mode = on ? 1 : 0;
  return ioctlsocket(static_cast<SOCKET>(h), FIONBIO, &mode) == 0;
}
using sock_len_t = int;
#else
inline int last_err() { return errno; }
inline bool err_is_timeout(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
inline bool err_is_interrupt(int e) { return e == EINTR; }
inline void close_handle(socket_handle_t h) { ::close(h); }
inline bool set_nonblocking(socket_handle_t h, bool on) {
  int flags = fcntl(h, F_GETFL, 0);
  if (flags < 0) return false;
  flags = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
  return fcntl(h, F_SETFL, flags) == 0;
}
using sock_len_t = socklen_t;
#endif

std::string addr_to_string(const sockaddr* sa, sock_len_t len) {
  char host[NI_MAXHOST];
  char serv[NI_MAXSERV];
  if (getnameinfo(sa, len, host, sizeof host, serv, sizeof serv, NI_NUMERICHOST | NI_NUMERICSERV) != 0) return "";
  std::string h = host;
  if (h.find(':') != std::string::npos) h = "[" + h + "]";
  return h + ":" + serv;
}

#ifndef _WIN32
// poll() on one descriptor until `events` is reported or the timeout runs out;
// an interrupted wait is resumed with the time that is left. poll() has no
// descriptor limit, unlike select(), whose fd_set holds descriptors below
// FD_SETSIZE only. Returns 1 ready, 0 timeout, -1 error (errno set; EBADF for
// a descriptor that is not open).
int poll_fd(socket_handle_t h, short events, double timeout_s) {
  if (h < 0) {
    errno = EBADF;
    return -1;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (true) {
    double left = std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count();
    if (left < 0) left = 0;
    pollfd p;
    p.fd = h;
    p.events = events;
    p.revents = 0;
    int ms = left > 2147483.0 ? 2147483000 : static_cast<int>(left * 1000.0 + 0.999);
    int r = ::poll(&p, 1, ms);
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (r == 0) return 0;
    if (p.revents & POLLNVAL) {
      errno = EBADF;
      return -1;
    }
    return 1;
  }
}
#endif

// Waits for readability (or writability) with a timeout: select() on Windows,
// poll() elsewhere. Returns 1 ready, 0 timeout, -1 error.
int wait_fd(socket_handle_t h, bool for_write, double timeout_s) {
#ifdef _WIN32
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(static_cast<SOCKET>(h), &fds);
  timeval tv;
  tv.tv_sec = static_cast<long>(timeout_s);
  tv.tv_usec = static_cast<long>((timeout_s - static_cast<double>(tv.tv_sec)) * 1e6);
  int nfds = static_cast<int>(h) + 1;
  int r;
  if (for_write) {
    r = select(nfds, nullptr, &fds, nullptr, &tv);
  } else {
    r = select(nfds, &fds, nullptr, nullptr, &tv);
  }
  return r;
#else
  return poll_fd(h, for_write ? POLLOUT : POLLIN, timeout_s);
#endif
}

// Stable code of a failed connect() from the socket error.
const char* connect_error_code(int e) {
#ifdef _WIN32
  if (e == WSAECONNREFUSED) return "TCP_REFUSED";
  if (e == WSAETIMEDOUT) return "TCP_TIMEOUT";
#else
  if (e == ECONNREFUSED) return "TCP_REFUSED";
  if (e == ETIMEDOUT) return "TCP_TIMEOUT";
#endif
  return "TCP_CONNECT_FAILED";
}

// Waits for a non-blocking connect to finish. Returns 1 connected, 0 timeout,
// -1 failed with the socket error in `err`. Windows reports a failed
// non-blocking connect in the exception set of select() (never in the write
// set), so both sets are watched there; POSIX reports it as writable (poll()
// adds POLLERR) with SO_ERROR set. SO_ERROR decides on both.
int wait_connect(socket_handle_t h, double timeout_s, int& err) {
#ifdef _WIN32
  fd_set wfds, efds;
  FD_ZERO(&wfds);
  FD_ZERO(&efds);
  FD_SET(static_cast<SOCKET>(h), &wfds);
  FD_SET(static_cast<SOCKET>(h), &efds);
  timeval tv;
  tv.tv_sec = static_cast<long>(timeout_s);
  tv.tv_usec = static_cast<long>((timeout_s - static_cast<double>(tv.tv_sec)) * 1e6);
  int r = select(static_cast<int>(h) + 1, nullptr, &wfds, &efds, &tv);
  if (r == 0) return 0;
  if (r < 0) {
    err = last_err();
    return -1;
  }
  int so = 0;
  sock_len_t el = sizeof so;
  bool failed_set = FD_ISSET(static_cast<SOCKET>(h), &efds) != 0;
  getsockopt(static_cast<SOCKET>(h), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so), &el);
  if (so == 0 && failed_set) so = WSAECONNREFUSED;
#else
  int r = poll_fd(h, POLLOUT, timeout_s);
  if (r == 0) return 0;
  if (r < 0) {
    err = last_err();
    return -1;
  }
  int so = 0;
  sock_len_t el = sizeof so;
  getsockopt(h, SOL_SOCKET, SO_ERROR, &so, &el);
#endif
  if (so != 0) {
    err = so;
    return -1;
  }
  return 1;
}

#ifdef _WIN32
// Windows answers a RST to a SYN by retransmitting the SYN (about two seconds
// until WSAECONNREFUSED). On a loopback destination no SYN can be lost, so
// the retransmissions are switched off there and a refused port is reported
// at once; other destinations keep the system behaviour. The ioctl exists
// since Windows 8; a failure is ignored (the connect then takes the slow path).
void fast_refusal_on_loopback(socket_handle_t h, const sockaddr* sa) {
  bool loopback = false;
  if (sa->sa_family == AF_INET) {
    loopback = (ntohl(reinterpret_cast<const sockaddr_in*>(sa)->sin_addr.s_addr) >> 24) == 127;
  } else if (sa->sa_family == AF_INET6) {
    static const uint8_t kLoop6[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    loopback = std::memcmp(&reinterpret_cast<const sockaddr_in6*>(sa)->sin6_addr, kLoop6, 16) == 0;
  }
  if (!loopback) return;
  struct InitialRto {
    USHORT rtt;                    // 0xFFFF: unspecified, keep the system value
    UCHAR max_syn_retransmissions;  // 0xFE: no SYN retransmissions
  } p = {0xFFFF, 0xFE};
  const DWORD kSioTcpInitialRto = 0x98000011u;  // _WSAIOW(IOC_VENDOR, 17)
  DWORD out = 0;
  WSAIoctl(static_cast<SOCKET>(h), kSioTcpInitialRto, &p, sizeof p, nullptr, 0, &out, nullptr, nullptr);
}
#endif

}  // namespace

Expected<bool> tcp_global_init() {
#ifdef _WIN32
  static std::once_flag once;
  static int status = -1;
  std::call_once(once, []() {
    WSADATA w;
    status = WSAStartup(MAKEWORD(2, 2), &w);
  });
  if (status != 0) return make_fail<bool>("TCP_INIT_FAILED");
#endif
  return make_ok(true);
}

std::string tcp_last_error_string() {
  int e = last_err();
  return "errno=" + std::to_string(e);
}

Expected<std::pair<std::string, uint16_t>> parse_host_port(const std::string& s) {
  using R = std::pair<std::string, uint16_t>;
  std::string t = detail::trim(s);
  if (t.empty()) return make_fail<R>("BAD_HOST_PORT");
  std::string host;
  std::string port;
  if (t[0] == '[') {
    size_t close = t.find(']');
    if (close == std::string::npos || close + 1 >= t.size() || t[close + 1] != ':') return make_fail<R>("BAD_HOST_PORT");
    host = t.substr(1, close - 1);
    port = t.substr(close + 2);
  } else {
    size_t colon = t.rfind(':');
    if (colon == std::string::npos) {
      host = "127.0.0.1";
      port = t;
    } else {
      host = t.substr(0, colon);
      port = t.substr(colon + 1);
    }
  }
  if (host.empty() || port.empty()) return make_fail<R>("BAD_HOST_PORT");
  for (char c : port) {
    if (c < '0' || c > '9') return make_fail<R>("BAD_HOST_PORT");
  }
  if (port.size() > 5) return make_fail<R>("BAD_HOST_PORT");
  long v = std::strtol(port.c_str(), nullptr, 10);
  if (v < 0 || v > 65535) return make_fail<R>("BAD_HOST_PORT");
  return make_ok(R(host, static_cast<uint16_t>(v)));
}

double monotonic_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

double wall_s() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

void sleep_s(double seconds) {
  if (seconds <= 0) return;
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

// ---------------------------------------------------------------------------
// TcpSocket
// ---------------------------------------------------------------------------
TcpSocket::~TcpSocket() { close(); }

TcpSocket::TcpSocket(TcpSocket&& o) noexcept : h_(o.h_) { o.h_ = kInvalidSocket; }

TcpSocket& TcpSocket::operator=(TcpSocket&& o) noexcept {
  if (this != &o) {
    close();
    h_ = o.h_;
    o.h_ = kInvalidSocket;
  }
  return *this;
}

socket_handle_t TcpSocket::release() {
  socket_handle_t h = h_;
  h_ = kInvalidSocket;
  return h;
}

Expected<bool> TcpSocket::set_timeout(double seconds) {
  if (!valid()) return make_fail<bool>("TCP_INVALID_SOCKET");
  if (seconds < 0) seconds = 0;
#ifdef _WIN32
  DWORD ms = static_cast<DWORD>(seconds * 1000.0);
  if (setsockopt(static_cast<SOCKET>(h_), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof ms) != 0)
    return make_fail<bool>("TCP_SETSOCKOPT_FAILED");
  if (setsockopt(static_cast<SOCKET>(h_), SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof ms) != 0)
    return make_fail<bool>("TCP_SETSOCKOPT_FAILED");
#else
  timeval tv;
  tv.tv_sec = static_cast<time_t>(seconds);
  tv.tv_usec = static_cast<suseconds_t>((seconds - static_cast<double>(tv.tv_sec)) * 1e6);
  if (setsockopt(h_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0) return make_fail<bool>("TCP_SETSOCKOPT_FAILED");
  if (setsockopt(h_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv) != 0) return make_fail<bool>("TCP_SETSOCKOPT_FAILED");
#endif
  return make_ok(true);
}

Expected<bool> TcpSocket::set_nodelay(bool on) {
  if (!valid()) return make_fail<bool>("TCP_INVALID_SOCKET");
  int flag = on ? 1 : 0;
#ifdef _WIN32
  if (setsockopt(static_cast<SOCKET>(h_), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&flag), sizeof flag) != 0)
    return make_fail<bool>("TCP_SETSOCKOPT_FAILED");
#else
  if (setsockopt(h_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof flag) != 0) return make_fail<bool>("TCP_SETSOCKOPT_FAILED");
#endif
  return make_ok(true);
}

Expected<size_t> TcpSocket::send_all(const uint8_t* p, size_t n) {
  if (!valid()) return make_fail<size_t>("TCP_INVALID_SOCKET");
  size_t sent = 0;
  while (sent < n) {
    size_t chunk = n - sent;
    if (chunk > (static_cast<size_t>(1) << 20)) chunk = static_cast<size_t>(1) << 20;
#ifdef _WIN32
    int r = ::send(static_cast<SOCKET>(h_), reinterpret_cast<const char*>(p + sent), static_cast<int>(chunk), 0);
#else
    ssize_t r = ::send(h_, p + sent, chunk, MSG_NOSIGNAL);
#endif
    if (r < 0) {
      int e = last_err();
      if (err_is_interrupt(e)) continue;
      if (err_is_timeout(e)) return make_fail<size_t>("TCP_TIMEOUT");
      return make_fail<size_t>("TCP_SEND_FAILED");
    }
    if (r == 0) return make_fail<size_t>("TCP_CLOSED");
    sent += static_cast<size_t>(r);
  }
  return make_ok(sent);
}

Expected<size_t> TcpSocket::recv_some(uint8_t* p, size_t n) {
  if (!valid()) return make_fail<size_t>("TCP_INVALID_SOCKET");
  if (n > (static_cast<size_t>(1) << 20)) n = static_cast<size_t>(1) << 20;
  while (true) {
#ifdef _WIN32
    int r = ::recv(static_cast<SOCKET>(h_), reinterpret_cast<char*>(p), static_cast<int>(n), 0);
#else
    ssize_t r = ::recv(h_, p, n, 0);
#endif
    if (r < 0) {
      int e = last_err();
      if (err_is_interrupt(e)) continue;
      if (err_is_timeout(e)) return make_fail<size_t>("TCP_TIMEOUT");
      return make_fail<size_t>("TCP_RECV_FAILED");
    }
    return make_ok(static_cast<size_t>(r));
  }
}

Expected<size_t> TcpSocket::recv_exact(uint8_t* p, size_t n) {
  size_t got = 0;
  while (got < n) {
    auto r = recv_some(p + got, n - got);
    if (!r.ok) return r;
    if (r.value == 0) return make_fail<size_t>("TCP_CLOSED");
    got += r.value;
  }
  return make_ok(got);
}

Expected<bool> TcpSocket::send_frame(const uint8_t* payload, size_t n) {
  std::vector<uint8_t> buf;
  buf.reserve(n + 8);
  for (int i = 7; i >= 0; --i) buf.push_back(static_cast<uint8_t>(static_cast<uint64_t>(n) >> (8 * i)));
  buf.insert(buf.end(), payload, payload + n);
  auto r = send_all(buf);
  if (!r.ok) return forward_fail<bool>(r);
  return make_ok(true);
}

Expected<std::vector<uint8_t>> TcpSocket::recv_frame(size_t max_len) {
  uint8_t hdr[8];
  auto r = recv_exact(hdr, 8);
  if (!r.ok) return forward_fail<std::vector<uint8_t>>(r);
  uint64_t len = 0;
  for (int i = 0; i < 8; ++i) len = (len << 8) | hdr[i];
  if (len > max_len) return make_fail<std::vector<uint8_t>>("FRAME_TOO_LARGE");
  std::vector<uint8_t> payload(static_cast<size_t>(len));
  if (len > 0) {
    auto r2 = recv_exact(payload.data(), payload.size());
    if (!r2.ok) return forward_fail<std::vector<uint8_t>>(r2);
  }
  return make_ok(std::move(payload));
}

void TcpSocket::shutdown_send() {
  if (!valid()) return;
#ifdef _WIN32
  ::shutdown(static_cast<SOCKET>(h_), SD_SEND);
#else
  ::shutdown(h_, SHUT_WR);
#endif
}

void TcpSocket::shutdown_both() {
  if (!valid()) return;
#ifdef _WIN32
  ::shutdown(static_cast<SOCKET>(h_), SD_BOTH);
#else
  ::shutdown(h_, SHUT_RDWR);
#endif
}

void TcpSocket::close() {
  if (valid()) {
    close_handle(h_);
    h_ = kInvalidSocket;
  }
}

std::string TcpSocket::peer() const {
  if (!valid()) return "";
  sockaddr_storage ss;
  sock_len_t len = sizeof ss;
  if (getpeername(static_cast<
#ifdef _WIN32
                      SOCKET
#else
                      int
#endif
                      >(h_),
                  reinterpret_cast<sockaddr*>(&ss), &len) != 0)
    return "";
  return addr_to_string(reinterpret_cast<sockaddr*>(&ss), len);
}

Expected<TcpSocket> TcpSocket::connect(const std::string& host, uint16_t port, double timeout_s, int retries,
                                       double retry_delay_s) {
  auto init = tcp_global_init();
  if (!init.ok) return forward_fail<TcpSocket>(init);
  if (retries < 1) retries = 1;
  std::string last_code = "TCP_CONNECT_FAILED";
  for (int attempt = 0; attempt < retries; ++attempt) {
    addrinfo hints;
    std::memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    std::string port_s = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) != 0 || res == nullptr) {
      last_code = "TCP_RESOLVE_FAILED";
      sleep_s(retry_delay_s);
      continue;
    }
    for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
      socket_handle_t h = static_cast<socket_handle_t>(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
      if (h == kInvalidSocket) continue;
      bool connected = false;
#ifdef _WIN32
      fast_refusal_on_loopback(h, ai->ai_addr);
#endif
      if (timeout_s > 0 && set_nonblocking(h, true)) {
#ifdef _WIN32
        int r = ::connect(static_cast<SOCKET>(h), ai->ai_addr, static_cast<int>(ai->ai_addrlen));
        int ce = r != 0 ? last_err() : 0;
        bool in_progress = (r != 0) && (ce == WSAEWOULDBLOCK);
#else
        int r = ::connect(h, ai->ai_addr, ai->ai_addrlen);
        int ce = r != 0 ? last_err() : 0;
        bool in_progress = (r != 0) && (ce == EINPROGRESS);
#endif
        if (r == 0) {
          connected = true;
        } else if (in_progress) {
          int err = 0;
          int w = wait_connect(h, timeout_s, err);
          if (w > 0) {
            connected = true;
          } else if (w == 0) {
            last_code = "TCP_TIMEOUT";
          } else {
            last_code = connect_error_code(err);
          }
        } else {
          last_code = connect_error_code(ce);
        }
        set_nonblocking(h, false);
      } else {
#ifdef _WIN32
        connected = ::connect(static_cast<SOCKET>(h), ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0;
#else
        connected = ::connect(h, ai->ai_addr, ai->ai_addrlen) == 0;
#endif
        if (!connected) last_code = connect_error_code(last_err());
      }
      if (connected) {
        freeaddrinfo(res);
        TcpSocket s(h);
        s.set_nodelay(true);
        return make_ok(std::move(s));
      }
      close_handle(h);
    }
    freeaddrinfo(res);
    if (attempt + 1 < retries) sleep_s(retry_delay_s);
  }
  return make_fail<TcpSocket>(last_code);
}

// ---------------------------------------------------------------------------
// TcpListener
// ---------------------------------------------------------------------------
TcpListener::~TcpListener() { close(); }

TcpListener::TcpListener(TcpListener&& o) noexcept : h_(o.h_), port_(o.port_), host_(std::move(o.host_)) {
  o.h_ = kInvalidSocket;
}

TcpListener& TcpListener::operator=(TcpListener&& o) noexcept {
  if (this != &o) {
    close();
    h_ = o.h_;
    port_ = o.port_;
    host_ = std::move(o.host_);
    o.h_ = kInvalidSocket;
  }
  return *this;
}

Expected<TcpListener> TcpListener::bind(const std::string& host, uint16_t port, int backlog) {
  auto init = tcp_global_init();
  if (!init.ok) return forward_fail<TcpListener>(init);
  addrinfo hints;
  std::memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* res = nullptr;
  std::string port_s = std::to_string(port);
  if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) != 0 || res == nullptr) {
    return make_fail<TcpListener>("TCP_RESOLVE_FAILED");
  }
  std::string code = "TCP_BIND_FAILED";
  for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
    socket_handle_t h = static_cast<socket_handle_t>(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
    if (h == kInvalidSocket) continue;
#ifndef _WIN32
    int one = 1;
    setsockopt(h, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
#ifdef _WIN32
    int br = ::bind(static_cast<SOCKET>(h), ai->ai_addr, static_cast<int>(ai->ai_addrlen));
#else
    int br = ::bind(h, ai->ai_addr, ai->ai_addrlen);
#endif
    if (br != 0) {
      close_handle(h);
      continue;
    }
#ifdef _WIN32
    if (::listen(static_cast<SOCKET>(h), backlog) != 0) {
#else
    if (::listen(h, backlog) != 0) {
#endif
      code = "TCP_LISTEN_FAILED";
      close_handle(h);
      continue;
    }
    sockaddr_storage ss;
    sock_len_t len = sizeof ss;
    uint16_t bound = port;
#ifdef _WIN32
    if (getsockname(static_cast<SOCKET>(h), reinterpret_cast<sockaddr*>(&ss), &len) == 0) {
#else
    if (getsockname(h, reinterpret_cast<sockaddr*>(&ss), &len) == 0) {
#endif
      if (ss.ss_family == AF_INET) bound = ntohs(reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
      if (ss.ss_family == AF_INET6) bound = ntohs(reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port);
    }
    freeaddrinfo(res);
    TcpListener l;
    l.h_ = h;
    l.port_ = bound;
    l.host_ = host;
    return make_ok(std::move(l));
  }
  freeaddrinfo(res);
  return make_fail<TcpListener>(code);
}

Expected<TcpSocket> TcpListener::accept(double timeout_s) {
  if (!valid()) return make_fail<TcpSocket>("TCP_INVALID_SOCKET");
  if (timeout_s > 0) {
    int w = wait_fd(h_, false, timeout_s);
    if (w == 0) return make_fail<TcpSocket>("TCP_TIMEOUT");
    if (w < 0) return make_fail<TcpSocket>("TCP_ACCEPT_FAILED");
  }
  sockaddr_storage ss;
  sock_len_t len = sizeof ss;
  while (true) {
#ifdef _WIN32
    SOCKET c = ::accept(static_cast<SOCKET>(h_), reinterpret_cast<sockaddr*>(&ss), &len);
    if (c == INVALID_SOCKET) {
#else
    int c = ::accept(h_, reinterpret_cast<sockaddr*>(&ss), &len);
    if (c < 0) {
#endif
      int e = last_err();
      if (err_is_interrupt(e)) continue;
      return make_fail<TcpSocket>("TCP_ACCEPT_FAILED");
    }
    TcpSocket s(static_cast<socket_handle_t>(c));
    s.set_nodelay(true);
    return make_ok(std::move(s));
  }
}

void TcpListener::close() {
  if (valid()) {
    close_handle(h_);
    h_ = kInvalidSocket;
  }
}

}  // namespace platform
}  // namespace opencrab2
