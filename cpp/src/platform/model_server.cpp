// opencrab2/platform/model_server.cpp (see model_server.hpp)
#include <opencrab2/platform/model_server.hpp>

#include <chrono>
#include <cstdio>

namespace opencrab2 {
namespace platform {

json::Value SessionBinding::to_json() const {
  json::Value o = json::Value::object();
  o.set("session_id", json::Value::string(session_id));
  o.set("model_version", json::Value::string(model_version));
  o.set("numeric_contract_hash", json::Value::string(numeric_contract_hash));
  o.set("profile_id", json::Value::string(profile_id));
  o.set("nonce", json::Value::string(detail::hex_encode(nonce)));
  o.set("length_bucket", json::Value::integer(length_bucket));
  return o;
}

// ---------------------------------------------------------------------------
// RetransmitLog
// ---------------------------------------------------------------------------
Expected<const std::vector<uint8_t>*> RetransmitLog::lookup(const std::string& gate_id, int64_t seq,
                                                            const std::string& request_digest) const {
  std::string key = gate_id + "#" + std::to_string(seq);
  auto it = entries_.find(key);
  if (it == entries_.end()) return make_ok<const std::vector<uint8_t>*>(nullptr);
  if (!detail::constant_time_equal(it->second.digest, request_digest))
    return make_fail<const std::vector<uint8_t>*>("SEQ_REUSED_DIFFERENT_INPUT");
  return make_ok<const std::vector<uint8_t>*>(&it->second.reply);
}

void RetransmitLog::record(const std::string& gate_id, int64_t seq, const std::string& request_digest,
                           std::vector<uint8_t> reply) {
  std::string key = gate_id + "#" + std::to_string(seq);
  while (order_.size() >= max_entries_ && !order_.empty()) {
    auto old = entries_.find(order_.front());
    if (old != entries_.end()) {
      bytes_ -= old->second.reply.size();
      entries_.erase(old);
    }
    order_.pop_front();
  }
  bytes_ += reply.size();
  entries_[key] = Entry{request_digest, std::move(reply)};
  order_.push_back(key);
  auto ls = last_seq_.find(gate_id);
  if (ls == last_seq_.end() || ls->second < seq) last_seq_[gate_id] = seq;
}

std::map<std::string, int64_t> RetransmitLog::last_seq() const { return last_seq_; }

// ---------------------------------------------------------------------------
// ModelServer
// ---------------------------------------------------------------------------
struct ModelServer::Connection {
  WebSocket ws;
  std::string peer;
  std::shared_ptr<Session> session;
  std::deque<std::vector<uint8_t>> inbound;
  std::mutex mu;
  std::condition_variable cv;
  bool reader_done = false;
  bool paused = false;
  std::atomic<bool> closed{false};
  std::string close_code;
  Connection(WebSocket&& w, std::string p) : ws(std::move(w)), peer(std::move(p)) {}
};

ModelServer::ModelServer(ModelServerConfig cfg, ProtocolHandler handler) : cfg_(std::move(cfg)), handler_(std::move(handler)) {
  if (!cfg_.log) {
    cfg_.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
  }
}

ModelServer::~ModelServer() { stop(); }

uint16_t ModelServer::port() const { return http_ ? http_->port() : cfg_.port; }

void ModelServer::log(const std::string& event, const std::map<std::string, json::Value>& fields) {
  json::Value o = json::Value::object();
  o.set("ts", json::Value::number(wall_s()));
  o.set("component", json::Value::string("model_server"));
  o.set("event", json::Value::string(event));
  for (const auto& kv : fields) o.set(kv.first, kv.second);
  cfg_.log(o.dump(false));
}

Expected<bool> ModelServer::start() {
  auto l = TcpListener::bind(cfg_.host, cfg_.port, 32);
  if (!l.ok) return forward_fail<bool>(l);
  http_.reset(new HttpServer(std::move(l.value), [this](HttpServer::Exchange& ex) { on_http(ex); }));
  http_->set_request_timeout(15.0);
  auto s = http_->start();
  if (!s.ok) return s;
  stopping_ = false;
  housekeeping_ = std::thread([this]() { housekeeping_loop(); });
  log("start", {{"host", json::Value::string(cfg_.host)}, {"port", json::Value::integer(http_->port())},
                {"model_version", json::Value::string(cfg_.model_version)},
                {"ttl_s", json::Value::number(cfg_.session_ttl_s)}});
  return make_ok(true);
}

void ModelServer::stop() {
  if (!http_) return;
  stopping_ = true;
  if (housekeeping_.joinable()) housekeeping_.join();
  http_->stop();
  http_.reset();
}

ModelServer::Stats ModelServer::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

size_t ModelServer::live_sessions() const {
  std::lock_guard<std::mutex> lk(mu_);
  return sessions_.size();
}

json::Value ModelServer::stats_json() const {
  Stats s = stats();
  json::Value o = json::Value::object();
  auto put = [&](const char* k, uint64_t v) { o.set(k, json::Value::integer(static_cast<int64_t>(v))); };
  put("connections", s.connections);
  put("sessions_opened", s.sessions_opened);
  put("sessions_resumed", s.sessions_resumed);
  put("sessions_expired", s.sessions_expired);
  put("rejected_version", s.rejected_version);
  put("rejected_contract", s.rejected_contract);
  put("rejected_profile", s.rejected_profile);
  put("rejected_expired", s.rejected_expired);
  put("rejected_duplicate_session", s.rejected_duplicate_session);
  put("rejected_origin", s.rejected_origin);
  put("rejected_unknown_session", s.rejected_unknown_session);
  put("rejected_nonce", s.rejected_nonce);
  put("rejected_length_bucket", s.rejected_length_bucket);
  put("duplicate_seq_retransmits", s.duplicate_seq_retransmits);
  put("seq_violations", s.seq_violations);
  put("plaintext_rejections", s.plaintext_rejections);
  put("handler_rejections", s.handler_rejections);
  put("flow_pause_frames", s.flow_pause_frames);
  put("flow_resume_frames", s.flow_resume_frames);
  put("flow_violations", s.flow_violations);
  put("frames_in", s.frames_in);
  put("frames_out", s.frames_out);
  put("bytes_in", s.bytes_in);
  put("bytes_out", s.bytes_out);
  put("cancels", s.cancels);
  put("bad_frames", s.bad_frames);
  o.set("live_sessions", json::Value::integer(static_cast<int64_t>(live_sessions())));
  return o;
}

size_t ModelServer::expire_sessions(double now_wall_s) {
  std::lock_guard<std::mutex> lk(mu_);
  size_t n = 0;
  for (auto it = sessions_.begin(); it != sessions_.end();) {
    bool expired = false;
    {
      std::lock_guard<std::mutex> sl(it->second->mu);
      expired = !it->second->attached && (now_wall_s - it->second->last_active_wall_s) > cfg_.session_ttl_s;
    }
    if (expired) {
      it = sessions_.erase(it);
      ++n;
      stats_.sessions_expired++;
    } else {
      ++it;
    }
  }
  return n;
}

void ModelServer::housekeeping_loop() {
  double last = monotonic_s();
  while (!stopping_) {
    sleep_s(0.05);
    if (monotonic_s() - last >= 1.0) {
      last = monotonic_s();
      size_t n = expire_sessions(wall_s());
      if (n) log("sessions_expired", {{"count", json::Value::integer(static_cast<int64_t>(n))}});
    }
  }
}

void ModelServer::on_http(HttpServer::Exchange& ex) {
  if (ex.request.path == "/healthz") {
    ex.respond(HttpResponse::json(200, stats_json()));
    return;
  }
  if (ex.request.path != "/session") {
    ex.respond(HttpResponse::error(404, "NOT_FOUND"));
    return;
  }
  if (!cfg_.origin_allowlist.empty() && !origin_allowed(ex.request, cfg_.origin_allowlist, true)) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stats_.rejected_origin++;
    }
    ex.respond(HttpResponse::error(403, "ORIGIN_REJECTED"));
    return;
  }
  if (!is_websocket_upgrade(ex.request)) {
    ex.respond(HttpResponse::error(426, "UPGRADE_REQUIRED"));
    return;
  }
  auto ws = websocket_accept(std::move(ex.socket), ex.request);
  // websocket_accept took the handle; mark hijacked either way so the server
  // does not write a second response.
  ex.hijacked = true;
  ex.responded = true;
  if (!ws.ok) {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.bad_frames++;
    return;
  }
  ws.value.set_max_message_bytes(cfg_.max_frame_bytes);
  {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.connections++;
  }
  auto conn = std::make_shared<Connection>(std::move(ws.value), ex.peer);
  run_connection(conn);
}

Expected<bool> ModelServer::send_frame(Connection& conn, const wire::Value& v) {
  std::vector<uint8_t> f = wire::frame(v);
  auto r = conn.ws.send_binary(f);
  std::lock_guard<std::mutex> lk(mu_);
  if (r.ok) {
    stats_.frames_out++;
    stats_.bytes_out += f.size();
  }
  return r;
}

void ModelServer::reject(Connection& conn, const std::string& code, bool close) {
  wire::Value r = wire::Value::make_dict();
  r.set("type", wire::Value::from_str("REJECT"));
  r.set("code", wire::Value::from_str(code));
  send_frame(conn, r);
  log("reject", {{"code", json::Value::string(code)}, {"peer", json::Value::string(conn.peer)}});
  if (close) {
    conn.ws.send_close(1008, code);
    conn.closed = true;
    conn.close_code = code;
  }
}

Expected<std::shared_ptr<Session>> ModelServer::handle_hello(Connection& conn, const wire::Value& v) {
  using R = std::shared_ptr<Session>;
  (void)conn;
  SessionBinding b;
  b.session_id = v.str_or("session_id", "");
  b.model_version = v.str_or("model_version", "");
  b.numeric_contract_hash = v.str_or("numeric_contract_hash", "");
  b.profile_id = v.str_or("profile_id", "");
  b.length_bucket = v.int_or("length_bucket", 0);
  const wire::Value* nonce = v.get("nonce");
  if (nonce && nonce->is_bytes()) b.nonce = nonce->bytes;
  if (nonce && nonce->is_str()) b.nonce.assign(nonce->s.begin(), nonce->s.end());
  if (b.session_id.empty() || b.session_id.size() > 128 || b.nonce.empty()) return make_fail<R>("BAD_HELLO");
  std::lock_guard<std::mutex> lk(mu_);
  if (b.model_version != cfg_.model_version) {
    stats_.rejected_version++;
    return make_fail<R>("VERSION_MISMATCH");
  }
  if (b.numeric_contract_hash != cfg_.numeric_contract_hash) {
    stats_.rejected_contract++;
    return make_fail<R>("CONTRACT_MISMATCH");
  }
  if (b.profile_id != cfg_.profile_id) {
    stats_.rejected_profile++;
    return make_fail<R>("PROFILE_MISMATCH");
  }
  if (b.length_bucket <= 0 || b.length_bucket > (1 << 20)) {
    stats_.rejected_length_bucket++;
    return make_fail<R>("BAD_LENGTH_BUCKET");
  }
  auto it = sessions_.find(b.session_id);
  if (it != sessions_.end()) {
    std::lock_guard<std::mutex> sl(it->second->mu);
    bool expired = (wall_s() - it->second->last_active_wall_s) > cfg_.session_ttl_s && !it->second->attached;
    if (!expired) {
      stats_.rejected_duplicate_session++;
      return make_fail<R>("DUPLICATE_SESSION");
    }
    sessions_.erase(it);
    stats_.sessions_expired++;
  }
  auto s = std::make_shared<Session>(cfg_.retransmit_log_entries);
  s->binding = b;
  s->created_wall_s = wall_s();
  s->last_active_wall_s = s->created_wall_s;
  s->attached = true;
  sessions_[b.session_id] = s;
  stats_.sessions_opened++;
  return make_ok(s);
}

Expected<std::shared_ptr<Session>> ModelServer::handle_resume(Connection& conn, const wire::Value& v) {
  using R = std::shared_ptr<Session>;
  (void)conn;
  std::string sid = v.str_or("session_id", "");
  const wire::Value* nonce = v.get("nonce");
  std::vector<uint8_t> nb;
  if (nonce && nonce->is_bytes()) nb = nonce->bytes;
  if (nonce && nonce->is_str()) nb.assign(nonce->s.begin(), nonce->s.end());
  std::lock_guard<std::mutex> lk(mu_);
  auto it = sessions_.find(sid);
  if (it == sessions_.end()) {
    stats_.rejected_unknown_session++;
    return make_fail<R>("SESSION_UNKNOWN");
  }
  std::lock_guard<std::mutex> sl(it->second->mu);
  if ((wall_s() - it->second->last_active_wall_s) > cfg_.session_ttl_s) {
    stats_.rejected_expired++;
    sessions_.erase(it);
    stats_.sessions_expired++;
    return make_fail<R>("SESSION_EXPIRED");
  }
  if (it->second->attached) {
    stats_.rejected_duplicate_session++;
    return make_fail<R>("DUPLICATE_SESSION");
  }
  if (nb.size() != it->second->binding.nonce.size() ||
      !detail::constant_time_equal(nb.data(), it->second->binding.nonce.data(), nb.size())) {
    stats_.rejected_nonce++;
    return make_fail<R>("NONCE_MISMATCH");
  }
  it->second->attached = true;
  it->second->last_active_wall_s = wall_s();
  stats_.sessions_resumed++;
  return make_ok(it->second);
}

namespace {
bool looks_like_plaintext(const wire::Value& payload) {
  if (payload.is_str()) return true;
  if (!payload.is_dict()) return false;
  static const char* banned[] = {"prompt", "text", "tokens", "plaintext", "input_ids"};
  for (const char* k : banned) {
    if (payload.has(k)) return true;
  }
  return false;
}
}  // namespace

void ModelServer::handle_msg(Connection& conn, Session& s, const wire::Value& v) {
  std::string gate_id = v.str_or("gate_id", "");
  int64_t seq = v.int_or("seq", -1);
  const wire::Value* payload = v.get("payload");
  if (gate_id.empty() || seq < 0 || !payload) {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.bad_frames++;
    reject(conn, "BAD_MSG", true);
    return;
  }
  if (looks_like_plaintext(*payload)) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stats_.plaintext_rejections++;
    }
    reject(conn, "PLAINTEXT_NOT_ACCEPTED", true);
    return;
  }
  std::vector<uint8_t> enc = wire::encode(*payload);
  auto dg = detail::sha256(enc);
  std::string digest = detail::hex_encode(dg.data(), dg.size());
  // Lock order everywhere: server mu_ before a session mu; never the reverse.
  std::vector<uint8_t> reply_bytes;
  bool seq_violation = false;
  bool replay = false;
  {
    std::lock_guard<std::mutex> sl(s.mu);
    s.last_active_wall_s = wall_s();
    s.frames_in++;
    s.bytes_in += enc.size();
    auto prior = s.log.lookup(gate_id, seq, digest);
    if (!prior.ok) {
      seq_violation = true;
    } else if (prior.value != nullptr) {
      reply_bytes = *prior.value;
      s.replays++;
      replay = true;
    }
  }
  if (seq_violation) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stats_.seq_violations++;
    }
    reject(conn, "SEQ_REUSED_DIFFERENT_INPUT", true);
    return;
  }
  if (replay) {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.duplicate_seq_retransmits++;
  }
  if (reply_bytes.empty()) {
    auto out = handler_(s, gate_id, seq, *payload);
    if (!out.ok) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        stats_.handler_rejections++;
      }
      reject(conn, out.code, false);
      return;
    }
    wire::Value reply = wire::Value::make_dict();
    reply.set("type", wire::Value::from_str("MSG"));
    reply.set("gate_id", wire::Value::from_str(gate_id));
    reply.set("seq", wire::Value::from_int(seq));
    reply.set("payload", std::move(out.value));
    reply_bytes = wire::frame(reply);
    std::lock_guard<std::mutex> sl(s.mu);
    s.log.record(gate_id, seq, digest, reply_bytes);
  }
  auto r = conn.ws.send_binary(reply_bytes);
  {
    std::lock_guard<std::mutex> sl(s.mu);
    s.frames_out++;
    s.bytes_out += reply_bytes.size();
  }
  std::lock_guard<std::mutex> lk(mu_);
  if (r.ok) {
    stats_.frames_out++;
    stats_.bytes_out += reply_bytes.size();
  }
  log("msg", {{"session_id", json::Value::string(s.binding.session_id)},
              {"gate_id", json::Value::string(gate_id)},
              {"seq", json::Value::integer(seq)},
              {"bytes_in", json::Value::integer(static_cast<int64_t>(enc.size()))},
              {"bytes_out", json::Value::integer(static_cast<int64_t>(reply_bytes.size()))}});
}

void ModelServer::reader_loop(std::shared_ptr<Connection> conn) {
  size_t high = cfg_.max_queue_frames * 3 / 4;
  if (high == 0) high = 1;
  size_t low = cfg_.max_queue_frames / 4;
  while (true) {
    auto m = conn->ws.recv_message();
    if (!m.ok || m.value.is_close()) {
      std::lock_guard<std::mutex> lk(conn->mu);
      conn->reader_done = true;
      conn->cv.notify_all();
      return;
    }
    if (!m.value.is_binary()) continue;
    {
      std::lock_guard<std::mutex> glk(mu_);
      stats_.frames_in++;
      stats_.bytes_in += m.value.payload.size();
    }
    bool violation = false;
    bool send_pause = false;
    {
      std::lock_guard<std::mutex> lk(conn->mu);
      if (conn->inbound.size() >= cfg_.max_queue_frames) {
        violation = true;
      } else {
        conn->inbound.push_back(std::move(m.value.payload));
        if (!conn->paused && conn->inbound.size() >= high) {
          conn->paused = true;
          send_pause = true;
        }
      }
      conn->cv.notify_all();
    }
    if (violation) {
      {
        std::lock_guard<std::mutex> glk(mu_);
        stats_.flow_violations++;
      }
      reject(*conn, "FLOW_VIOLATION", true);
      std::lock_guard<std::mutex> lk(conn->mu);
      conn->reader_done = true;
      conn->cv.notify_all();
      return;
    }
    if (send_pause) {
      wire::Value f = wire::Value::make_dict();
      f.set("type", wire::Value::from_str("FLOW"));
      f.set("pause", wire::Value::from_bool(true));
      f.set("queue_depth", wire::Value::from_int(static_cast<int64_t>(high)));
      send_frame(*conn, f);
      std::lock_guard<std::mutex> glk(mu_);
      stats_.flow_pause_frames++;
    }
    (void)low;
  }
}

void ModelServer::run_connection(std::shared_ptr<Connection> conn) {
  log("connection", {{"peer", json::Value::string(conn->peer)}});
  std::thread reader([this, conn]() { reader_loop(conn); });
  size_t low = cfg_.max_queue_frames / 4;
  double idle_deadline = wall_s() + cfg_.session_ttl_s;
  while (true) {
    std::vector<uint8_t> frame;
    bool send_resume = false;
    {
      std::unique_lock<std::mutex> lk(conn->mu);
      while (conn->inbound.empty() && !conn->reader_done && !conn->closed && !stopping_) {
        conn->cv.wait_for(lk, std::chrono::milliseconds(200));
        if (wall_s() > idle_deadline) break;
      }
      if (conn->closed || stopping_) break;
      if (conn->inbound.empty()) {
        if (conn->reader_done) break;
        if (wall_s() > idle_deadline) {
          lk.unlock();
          reject(*conn, "SESSION_EXPIRED", true);
          break;
        }
        continue;
      }
      frame = std::move(conn->inbound.front());
      conn->inbound.pop_front();
      if (conn->paused && conn->inbound.size() <= low) {
        conn->paused = false;
        send_resume = true;
      }
    }
    if (send_resume) {
      wire::Value f = wire::Value::make_dict();
      f.set("type", wire::Value::from_str("FLOW"));
      f.set("pause", wire::Value::from_bool(false));
      f.set("queue_depth", wire::Value::from_int(static_cast<int64_t>(low)));
      send_frame(*conn, f);
      std::lock_guard<std::mutex> glk(mu_);
      stats_.flow_resume_frames++;
    }
    auto v = wire::decode_frame(frame);
    if (!v.ok || !v.value.is_dict()) {
      {
        std::lock_guard<std::mutex> glk(mu_);
        stats_.bad_frames++;
      }
      reject(*conn, "BAD_FRAME", true);
      break;
    }
    std::string type = v.value.str_or("type", "");
    idle_deadline = wall_s() + cfg_.session_ttl_s;
    if (type == "HELLO") {
      if (conn->session) {
        reject(*conn, "ALREADY_BOUND", true);
        break;
      }
      auto s = handle_hello(*conn, v.value);
      if (!s.ok) {
        reject(*conn, s.code, true);
        break;
      }
      conn->session = s.value;
      wire::Value ok = wire::Value::make_dict();
      ok.set("type", wire::Value::from_str("HELLO_OK"));
      ok.set("session_id", wire::Value::from_str(s.value->binding.session_id));
      ok.set("ttl_s", wire::Value::from_float(cfg_.session_ttl_s));
      ok.set("max_queue_frames", wire::Value::from_int(static_cast<int64_t>(cfg_.max_queue_frames)));
      ok.set("length_bucket", wire::Value::from_int(s.value->binding.length_bucket));
      send_frame(*conn, ok);
      log("hello", {{"session_id", json::Value::string(s.value->binding.session_id)},
                    {"length_bucket", json::Value::integer(s.value->binding.length_bucket)}});
      continue;
    }
    if (type == "RESUME") {
      if (conn->session) {
        reject(*conn, "ALREADY_BOUND", true);
        break;
      }
      auto s = handle_resume(*conn, v.value);
      if (!s.ok) {
        reject(*conn, s.code, true);
        break;
      }
      conn->session = s.value;
      wire::Value ok = wire::Value::make_dict();
      ok.set("type", wire::Value::from_str("RESUME_OK"));
      ok.set("session_id", wire::Value::from_str(s.value->binding.session_id));
      wire::Value last = wire::Value::make_dict();
      {
        std::lock_guard<std::mutex> sl(s.value->mu);
        for (const auto& kv : s.value->log.last_seq()) last.set(kv.first, wire::Value::from_int(kv.second));
      }
      ok.set("last_seq", last);
      send_frame(*conn, ok);
      log("resume", {{"session_id", json::Value::string(s.value->binding.session_id)}});
      continue;
    }
    if (type == "PING") {
      wire::Value p = wire::Value::make_dict();
      p.set("type", wire::Value::from_str("PONG"));
      send_frame(*conn, p);
      continue;
    }
    if (!conn->session) {
      reject(*conn, "NOT_BOUND", true);
      break;
    }
    if (type == "MSG") {
      handle_msg(*conn, *conn->session, v.value);
      if (conn->closed) break;
      continue;
    }
    if (type == "CANCEL") {
      conn->session->cancelled = true;
      {
        std::lock_guard<std::mutex> glk(mu_);
        stats_.cancels++;
      }
      wire::Value ok = wire::Value::make_dict();
      ok.set("type", wire::Value::from_str("CANCEL_OK"));
      send_frame(*conn, ok);
      log("cancel", {{"session_id", json::Value::string(conn->session->binding.session_id)}});
      continue;
    }
    reject(*conn, "UNKNOWN_FRAME_TYPE", true);
    break;
  }
  if (conn->session) {
    std::lock_guard<std::mutex> sl(conn->session->mu);
    conn->session->attached = false;
    conn->session->last_active_wall_s = wall_s();
  }
  // Graceful shutdown: half-close so that queued REJECT / close frames are
  // delivered, give the reader up to two seconds to see the peer's close
  // reply or EOF, then close. An immediate close with a pending receive can
  // become abortive on Windows and discard what the peer has not read yet.
  conn->ws.socket().shutdown_send();
  {
    std::unique_lock<std::mutex> lk(conn->mu);
    conn->cv.wait_for(lk, std::chrono::seconds(2), [&]() { return conn->reader_done; });
  }
  // A peer that stays connected leaves the reader blocked in recv(). Shutting
  // down both directions ends that recv() (SHUT_RDWR on POSIX), and the reader
  // is joined BEFORE the descriptor is closed: a recv() on a closed descriptor
  // could read from the next connection that reuses its number. Should a host
  // leave the reader blocked after the shutdown, closing the socket ends it.
  conn->ws.socket().shutdown_both();
  bool reader_finished = false;
  {
    std::unique_lock<std::mutex> lk(conn->mu);
    reader_finished = conn->cv.wait_for(lk, std::chrono::seconds(2), [&]() { return conn->reader_done; });
  }
  if (reader_finished && reader.joinable()) reader.join();
  conn->ws.close();
  if (reader.joinable()) reader.join();
  log("disconnect", {{"peer", json::Value::string(conn->peer)},
                     {"session_id", json::Value::string(conn->session ? conn->session->binding.session_id : "")}});
}

// ---------------------------------------------------------------------------
// Demo protocol
// ---------------------------------------------------------------------------
namespace demo {

Expected<BeaverHalf> parse_beaver_half(const Bundle& bundle, size_t n) {
  if (bundle.payload.size() != 3 * n * 8) return make_fail<BeaverHalf>("BUNDLE_SIZE_MISMATCH");
  auto parts = bundle.parts();
  if (parts.size() != 3 || parts[0] != "a" || parts[1] != "b" || parts[2] != "c")
    return make_fail<BeaverHalf>("BUNDLE_BAD_PARTS");
  BeaverHalf h;
  h.a = le_to_u64(bundle.payload.data(), n * 8);
  h.b = le_to_u64(bundle.payload.data() + n * 8, n * 8);
  h.c = le_to_u64(bundle.payload.data() + 2 * n * 8, n * 8);
  return make_ok(std::move(h));
}

std::vector<uint64_t> ring_sub(const std::vector<uint64_t>& x, const std::vector<uint64_t>& y) {
  std::vector<uint64_t> z(x.size());
  for (size_t i = 0; i < x.size(); ++i) z[i] = x[i] - y[i];
  return z;
}

std::vector<uint64_t> ring_add(const std::vector<uint64_t>& x, const std::vector<uint64_t>& y) {
  std::vector<uint64_t> z(x.size());
  for (size_t i = 0; i < x.size(); ++i) z[i] = x[i] + y[i];
  return z;
}

std::vector<uint64_t> beaver_combine(const BeaverHalf& h, const std::vector<uint64_t>& d, const std::vector<uint64_t>& e,
                                     int party) {
  std::vector<uint64_t> z(h.c.size());
  for (size_t i = 0; i < z.size(); ++i) {
    uint64_t v = h.c[i] + d[i] * h.b[i] + e[i] * h.a[i];
    if (party == 0) v += d[i] * e[i];
    z[i] = v;
  }
  return z;
}

Expected<std::vector<uint64_t>> array_field(const wire::Value& dict, const std::string& key, size_t n) {
  const wire::Value* v = dict.get(key);
  if (!v || !v->is_array() || v->array.dtype != wire::DType::U64) return make_fail<std::vector<uint64_t>>("DEMO_BAD_PAYLOAD");
  std::vector<uint64_t> out = v->array.to_u64();
  if (out.size() != n) return make_fail<std::vector<uint64_t>>("DEMO_SHAPE_MISMATCH");
  return make_ok(std::move(out));
}

}  // namespace demo

DemoBeaverP0::DemoBeaverP0(std::shared_ptr<DealerConnection> dealer, std::vector<uint64_t> w)
    : dealer_(std::move(dealer)), w_(std::move(w)), rng_(detail::ChaCha20::from_os()) {}

ProtocolHandler DemoBeaverP0::as_handler() {
  return [this](Session& s, const std::string& g, int64_t seq, const wire::Value& p) { return handle(s, g, seq, p); };
}

Expected<wire::Value> DemoBeaverP0::handle(Session& session, const std::string& gate_id, int64_t seq,
                                           const wire::Value& payload) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!session.handler_state) session.handler_state = std::make_shared<SessionState>();
  auto* st = static_cast<SessionState*>(session.handler_state.get());
  size_t n = w_.size();
  if (gate_id == "share_input") {
    auto x0 = demo::array_field(payload, "x0", n);
    if (!x0.ok) return forward_fail<wire::Value>(x0);
    if (!dealer_ || !dealer_->valid()) return make_fail<wire::Value>("DEALER_UNAVAILABLE");
    wire::Value params = wire::Value::make_dict();
    params.set("shape", wire::Value::from_list({wire::Value::from_int(static_cast<int64_t>(n))}));
    auto bundle = dealer_->request("beaver", params, 1, session.binding.session_id + ":" + std::to_string(seq));
    if (!bundle.ok) return forward_fail<wire::Value>(bundle);
    auto half = demo::parse_beaver_half(bundle.value, n);
    if (!half.ok) return forward_fail<wire::Value>(half);
    StepState& step = st->steps[seq];
    step.x0 = std::move(x0.value);
    step.triple = std::move(half.value);
    std::vector<uint64_t> w1(n);
    rng_.fill_u64(w1.data(), n);
    step.w0 = demo::ring_sub(w_, w1);
    step.have_input = true;
    wire::Value reply = wire::Value::make_dict();
    reply.set("w1", wire::Value::from_u64_vector(w1));
    return make_ok(std::move(reply));
  }
  if (gate_id == "beaver_open") {
    auto it = st->steps.find(seq);
    if (it == st->steps.end() || !it->second.have_input) return make_fail<wire::Value>("DEMO_NO_INPUT");
    StepState& step = it->second;
    auto d1 = demo::array_field(payload, "d1", n);
    if (!d1.ok) return forward_fail<wire::Value>(d1);
    auto e1 = demo::array_field(payload, "e1", n);
    if (!e1.ok) return forward_fail<wire::Value>(e1);
    std::vector<uint64_t> d0 = demo::ring_sub(step.x0, step.triple.a);
    std::vector<uint64_t> e0 = demo::ring_sub(step.w0, step.triple.b);
    std::vector<uint64_t> d = demo::ring_add(d0, d1.value);
    std::vector<uint64_t> e = demo::ring_add(e0, e1.value);
    std::vector<uint64_t> z0 = demo::beaver_combine(step.triple, d, e, 0);
    wire::Value reply = wire::Value::make_dict();
    reply.set("d0", wire::Value::from_u64_vector(d0));
    reply.set("e0", wire::Value::from_u64_vector(e0));
    reply.set("z0", wire::Value::from_u64_vector(z0));  // RELEASE_FINAL to P1
    st->steps.erase(it);                               // material is single use
    return make_ok(std::move(reply));
  }
  return make_fail<wire::Value>("UNKNOWN_GATE");
}

ProtocolHandler make_ack_handler() {
  return [](Session&, const std::string&, int64_t, const wire::Value& payload) {
    wire::Value r = wire::Value::make_dict();
    r.set("ack", wire::Value::from_bool(true));
    r.set("bytes", wire::Value::from_int(static_cast<int64_t>(wire::encode(payload).size())));
    return make_ok(std::move(r));
  };
}

}  // namespace platform
}  // namespace opencrab2
