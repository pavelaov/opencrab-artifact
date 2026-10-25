// OpenCrab v2 browser client: talks only to the local sidecar (same origin), no third-party code.
// The bearer token arrives once in the URL fragment (#token=...), is kept in sessionStorage and sent as
// an Authorization header; the fragment is removed from the address bar so it is never logged by proxies.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { token: null, session: null, since: 0, reader: null, finished: false };

  function loadToken() {
    const m = /token=([0-9a-f]+)/.exec(location.hash || "");
    if (m) {
      sessionStorage.setItem("oc2_token", m[1]);
      history.replaceState(null, "", location.pathname);
    }
    state.token = sessionStorage.getItem("oc2_token");
    $("b-token").textContent = state.token ? "present (" + state.token.slice(0, 6) + "...)" : "missing: open the URL printed by the sidecar";
    $("b-sidecar").textContent = location.origin;
  }

  function headers(json) {
    const h = { "Authorization": "Bearer " + (state.token || "") };
    if (json) h["Content-Type"] = "application/json";
    return h;
  }

  async function api(method, path, body) {
    const res = await fetch(path, { method: method, headers: headers(body !== undefined),
                                    body: body === undefined ? undefined : JSON.stringify(body) });
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const code = data && data.code ? data.code : ("HTTP_" + res.status);
      throw new Error(code);
    }
    return data;
  }

  function setState(text, bad) {
    const el = $("state");
    el.textContent = text;
    el.className = bad ? "bad" : "muted";
  }

  function log(line, cls) {
    const div = document.createElement("div");
    div.textContent = line;
    if (cls) div.className = cls;
    $("log").appendChild(div);
    $("log").scrollTop = $("log").scrollHeight;
  }

  function showBinding(b) {
    if (!b) return;
    $("b-server").textContent = b.server || "";
    $("b-model").textContent = b.model_version || "";
    $("b-contract").textContent = b.numeric_contract_hash || "";
    $("b-profile").textContent = b.profile_id || "";
    $("b-dealer").textContent = b.dealer_mode || "";
    $("b-session").textContent = b.session_id || "";
  }

  async function refreshStatus() {
    try {
      const st = await api("GET", "/v1/status");
      $("b-server").textContent = st.server;
      $("b-model").textContent = st.model_version || "(learned from the dealer at session start)";
      $("b-contract").textContent = st.numeric_contract_hash;
      $("b-profile").textContent = st.profile_id;
    } catch (e) {
      setState("status unavailable: " + e.message, true);
    }
  }

  function handleEvent(type, ev) {
    if (type === "token") {
      $("out").textContent += ev.text !== undefined && ev.text !== "" ? ev.text : ("[" + ev.token_id + "]");
    } else if (type === "epoch") {
      log("epoch " + ev.epoch + " " + ev.gate + " plan=" + JSON.stringify(ev.plan) + (ev.switched ? " SWITCH" : "") +
          " exec=" + Number(ev.exec_s).toFixed(3) + "s bytes=" + ev.bytes + " rounds=" + ev.rounds +
          (ev.installed_now && ev.installed_now.length ? " installed=" + ev.installed_now.join(",") : ""));
    } else if (type === "status") {
      if (ev.binding) showBinding(ev.binding);
      log("status " + (ev.event || "") + (ev.epoch !== undefined ? " epoch=" + ev.epoch : ""));
      if (ev.event === "cancelled") { setState("cancelled"); $("resume").disabled = false; $("cancel").disabled = true; }
    } else if (type === "done") {
      state.finished = true;
      setState(ev.status + " (" + ev.tokens + " tokens, T_cold " + Number(ev.T_cold_s).toFixed(2) + " s, eligible=" + ev.security_eligible + ")");
      log("done " + JSON.stringify(ev));
      $("cancel").disabled = true;
      $("resume").disabled = !(ev.status === "cancelled");
      $("go").disabled = false;
    } else if (type === "error") {
      setState("error " + ev.code, true);
      log("error " + ev.code, "bad");
      $("cancel").disabled = true;
      $("resume").disabled = !ev.recoverable;
      $("go").disabled = false;
    }
  }

  async function stream() {
    if (!state.session) return;
    const res = await fetch("/v1/stream?session_id=" + encodeURIComponent(state.session) + "&since=" + state.since,
                            { headers: headers(false) });
    if (!res.ok) { setState("stream refused " + res.status, true); return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let type = "message", data = null, id = null;
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) type = line.slice(7).trim();
          else if (line.startsWith("data: ")) data = line.slice(6);
          else if (line.startsWith("id: ")) id = parseInt(line.slice(4), 10);
        }
        if (id !== null && !isNaN(id)) state.since = id + 1;
        if (type === "end") return;
        if (data !== null) {
          let ev = {};
          try { ev = JSON.parse(data); } catch (e) { ev = {}; }
          handleEvent(type, ev);
        }
      }
    }
  }

  $("go").addEventListener("click", async () => {
    const prompt = $("prompt").value;
    if (!prompt) return;
    $("out").textContent = "";
    $("log").textContent = "";
    state.since = 0;
    state.finished = false;
    try {
      const r = await api("POST", "/v1/generate", { prompt: prompt, n_generate: parseInt($("n").value, 10) || 8,
                                                     policy: $("policy").value });
      state.session = r.session_id;
      $("b-session").textContent = r.session_id;
      setState("running");
      $("go").disabled = true;
      $("cancel").disabled = false;
      $("resume").disabled = true;
      await stream();
    } catch (e) {
      setState("request refused: " + e.message, true);
    }
  });

  $("cancel").addEventListener("click", async () => {
    if (!state.session) return;
    try { await api("POST", "/v1/cancel", { session_id: state.session }); setState("cancel requested"); }
    catch (e) { setState("cancel refused: " + e.message, true); }
  });

  $("resume").addEventListener("click", async () => {
    if (!state.session) return;
    try {
      await api("POST", "/v1/resume", { session_id: state.session });
      setState("resuming");
      $("resume").disabled = true;
      $("cancel").disabled = false;
      $("go").disabled = true;
      await stream();
    } catch (e) { setState("resume refused: " + e.message, true); }
  });

  loadToken();
  refreshStatus();
})();
