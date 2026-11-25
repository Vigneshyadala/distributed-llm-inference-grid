/* Distributed Memory Grid: live dashboard.
 * Source of truth is the engine's event feed (/ws/events). When no server answers, the page replays a
 * recorded real run (assets/recorded.json) and says so in the badge. Nothing here is invented. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PALETTE = ["#ff3b5c", "#ffb088", "#ffb020", "#2dd4bf", "#8b5cf6", "#60a5fa", "#f472b6", "#a3e635"];
  const colorOf = (id) => PALETTE[((parseInt(String(id).replace(/\D/g, ""), 10) || 1) - 1) % PALETTE.length];
  const MAX_ROWS = 9, COLS = 36;

  $("#year").textContent = new Date().getFullYear();
  if (!reduce) {
    const ambient = $("#ambient");
    for (let i = 0; i < 20; i++) {
      const e = document.createElement("div"), size = 2 + Math.random() * 3;
      e.className = "ember";
      e.style.cssText = `width:${size}px;height:${size}px;left:${Math.random() * 100}%;bottom:-10px;` +
        `animation-duration:${11 + Math.random() * 14}s;animation-delay:${Math.random() * 14}s`;
      ambient.appendChild(e);
    }
  }

  // ------------------------------------------------------------------------------- state
  const S = { mode: "connecting", maxBatch: 8, total: 0, blockSize: 4, nextStep: 0, reqs: new Map(), order: [],
              blocks: [], rate: [], paused: false };

  function resetState() {
    S.nextStep = 0; S.reqs.clear(); S.order = []; S.rate = [];
    dirty();
  }
  function ensureReq(id, init) {
    let r = S.reqs.get(id);
    if (!r) {
      r = Object.assign({ id, cells: new Map(), arrival: S.nextStep, admitted: null, kind: null, tokens: 0 }, init || {});
      S.reqs.set(id, r); S.order.push(id);
      while (S.order.length > MAX_ROWS) {
        const drop = S.order.findIndex((k) => S.reqs.get(k).kind || S.reqs.get(k).rejected);
        const k = S.order.splice(drop < 0 ? 0 : drop, 1)[0];
        S.reqs.delete(k);
      }
    }
    return r;
  }

  // ------------------------------------------------------------------------------ badge
  const badge = $("#badge");
  function setMode(mode) {
    S.mode = mode;
    badge.className = "badge " + (mode === "live" ? "live" : mode === "replay" ? "" : "off");
    $("#badge-text").textContent = { live: "LIVE", replay: "RECORDED RUN", off: "OFFLINE", connecting: "CONNECTING" }[mode];
    const live = mode === "live";
    $("#send").disabled = !live; $("#burst").disabled = !live;
    $("#replay").textContent = mode === "replay" ? "● Back to live" : "↻ Replay recorded run";
    $("#mode-note").textContent = live ? "Connected to the engine. Send a request and watch it move through the lanes and the cache."
      : mode === "replay" ? "Replaying a real run recorded from the engine. Start the server (python scripts/serve.py) to go live."
      : mode === "off" ? "The engine is not reachable. Retrying..." : "";
  }

  // ------------------------------------------------------------------------------ terminal
  const term = $("#term");
  $("#pause").onclick = () => { S.paused = !S.paused; $("#pause").textContent = S.paused ? "resume" : "pause"; };
  function line(cls, text) {
    if (S.paused) return;
    const d = document.createElement("div");
    d.className = "ln " + cls; d.textContent = text;
    term.appendChild(d);
    while (term.childElementCount > 300) term.firstChild.remove();
    term.scrollTop = term.scrollHeight;
  }
  function describe(ev) {
    switch (ev.type) {
      case "snapshot": return ["sys", `connected: step ${ev.stats.step}, ${ev.stats.running} running, ${ev.stats.waiting} waiting, ${ev.stats.blocks_free}/${ev.stats.blocks_total} blocks free`];
      case "submitted": return ["sub-l", `+ ${ev.id} submitted (${ev.prompt_len} prompt tokens, up to ${ev.max_new_tokens} new)`];
      case "rejected": return ["rej", `x ${ev.id || "request"} rejected: ${ev.reason}`];
      case "step": return ["step", `step ${String(ev.step).padStart(4)}  prefill [${ev.prefill.join(",") || "-"}]  decode [${ev.decode.join(",") || "-"}]  running ${ev.running}  waiting ${ev.waiting}  blocks ${ev.blocks_used}/${ev.stats.blocks_total}  ${ev.ms.toFixed(1)} ms`];
      case "finished": return ["fin", `- ${ev.id} ${ev.kind} (${ev.reason}) after ${ev.tokens} tokens, waited ${ev.wait_steps} steps, latency ${ev.latency_steps} steps`];
      case "stopped": return ["sys", "the engine stopped"];
    }
    return null;
  }

  // ------------------------------------------------------------------------------ numbers
  function setNum(el, text) { if (el.textContent !== text) { el.textContent = text; if (!reduce) { el.classList.remove("tick"); void el.offsetWidth; el.classList.add("tick"); } } }
  function updateStats(st) {
    S.maxBatch = st.max_batch_size || S.maxBatch;
    const used = st.blocks_total - st.blocks_free;
    setNum($("#s-steps"), String(st.step));
    setNum($("#s-run"), String(st.running));
    setNum($("#s-mem"), (st.blocks_total ? Math.round(100 * used / st.blocks_total) : 0) + "%");
    $("#g-sched").textContent = `${st.running} running`; $("#g-wait").textContent = `${st.waiting} waiting`;
    $("#g-cache").textContent = `KV cache: blocks ${used} / ${st.blocks_total}`;
    $("#graphic").classList.toggle("hot", st.running > 0 || st.waiting > 0);
    $("#g-used").textContent = `${used} / ${st.blocks_total}`; $("#g-res").textContent = `${st.reserved_blocks} / ${st.blocks_total}`;
    $("#bar-used").style.width = (st.blocks_total ? 100 * used / st.blocks_total : 0) + "%";
    $("#bar-res").style.width = (st.blocks_total ? 100 * st.reserved_blocks / st.blocks_total : 0) + "%";
  }
  function updateRate() {
    const now = performance.now();
    S.rate = S.rate.filter((x) => now - x.t < 3000);
    const n = S.rate.reduce((a, x) => a + x.n, 0);
    const span = S.rate.length ? Math.max(1000, now - S.rate[0].t) : 1000;
    setNum($("#s-rate"), String(Math.round(1000 * n / span)));
  }
  setInterval(updateRate, 500);

  // ------------------------------------------------------------------------------ block map
  const blocksEl = $("#blocks");
  function buildBlocks(n) {
    blocksEl.textContent = ""; S.blocks = [];
    for (let k = 0; k < n; k++) {
      const b = document.createElement("div"), f = document.createElement("i");
      b.className = "blk"; b.appendChild(f); b.title = `block ${k}: free`;
      blocksEl.appendChild(b); S.blocks.push({ el: b, fill: f, key: "", owner: null });
    }
  }
  function renderBlocks(map) {
    if (!map) return;
    if (map.length !== S.blocks.length) buildBlocks(map.length);
    map.forEach((entry, k) => {
      const cell = S.blocks[k], key = entry ? entry[0] + ":" + entry[1] : "";
      if (key === cell.key) return;
      const owner = entry ? entry[0] : null;
      cell.key = key;
      if (!entry) { cell.el.classList.remove("used", "flash"); cell.fill.style.height = "0"; cell.el.title = `block ${k}: free`; cell.owner = null; return; }
      const color = colorOf(owner);
      cell.el.style.setProperty("--c", color); cell.el.classList.add("used");
      cell.fill.style.height = Math.min(100, 100 * entry[1] / S.blockSize) + "%";
      cell.el.title = `block ${k}: ${owner}, ${entry[1]} of ${S.blockSize} tokens`;
      if (owner !== cell.owner && !reduce) { cell.el.classList.remove("flash"); void cell.el.offsetWidth; cell.el.classList.add("flash"); }
      cell.owner = owner;
    });
  }

  // ------------------------------------------------------------------------------ swimlanes
  const canvas = $("#lanes"), g = canvas.getContext("2d");
  let needDraw = true;
  function dirty() { needDraw = true; }
  function draw() {
    requestAnimationFrame(draw);
    if (!needDraw) return; needDraw = false;
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const labelW = 62, top = 8, bottom = 22, rowH = Math.min(30, (H - top - bottom) / MAX_ROWS);
    const last = S.nextStep - 1, first = Math.max(0, last - COLS + 1), cw = (W - labelW - 6) / COLS;
    const x = (step) => labelW + (step - first) * cw;
    g.font = "11px 'JetBrains Mono', monospace"; g.textBaseline = "middle";
    g.strokeStyle = "rgba(255,178,150,.08)"; g.fillStyle = "#8a6670";
    for (let s = first; s <= first + COLS; s++) if (s % 5 === 0) {
      g.beginPath(); g.moveTo(x(s), top); g.lineTo(x(s), H - bottom); g.stroke();
      g.fillText(String(s), x(s) - 6, H - 8);
    }
    S.order.forEach((id, i) => {
      const r = S.reqs.get(id), y = top + i * rowH, mid = y + rowH / 2, color = colorOf(id);
      g.fillStyle = color; g.fillText(id, 4, mid);
      const endWait = r.admitted !== null ? r.admitted : (r.rejected ? r.arrival : last + 1);
      if (!r.rejected && endWait > r.arrival) {
        g.strokeStyle = color; g.globalAlpha = .55; g.setLineDash([2, 4]); g.beginPath();
        g.moveTo(Math.max(x(r.arrival), labelW) + 2, mid); g.lineTo(x(Math.min(endWait, last + 1)) - 2, mid); g.stroke();
        g.setLineDash([]); g.globalAlpha = 1;
      }
      r.cells.forEach((kind, step) => {
        if (step < first) return;
        const cx = x(step) + 1, cwid = cw - 2, ch = rowH - 6;
        g.fillStyle = color; g.globalAlpha = kind === "P" ? 1 : .55;
        if (kind === "P" && !reduce) { g.shadowColor = color; g.shadowBlur = 10; }
        g.beginPath(); g.roundRect(cx, y + 3, cwid, ch, 4); g.fill(); g.shadowBlur = 0; g.globalAlpha = 1;
        if (cwid > 9) { g.fillStyle = "#1a0208"; g.fillText(kind, cx + cwid / 2 - 3, mid + 1); }
      });
      if (r.rejected) { g.fillStyle = "#ffb020"; g.fillText("✕ refused: could never fit", x(Math.max(r.arrival, first)) + 4, mid); }
      else if (r.kind) {
        const end = r.endStep !== undefined ? r.endStep : last;
        if (end >= first) { g.fillStyle = r.kind === "done" ? "#2dd4bf" : "#ffb020"; g.fillText(r.kind === "done" ? "✓" : "–", x(end + 1) + 2, mid); }
      }
    });
    if (!S.order.length) { g.fillStyle = "#8a6670"; g.fillText("waiting for requests...", labelW, H / 2); }
  }
  addEventListener("resize", dirty);
  requestAnimationFrame(draw);
  function renderLegend() {
    $("#lane-legend").innerHTML = "";
    S.order.forEach((id) => {
      const r = S.reqs.get(id), s = document.createElement("span"), i = document.createElement("i");
      i.style.background = colorOf(id);
      s.append(i, `${id} ${r.rejected ? "refused" : r.kind ? r.kind + " (" + r.tokens + " tok)" : r.admitted === null ? "waiting" : "running"}`);
      $("#lane-legend").appendChild(s);
    });
  }

  // ------------------------------------------------------------------------------ events
  function applyStep(rec) {
    rec.prefill.forEach((id) => { const r = ensureReq(id, { arrival: rec.step }); r.cells.set(rec.step, "P"); if (r.admitted === null) r.admitted = rec.step; });
    rec.decode.forEach((id) => {
      const r = ensureReq(id, { arrival: rec.step }); r.cells.set(rec.step, "d");
      if (r.admitted === null) r.admitted = rec.step;      // joined the page mid-run: it started earlier
    });
    S.nextStep = rec.step + 1;
  }
  // Opening the page after a request already finished: its "finished" event was missed, so infer it.
  function markEndedBefore(steps) {
    if (!steps.length) return;
    const lastRec = steps[steps.length - 1], ended = new Set(), active = new Set(lastRec.prefill.concat(lastRec.decode));
    steps.forEach((rec) => rec.finished.forEach((id) => ended.add(id)));
    S.reqs.forEach((r) => {
      if ((ended.has(r.id) || !active.has(r.id)) && !r.kind) {
        r.kind = "done"; r.endStep = Math.max(...r.cells.keys()); r.tokens = r.cells.size;
      }
    });
  }
  function beat() { const g = $("#graphic"); g.classList.remove("beat"); void g.offsetWidth; g.classList.add("beat"); }
  document.addEventListener("pointermove", (e) => {
    const t = e.target.closest && e.target.closest(".panel, .stat"); if (!t) return;
    const r = t.getBoundingClientRect(); t.style.setProperty("--mx", (e.clientX - r.left) + "px"); t.style.setProperty("--my", (e.clientY - r.top) + "px");
  });
  function handle(ev) {
    const d = describe(ev);
    switch (ev.type) {
      case "snapshot":
        resetState(); S.total = ev.stats.blocks_total; S.blockSize = (ev.cache && ev.cache.block_size) || S.blockSize;
        buildBlocks(S.total); (ev.recent_steps || []).forEach(applyStep); S.nextStep = ev.stats.step;
        markEndedBefore(ev.recent_steps || []);
        renderBlocks(ev.block_map); updateStats(ev.stats); break;
      case "submitted": ensureReq(ev.id, { arrival: S.nextStep, prompt: ev.prompt_len, max: ev.max_new_tokens }); break;
      case "rejected": if (ev.id) ensureReq(ev.id, { arrival: S.nextStep, rejected: true }); break;
      case "step":
        beat(); applyStep(ev); updateStats(ev.stats); renderBlocks(ev.block_map);
        S.rate.push({ t: performance.now(), n: ev.prefill.length + ev.decode.length }); updateRate(); break;
      case "finished": { const r = S.reqs.get(ev.id); if (r) { r.kind = ev.kind; r.tokens = ev.tokens; r.endStep = S.nextStep - 1; } break; }
    }
    if (d) line(d[0], d[1]);
    dirty(); renderLegend();
  }

  // ------------------------------------------------------------------------------ sources
  let sock = null, retryTimer = null, replayTimers = [], recorded = null, wantLive = true;
  function stopReplay() { replayTimers.forEach(clearTimeout); replayTimers = []; }
  async function startReplay() {
    stopReplay(); setMode("replay");
    try {
      recorded = recorded || await (await fetch("assets/recorded.json")).json();
    } catch (e) { setMode("off"); $("#mode-note").textContent = "No server and no recording found. Start the server: python scripts/serve.py"; return; }
    const events = recorded.events, endAt = events[events.length - 1].dt + 2500;
    events.forEach((ev) => replayTimers.push(setTimeout(() => handle(ev), reduce ? Math.min(ev.dt, 50 * events.indexOf(ev)) : ev.dt)));
    replayTimers.push(setTimeout(startReplay, endAt));            // loop
  }
  function connectLive() {
    clearTimeout(retryTimer);
    let opened = false;
    const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/events";
    let s;
    try { s = new WebSocket(url); } catch (e) { return toReplay(); }
    sock = s;
    const timer = setTimeout(() => { if (!opened) { s.close(); } }, 2500);
    s.onopen = () => { opened = true; clearTimeout(timer); stopReplay(); setMode("live"); };
    s.onmessage = (e) => handle(JSON.parse(e.data));
    s.onclose = () => {
      clearTimeout(timer); if (sock !== s) return;
      if (opened) { setMode("off"); retryTimer = setTimeout(connectLive, 2000); } else toReplay();
    };
    s.onerror = () => {};
  }
  function toReplay() { if (S.mode !== "replay") startReplay(); retryTimer = setTimeout(() => wantLive && connectLive(), 15000); }
  $("#replay").onclick = () => {
    if (S.mode === "replay") { wantLive = true; connectLive(); }
    else { wantLive = false; clearTimeout(retryTimer); const old = sock; sock = null; if (old) old.close(); startReplay(); }
  };

  // ------------------------------------------------------------------------------ try it
  const results = $("#results"), maxnew = $("#maxnew");
  maxnew.oninput = () => { $("#maxnew-out").textContent = maxnew.value; };
  function streamReply(prompt, max) {
    const card = document.createElement("div"), head = document.createElement("h4"), body = document.createElement("div");
    card.className = "res-card"; head.textContent = "sending..."; card.append(head, body);
    results.prepend(card); while (results.childElementCount > 6) results.lastChild.remove();
    const s = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/v1/stream");
    const fail = (t) => { head.textContent = t; card.style.setProperty("--c", "#ffb020"); };
    s.onopen = () => s.send(JSON.stringify({ prompt, max_new_tokens: max }));
    s.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "accepted") { card.style.setProperty("--c", colorOf(m.id)); head.textContent = m.id; head.append(` · ${m.prompt_tokens} prompt tokens`); }
      else if (m.type === "token") { const c = document.createElement("span"); c.className = "chip"; c.textContent = m.token; body.appendChild(c); }
      else if (m.type === "done") { const sp = document.createElement("span"); sp.textContent = " · done"; head.appendChild(sp); }
      else if (m.type === "rejected") fail("refused: " + m.reason);
      else if (m.type === "error") fail("error " + m.status + ": " + JSON.stringify(m.detail));
    };
    s.onerror = () => fail("could not reach the engine");
  }
  $("#form").onsubmit = (e) => { e.preventDefault(); const p = $("#prompt").value.trim(); if (p) streamReply(p, Number(maxnew.value)); };
  $("#burst").onclick = () => {
    [["hello grid", 10], ["tell me about paged attention", 14], ["a b c d e f g h", 6], ["short", 4]]
      .forEach(([p, n], i) => setTimeout(() => streamReply(p, n), i * 150));
  };

  // ------------------------------------------------------------------------------ go
  setMode("connecting");
  if (new URLSearchParams(location.search).has("replay")) { wantLive = false; startReplay(); } else connectLive();
})();
