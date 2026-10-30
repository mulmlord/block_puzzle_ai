"use strict";

// ---- constants --------------------------------------------------------------------------------
const N = 10;
const CELL = 46;
const GAP = 4;
const PAD = 22;
const TRAY_H = 150;
const W = PAD * 2 + N * CELL;
const H = PAD + N * CELL + TRAY_H;
const TURBO = 61; // speed slider value meaning "no animation"
// One colour per piece group (index = group id from the recovered piece table).
const COLORS = [
  "#f2c14e", "#4cc9f0", "#f78c6b", "#7bd389", "#ef476f", "#9b8cff",
  "#06d6a0", "#ff9f1c", "#3a86ff", "#c77dff", "#f15bb5", "#a7c957",
];
const EMPTY = "#232a38";

const $ = (id) => document.getElementById(id);
const canvas = $("board");
const ctx = canvas.getContext("2d");

const ui = {
  ws: null,
  resolve: null,
  playing: false,
  busy: false,
  current: null, // latest snapshot from the server
  speed: 4,
};

// ---- canvas drawing ---------------------------------------------------------------------------
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.aspectRatio = `${W} / ${H}`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const ch = (v) => Math.max(0, Math.min(255, Math.round(v + amount * 255)));
  const r = ch(n >> 16), g = ch((n >> 8) & 255), b = ch(n & 255);
  return `rgb(${r},${g},${b})`;
}

function drawBlock(x, y, size, color, { alpha = 1, scale = 1, outline = false } = {}) {
  const s = size * scale;
  const ox = x + (size - s) / 2, oy = y + (size - s) / 2;
  const r = Math.max(3, s * 0.18);
  ctx.save();
  ctx.globalAlpha = alpha;
  if (outline) {
    roundRect(ox + 1.5, oy + 1.5, s - 3, s - 3, r);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();
    return;
  }
  const grad = ctx.createLinearGradient(ox, oy, ox, oy + s);
  grad.addColorStop(0, shade(color, 0.12));
  grad.addColorStop(1, shade(color, -0.1));
  roundRect(ox, oy, s, s, r);
  ctx.fillStyle = grad;
  ctx.fill();
  roundRect(ox + s * 0.14, oy + s * 0.1, s * 0.72, s * 0.22, r * 0.6);
  ctx.fillStyle = "rgba(255,255,255,0.22)";
  ctx.fill();
  ctx.restore();
}

function cellXY(i) {
  return [PAD + (i % N) * CELL + GAP / 2, PAD + Math.floor(i / N) * CELL + GAP / 2];
}

function drawBoard(board, { ghost = null, flash = null } = {}) {
  const size = CELL - GAP;
  for (let i = 0; i < N * N; i++) {
    const [x, y] = cellXY(i);
    roundRect(x, y, size, size, 7);
    ctx.fillStyle = EMPTY;
    ctx.fill();
    if (board[i] >= 0) drawBlock(x, y, size, COLORS[board[i]]);
  }
  if (ghost) {
    for (const i of ghost.cells) {
      const [x, y] = cellXY(i);
      drawBlock(x, y, size, ghost.color, { alpha: ghost.alpha * 0.45 });
      drawBlock(x, y, size, ghost.color, { outline: true, alpha: ghost.alpha });
    }
  }
  if (flash) {
    for (const i of flash.cells) {
      const [x, y] = cellXY(i);
      roundRect(x, y, size, size, 7);
      ctx.fillStyle = EMPTY;
      ctx.fill();
      drawBlock(x, y, size, "#ffffff", { alpha: flash.alpha, scale: flash.scale });
    }
  }
}

function drawTray(pending, { highlight = -1, hidden = -1 } = {}) {
  const top = PAD + N * CELL + 18;
  const slotW = (W - PAD * 2) / 3;
  const size = 24;
  pending.forEach((piece, slot) => {
    const x0 = PAD + slot * slotW;
    roundRect(x0 + 6, top, slotW - 12, TRAY_H - 36, 14);
    ctx.fillStyle = slot === highlight ? "rgba(91,140,255,0.18)" : "rgba(255,255,255,0.03)";
    ctx.fill();
    if (slot === highlight) {
      ctx.strokeStyle = "#5b8cff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (slot === hidden) return;
    const h = 1 + Math.max(...piece.cells.map((c) => c[0]));
    const w = 1 + Math.max(...piece.cells.map((c) => c[1]));
    const cx = x0 + slotW / 2 - (w * size) / 2;
    const cy = top + (TRAY_H - 36) / 2 - (h * size) / 2;
    for (const [r, c] of piece.cells) {
      drawBlock(cx + c * size + 1, cy + r * size + 1, size - 2, COLORS[piece.group]);
    }
  });
}

function drawState(snapshot, extras = {}) {
  ctx.clearRect(0, 0, W, H);
  drawBoard(extras.board || snapshot.board, extras);
  drawTray(extras.pending || snapshot.pending, extras);
}

// ---- move animation ---------------------------------------------------------------------------
function animateStep(prev, msg) {
  const duration = ui.speed >= TURBO ? 0 : Math.min(900, 1000 / ui.speed);
  const color = COLORS[msg.group];
  const placedBoard = prev.board.slice();
  for (const i of msg.placed) placedBoard[i] = msg.group;
  if (duration < 60) {
    drawState(msg.state);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const start = performance.now();
    function frame(now) {
      const p = Math.min(1, (now - start) / duration);
      if (p < 0.35) {
        const pulse = 0.6 + 0.4 * Math.sin((p / 0.35) * Math.PI);
        drawState(prev, { ghost: { cells: msg.placed, color, alpha: pulse }, highlight: msg.move.slot });
      } else if (p < 0.6 || !msg.cleared.length) {
        drawState(prev, { board: placedBoard, hidden: msg.move.slot, highlight: msg.move.slot });
      } else {
        const q = (p - 0.6) / 0.4;
        drawState(msg.state, { board: placedBoard, flash: { cells: msg.cleared, alpha: 1 - q, scale: 1 - 0.5 * q } });
      }
      if (p < 1) requestAnimationFrame(frame);
      else {
        drawState(msg.state);
        resolve();
      }
    }
    requestAnimationFrame(frame);
  });
}

// ---- server connection ------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    $("conn").classList.add("on");
    ui.ws = ws;
    newGame();
  };
  ws.onclose = () => {
    $("conn").classList.remove("on");
    ui.ws = null;
    ui.busy = false;
    setTimeout(connect, 2000);
  };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (ui.resolve) {
      const resolve = ui.resolve;
      ui.resolve = null;
      resolve(msg);
    }
  };
}

function request(payload) {
  return new Promise((resolve) => {
    ui.resolve = resolve;
    ui.ws.send(JSON.stringify(payload));
  });
}

// ---- game flow --------------------------------------------------------------------------------
async function newGame(seed) {
  if (!ui.ws) return;
  const raw = seed ?? $("seed").value;
  const msg = await request({ cmd: "new", agent: $("agent").value || "heuristic", seed: raw === "" ? null : Number(raw) });
  if (msg.type === "error") return showToast(msg.message);
  ui.current = msg.state;
  $("seed").placeholder = `random (now ${msg.seed})`;
  $("overlay").classList.add("hidden");
  updateStats(msg.state, msg.best);
  for (const id of ["legal", "points", "expected"]) $(id).textContent = "–";
  drawState(msg.state);
}

async function step() {
  if (ui.busy || !ui.ws || !ui.current || ui.current.done) return;
  ui.busy = true;
  const prev = ui.current;
  const msg = await request({ cmd: "step" });
  if (msg.type !== "step") {
    ui.busy = false;
    return;
  }
  ui.current = msg.state;
  updateStats(msg.state, msg.best);
  updateInsight(msg);
  if (msg.lines) showToast(`+${msg.points.toLocaleString()} · ${msg.lines} line${msg.lines > 1 ? "s" : ""}`);
  await animateStep(prev, msg);
  if (msg.state.done) gameOver(msg);
  ui.busy = false;
}

function gameOver(msg) {
  $("overlay-score").textContent = `${msg.state.score.toLocaleString()} points in ${msg.state.moves} moves`;
  $("overlay").classList.remove("hidden");
  renderHistory(msg.history);
  if (ui.playing && $("auto").checked) {
    setTimeout(() => {
      $("seed").value = "";
      newGame().then(loop);
    }, ui.speed >= TURBO ? 400 : 1600);
  }
}

async function loop() {
  while (ui.playing && ui.current && !ui.current.done) {
    const started = performance.now();
    await step();
    const wait = ui.speed >= TURBO ? 0 : Math.max(0, 1000 / ui.speed - (performance.now() - started));
    await new Promise((r) => setTimeout(r, wait));
  }
}

function setPlaying(on) {
  ui.playing = on;
  $("play").textContent = on ? "❚❚ Pause" : "▶ Play";
  if (on && ui.current?.done) newGame().then(loop);
  else if (on) loop();
}

// ---- panels -----------------------------------------------------------------------------------
function updateStats(state, best) {
  $("score").textContent = state.score.toLocaleString();
  $("best").textContent = (best || 0).toLocaleString();
  $("moves").textContent = state.moves.toLocaleString();
  $("lines").textContent = state.lines.toLocaleString();
}

function updateInsight(msg) {
  $("legal").textContent = msg.legal_moves;
  $("points").textContent = `+${msg.points}${msg.lines ? ` (${msg.lines} line${msg.lines > 1 ? "s" : ""})` : ""}`;
  $("expected").textContent = msg.expected == null ? "n/a" : Math.round(msg.expected).toLocaleString();
}

let toastTimer = null;
function showToast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 1100);
}

function renderHistory(history) {
  if (!history.length) return;
  const scores = history.map((g) => g.score);
  const max = Math.max(...scores);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  $("session-summary").textContent =
    `${history.length} game${history.length > 1 ? "s" : ""} · mean ${Math.round(mean).toLocaleString()} · best ${max.toLocaleString()}`;
  $("history").innerHTML = history
    .slice()
    .reverse()
    .map(
      (g) => `<li class="${g.score === max ? "best" : ""}" title="seed ${g.seed}, ${g.moves} moves">
        <span>#${history.indexOf(g) + 1}</span>
        <span class="bar" style="width:${(100 * g.score) / max}%"></span>
        <span class="val">${g.score.toLocaleString()}</span></li>`,
    )
    .join("");
}

// ---- training chart (single series + labelled baseline reference) -----------------------------
const trim = (s) => s.replace(/\.0$/, "");
const fmtK = (v) =>
  v >= 1e6 ? `${trim((v / 1e6).toFixed(1))}M` : v >= 1000 ? `${trim((v / 1000).toFixed(v >= 10000 ? 0 : 1))}k` : `${Math.round(v)}`;

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

function renderChart(points, baseline) {
  const el = $("chart");
  const cw = 380, ch = 210, m = { l: 40, r: 14, t: 14, b: 28 };
  const iw = cw - m.l - m.r, ih = ch - m.t - m.b;
  if (!points.length) {
    el.innerHTML = `<svg viewBox="0 0 ${cw} ${ch}"><text class="empty" x="${cw / 2}" y="${ch / 2}" text-anchor="middle">No evaluations yet: the first one runs after 10 minutes of training.</text></svg>`;
    $("chart-table").innerHTML = "";
    return;
  }
  const xMax = Math.max(1, ...points.map((p) => p.minutes));
  const yMax = niceMax(Math.max(...points.map((p) => p.mean), baseline || 0) * 1.08);
  const x = (v) => m.l + (v / xMax) * iw;
  const y = (v) => m.t + ih - (v / yMax) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  const xticks = [0, 0.5, 1].map((f) => f * xMax);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.minutes).toFixed(1)},${y(p.mean).toFixed(1)}`).join("");
  const ref = baseline
    ? `<line class="ref" x1="${m.l}" x2="${m.l + iw}" y1="${y(baseline)}" y2="${y(baseline)}"/>
       <text class="ref-label" x="${m.l + 4}" y="${y(baseline) - 5}">Heuristic baseline ${fmtK(baseline)}</text>`
    : "";
  el.innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" role="img" aria-label="Evaluation mean score over training time">
    <g class="grid">${ticks.map((t) => `<line x1="${m.l}" x2="${m.l + iw}" y1="${y(t)}" y2="${y(t)}"/>`).join("")}</g>
    <g class="axis">
      ${ticks.map((t) => `<text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end">${fmtK(t)}</text>`).join("")}
      ${xticks.map((t) => `<text x="${x(t)}" y="${ch - 8}" text-anchor="middle">${Math.round(t)} min</text>`).join("")}
    </g>
    ${ref}
    <path class="series" d="${path}"/>
    ${points.map((p) => `<circle class="dot" r="3" cx="${x(p.minutes)}" cy="${y(p.mean)}"/>`).join("")}
    <line class="crosshair" id="xh" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/>
    <circle class="dot" id="xh-dot" r="5" visibility="hidden"/>
    <rect id="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>
  </svg><div class="tooltip hidden" id="tip"></div>`;

  const svg = el.querySelector("svg");
  const tip = $("tip");
  $("hit").addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const mx = ((event.clientX - box.left) / box.width) * cw;
    const p = points.reduce((a, b) => (Math.abs(x(b.minutes) - mx) < Math.abs(x(a.minutes) - mx) ? b : a));
    const px = x(p.minutes), py = y(p.mean);
    $("xh").setAttribute("x1", px);
    $("xh").setAttribute("x2", px);
    $("xh").setAttribute("visibility", "visible");
    $("xh-dot").setAttribute("cx", px);
    $("xh-dot").setAttribute("cy", py);
    $("xh-dot").setAttribute("visibility", "visible");
    tip.innerHTML = `<b>${Math.round(p.mean).toLocaleString()}</b> mean<br><span>${p.minutes.toFixed(0)} min · ${(p.moves / 1e6).toFixed(1)}M moves<br>median ${Math.round(p.median).toLocaleString()} · max ${p.max.toLocaleString()}</span>`;
    tip.style.left = `${(px / cw) * box.width}px`;
    tip.style.top = `${(py / ch) * box.height}px`;
    tip.classList.remove("hidden");
  });
  $("hit").addEventListener("mouseleave", () => {
    $("xh").setAttribute("visibility", "hidden");
    $("xh-dot").setAttribute("visibility", "hidden");
    tip.classList.add("hidden");
  });

  $("chart-table").innerHTML =
    "<tr><th>Minutes</th><th>Moves</th><th>Mean</th><th>Median</th><th>Max</th></tr>" +
    points
      .map(
        (p) => `<tr><td>${p.minutes.toFixed(0)}</td><td>${(p.moves / 1e6).toFixed(1)}M</td><td>${Math.round(p.mean).toLocaleString()}</td><td>${Math.round(p.median).toLocaleString()}</td><td>${p.max.toLocaleString()}</td></tr>`,
      )
      .join("");
}

let training = null;
async function refreshTraining() {
  try {
    training = await (await fetch("/api/training")).json();
  } catch {
    return;
  }
  const select = $("run");
  const runs = Object.keys(training.runs);
  const keep = select.value;
  select.innerHTML = runs.map((r) => `<option>${r}</option>`).join("");
  select.value = runs.includes(keep) ? keep : runs.includes("main") ? "main" : runs[0] || "";
  drawTraining();
}

function drawTraining() {
  const points = (training && training.runs[$("run").value]) || [];
  renderChart(points, training?.baselines?.heuristic?.mean);
}

async function refreshAgents() {
  const agents = await (await fetch("/api/agents")).json();
  const select = $("agent");
  const keep = select.value;
  select.innerHTML = agents.map((a) => `<option value="${a.id}">${a.label}</option>`).join("");
  if (agents.some((a) => a.id === keep)) select.value = keep;
}

// ---- wiring -----------------------------------------------------------------------------------
$("play").onclick = () => setPlaying(!ui.playing);
$("step").onclick = () => {
  setPlaying(false);
  step();
};
$("new").onclick = () => newGame();
$("dice").onclick = () => {
  $("seed").value = Math.floor(Math.random() * 1e6);
};
$("agent").onchange = () => newGame();
$("run").onchange = drawTraining;
$("speed").oninput = (e) => {
  ui.speed = Number(e.target.value);
  $("speed-label").textContent = ui.speed >= TURBO ? "Turbo (no animation)" : `${ui.speed} moves/s`;
};

setupCanvas();
drawState({ board: Array(N * N).fill(-1), pending: [] });
refreshAgents().then(connect);
refreshTraining();
setInterval(refreshTraining, 30000);
setInterval(refreshAgents, 60000);
