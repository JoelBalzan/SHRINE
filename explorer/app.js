(function () {
"use strict";
const K = makeCore();
const $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
const NSTACK = 8, TW0 = 900, TW1 = 1150, TW = TW1 - TW0;
const S = { gen: 0, params: null, ctx: null, kc: null, kcAuto: null, flt: null, stack: [], order: [], sel: 0, row: null,
  mc: [], mcN: 200, mcDoneW: 0, workers: [], sd: null, rd: null, st: null, step4: 0, step5: 0, hoverCard: -1, hoverRow: -1, heat: null };

// ---------------------------------------------------------------- palette
let P = {};
const TOK = ["bg", "surface", "surface-2", "ink", "ink-2", "muted", "rule", "grid", "accent", "s23", "w", "deriv", "sig", "mc", "heat-0", "heat-1", "heat-2"];
let LUT = null;
function hexRGB(h) { h = h.trim().replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of TOK) P[k] = cs.getPropertyValue("--" + k).trim() || "#888";
  P.mono = cs.getPropertyValue("--font-mono").trim() || "monospace";
  const a = hexRGB(P["heat-0"]), b = hexRGB(P["heat-1"]), c = hexRGB(P["heat-2"]);
  LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255, u = t < 0.5 ? t * 2 : (t - 0.5) * 2, p = t < 0.5 ? a : b, q = t < 0.5 ? b : c;
    for (let j = 0; j < 3; j++) LUT[i * 3 + j] = p[j] + (q[j] - p[j]) * u;
  }
}
function alpha(hex, a) { const [r, g, b] = hexRGB(hex); return `rgba(${r},${g},${b},${a})`; }

// ---------------------------------------------------------------- formatting
const MINUS = "−";
function sig(v, n = 3) { if (v == null || !isFinite(v)) return "–"; if (v === 0) return "0"; const s = Number(v.toPrecision(n)).toString(); return s.replace("-", MINUS); }
function fx(v, d) { if (v == null || !isFinite(v)) return "–"; return v.toFixed(d).replace("-", MINUS); }
function fdm(v) { if (v == null || !isFinite(v)) return "–"; return (v >= 0 ? "+" : MINUS) + Math.abs(v).toFixed(4); }
const V = {};
function setV(k, v) { V[k] = v; for (const el of document.querySelectorAll(`[data-v="${k}"]`)) el.textContent = v; }

// ---------------------------------------------------------------- plotting
function prep(cv) {
  const w = cv.clientWidth; if (!w) return null;
  const h = +cv.dataset.h || cv.clientHeight || 200, r = Math.min(window.devicePixelRatio || 1, 2);
  cv.style.height = h + "px";
  if (cv.width !== Math.round(w * r) || cv.height !== Math.round(h * r)) { cv.width = Math.round(w * r); cv.height = Math.round(h * r); }
  const g = cv.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0); g.clearRect(0, 0, w, h);
  return { g, w, h };
}
function niceTicks(a, b, n) {
  const span = b - a; if (!(span > 0)) return [a];
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function p10(v) { const e = Math.round(Math.log10(v)); if (e >= 0 && e <= 3) return String(Math.pow(10, e)); return "10" + String(e).split("").map(c => SUP[c]).join(""); }
function logTicks(a, b) { const o = []; for (let e = Math.ceil(Math.log10(a) - 1e-9); e <= Math.floor(Math.log10(b) + 1e-9); e++) o.push(Math.pow(10, e)); return o; }
function tickFmt(t) {
  if (t.length < 2) return v => sig(v, 3);
  const step = Math.abs(t[1] - t[0]), d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v => fx(v, d);
}
class Ax {
  constructor(cv, m) {
    const p = prep(cv); this.ok = !!p; if (!p) return;
    Object.assign(this, p); this.cv = cv;
    this.m = Object.assign({ l: 56, r: 18, t: 10, b: 34 }, m || {});
    this.px0 = this.m.l; this.px1 = this.w - this.m.r; this.py0 = this.h - this.m.b; this.py1 = this.m.t;
    this.g.font = `11px ${P.mono}`; this.g.lineJoin = "round"; this.g.lineCap = "round";
  }
  xr(a, b, log) { this.xa = a; this.xb = b; this.xlog = !!log; return this; }
  yr(a, b, log) { this.ya = a; this.yb = b; this.ylog = !!log; return this; }
  X(v) { const f = this.xlog ? (Math.log10(v) - Math.log10(this.xa)) / (Math.log10(this.xb) - Math.log10(this.xa)) : (v - this.xa) / (this.xb - this.xa); return this.px0 + f * (this.px1 - this.px0); }
  Y(v) { const f = this.ylog ? (Math.log10(Math.max(v, 1e-300)) - Math.log10(this.ya)) / (Math.log10(this.yb) - Math.log10(this.ya)) : (v - this.ya) / (this.yb - this.ya); return this.py0 - f * (this.py0 - this.py1); }
  iX(px) { const f = (px - this.px0) / (this.px1 - this.px0); return this.xlog ? Math.pow(10, Math.log10(this.xa) + f * (Math.log10(this.xb) - Math.log10(this.xa))) : this.xa + f * (this.xb - this.xa); }
  iY(py) { const f = (this.py0 - py) / (this.py0 - this.py1); return this.ylog ? Math.pow(10, Math.log10(this.ya) + f * (Math.log10(this.yb) - Math.log10(this.ya))) : this.ya + f * (this.yb - this.ya); }
  frame(o) {
    o = o || {}; const g = this.g;
    const xt = o.xt || (this.xlog ? logTicks(this.xa, this.xb) : niceTicks(this.xa, this.xb, o.nx || Math.max(3, Math.floor((this.px1 - this.px0) / 80))));
    const yt = o.yt || (this.ylog ? logTicks(this.ya, this.yb) : niceTicks(this.ya, this.yb, o.ny || Math.max(3, Math.floor((this.py0 - this.py1) / 46))));
    const xf = o.xf || (this.xlog ? p10 : tickFmt(xt));
    let yf = o.yf || (this.ylog ? p10 : tickFmt(yt)), ylabel = o.ylabel;
    if (!o.yf && !this.ylog && yt.length > 1) {
      const mx = Math.max(...yt.map(Math.abs));
      if (mx > 0 && mx < 0.01) { const e = Math.floor(Math.log10(mx)), sc = Math.pow(10, -e), f2 = tickFmt(yt.map(v => v * sc)); yf = v => f2(v * sc); if (ylabel) ylabel += ` (×10${String(e).split("").map(c => SUP[c]).join("")})`; }
    }
    g.lineWidth = 1;
    if (o.grid !== false) {
      g.strokeStyle = P.grid; g.beginPath();
      for (const v of yt) { const y = Math.round(this.Y(v)) + 0.5; if (y < this.py1 - 1 || y > this.py0 + 1) continue; g.moveTo(this.px0, y); g.lineTo(this.px1, y); }
      if (o.xgrid) for (const v of xt) { const x = Math.round(this.X(v)) + 0.5; g.moveTo(x, this.py1); g.lineTo(x, this.py0); }
      g.stroke();
    }
    g.strokeStyle = P.rule; g.beginPath(); g.moveTo(this.px0, this.py0 + 0.5); g.lineTo(this.px1, this.py0 + 0.5);
    if (!o.noYaxis) { g.moveTo(this.px0 - 0.5, this.py1); g.lineTo(this.px0 - 0.5, this.py0); }
    g.stroke();
    g.fillStyle = P.muted; g.textBaseline = "top"; g.textAlign = "center";
    if (!o.noXlab) for (const v of xt) { const x = this.X(v); if (x < this.px0 - 2 || x > this.px1 + 2) continue; g.fillText(xf(v), x, this.py0 + 6); }
    g.textAlign = "right"; g.textBaseline = "middle";
    if (!o.noYlab) for (const v of yt) { const y = this.Y(v); if (y < this.py1 - 2 || y > this.py0 + 2) continue; g.fillText(yf(v), this.px0 - 6, y); }
    g.fillStyle = P["ink-2"];
    if (o.xlabel) { g.textBaseline = "bottom"; richText(g, o.xlabel, (this.px0 + this.px1) / 2, this.h - 3, "center"); }
    if (ylabel) { g.save(); g.translate(11, (this.py0 + this.py1) / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; richText(g, ylabel, 0, 0, "center"); g.restore(); }
    return this;
  }
  clip() { const g = this.g; g.save(); g.beginPath(); g.rect(this.px0, this.py1 - 1, this.px1 - this.px0, this.py0 - this.py1 + 2); g.clip(); return this; }
  unclip() { this.g.restore(); return this; }
  st(o) { const g = this.g; g.strokeStyle = o.c || P.ink; g.lineWidth = o.lw || 1.5; g.globalAlpha = o.a == null ? 1 : o.a; g.setLineDash(o.dash || []); }
  rs() { const g = this.g; g.globalAlpha = 1; g.setLineDash([]); }
  line(xs, ys, o) {
    o = o || {}; const g = this.g; this.st(o); g.beginPath(); let pen = false;
    const n = ys.length;
    for (let i = 0; i < n; i++) {
      const xv = typeof xs === "function" ? xs(i) : xs[i], yv = ys[i];
      if (!isFinite(yv) || !isFinite(xv) || (this.ylog && yv <= 0)) { pen = false; continue; }
      const x = this.X(xv), y = this.Y(yv);
      if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
    }
    g.stroke(); this.rs(); return this;
  }
  band(xs, lo, hi, o) {
    o = o || {}; const g = this.g, n = lo.length; g.fillStyle = o.c || P.accent; g.globalAlpha = o.a == null ? 0.16 : o.a; g.beginPath();
    for (let i = 0; i < n; i++) { const x = this.X(typeof xs === "function" ? xs(i) : xs[i]), y = this.Y(hi[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    for (let i = n - 1; i >= 0; i--) g.lineTo(this.X(typeof xs === "function" ? xs(i) : xs[i]), this.Y(lo[i]));
    g.closePath(); g.fill(); g.globalAlpha = 1; return this;
  }
  vline(x, o) { o = o || {}; const g = this.g; this.st(o); const X = Math.round(this.X(x)) + 0.5; g.beginPath(); g.moveTo(X, o.y0 != null ? this.Y(o.y0) : this.py1); g.lineTo(X, o.y1 != null ? this.Y(o.y1) : this.py0); g.stroke(); this.rs(); return this; }
  hline(y, o) { o = o || {}; const g = this.g; this.st(o); const Y = Math.round(this.Y(y)) + 0.5; g.beginPath(); g.moveTo(o.x0 != null ? this.X(o.x0) : this.px0, Y); g.lineTo(o.x1 != null ? this.X(o.x1) : this.px1, Y); g.stroke(); this.rs(); return this; }
  rectD(x0, x1, y0, y1, o) { const g = this.g; g.fillStyle = o.c; g.globalAlpha = o.a == null ? 0.15 : o.a; const a = this.X(x0), b = this.X(x1), c = this.Y(y0), d = this.Y(y1); g.fillRect(Math.min(a, b), Math.min(c, d), Math.abs(b - a), Math.abs(d - c)); g.globalAlpha = 1; return this; }
  dots(xs, ys, o) { o = o || {}; const g = this.g; g.fillStyle = o.c || P.ink; g.globalAlpha = o.a == null ? 1 : o.a; const r = o.r || 2.5;
    for (let i = 0; i < ys.length; i++) { if (!isFinite(xs[i]) || !isFinite(ys[i])) continue; g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI); g.fill(); }
    if (o.ring) { g.globalAlpha = 1; g.strokeStyle = o.ring; g.lineWidth = 1; for (let i = 0; i < ys.length; i++) { g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI); g.stroke(); } }
    g.globalAlpha = 1; return this; }
  text(s, x, y, o) { o = o || {}; const g = this.g; g.fillStyle = o.c || P["ink-2"]; g.textBaseline = o.bl || "middle";
    if (o.font) { g.font = o.font; g.textAlign = o.al || "left"; g.fillText(s, o.px ? x : this.X(x), o.px ? y : this.Y(y)); }
    else richText(g, s, o.px ? x : this.X(x), o.px ? y : this.Y(y), o.al || "left");
    g.font = `11px ${P.mono}`; return this; }
  bars(xs, ys, o) { o = o || {}; const g = this.g; g.fillStyle = o.c || P.mc; g.globalAlpha = o.a == null ? 1 : o.a; const base = o.base || 0;
    for (let i = 0; i < ys.length; i++) { const x0 = this.X(xs[i] - o.bw / 2), x1 = this.X(xs[i] + o.bw / 2), y0 = this.Y(o.b0 ? o.b0[i] : base), y1 = this.Y((o.b0 ? o.b0[i] : 0) + ys[i]); const gap = o.gap == null ? 1 : o.gap;
      const L = Math.min(x0, x1) + gap / 2, W = Math.max(Math.abs(x1 - x0) - gap, 0.8); g.fillRect(L, Math.min(y0, y1), W, Math.abs(y1 - y0)); }
    g.globalAlpha = 1; return this; }
}
function heatImg(get, nr, nc, vmin, vmax) {
  const c = document.createElement("canvas"); c.width = nc; c.height = nr;
  const g = c.getContext("2d"), im = g.createImageData(nc, nr), d = im.data, sc = 255 / (vmax - vmin);
  for (let r = 0; r < nr; r++) for (let q = 0; q < nc; q++) {
    let v = Math.round((get(r, q) - vmin) * sc); v = v < 0 ? 0 : v > 255 ? 255 : v;
    const o = (r * nc + q) * 4; d[o] = LUT[v * 3]; d[o + 1] = LUT[v * 3 + 1]; d[o + 2] = LUT[v * 3 + 2]; d[o + 3] = 255;
  }
  g.putImageData(im, 0, 0); return c;
}
function blit(ax, img, smooth) { const g = ax.g; g.imageSmoothingEnabled = !!smooth; g.drawImage(img, ax.px0, ax.py1, ax.px1 - ax.px0, ax.py0 - ax.py1); g.imageSmoothingEnabled = true; }

// rich canvas text: "_{...}" sets a subscript and "^{...}" a superscript
const RICH = /[_^]\{/;
function richSegs(str) {
  const out = []; let i = 0, buf = "";
  while (i < str.length) {
    if ((str[i] === "_" || str[i] === "^") && str[i + 1] === "{") {
      if (buf) { out.push({ t: buf, m: 0 }); buf = ""; }
      const j = str.indexOf("}", i + 2); out.push({ t: str.slice(i + 2, j), m: str[i] === "_" ? 1 : 2 }); i = j + 1;
    } else buf += str[i++];
  }
  if (buf) out.push({ t: buf, m: 0 });
  return out;
}
function richFonts(sz) { const sm = Math.round(sz * 0.78); return [`${sz}px ${P.mono}`, `${sm}px ${P.mono}`, `${sm}px ${P.mono}`]; }
function richWidth(g, str, sz) {
  sz = sz || 11; if (!RICH.test(str)) { g.font = `${sz}px ${P.mono}`; return g.measureText(str).width; }
  const f = richFonts(sz); let w = 0; for (const sg of richSegs(str)) { g.font = f[sg.m]; w += g.measureText(sg.t).width; } g.font = f[0]; return w;
}
// draws at (x, y) with the context's current textBaseline; align: left | center | right; halo: stroke colour drawn underneath
function richText(g, str, x, y, align, sz, halo) {
  sz = sz || 11; const f = richFonts(sz), segs = RICH.test(str) ? richSegs(str) : [{ t: str, m: 0 }];
  let w = 0; for (const sg of segs) { g.font = f[sg.m]; sg.w = g.measureText(sg.t).width; w += sg.w; }
  let X = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  const al = g.textAlign; g.textAlign = "left";
  for (const sg of segs) {
    g.font = f[sg.m]; const dy = sg.m === 1 ? sz * 0.32 : sg.m === 2 ? -sz * 0.38 : 0;
    if (halo) { const fs = g.fillStyle; g.strokeStyle = halo; g.lineWidth = 3.5; g.lineJoin = "round"; g.strokeText(sg.t, X, y + dy); g.fillStyle = fs; }
    g.fillText(sg.t, X, y + dy); X += sg.w;
  }
  g.textAlign = al; g.font = f[0]; return w;
}

// tooltip
const tipEl = $("#tip");
function tip(ev, html) { if (!html) { tipEl.hidden = true; return; } tipEl.innerHTML = html; tipEl.hidden = false; const pad = 14; let x = ev.clientX + pad, y = ev.clientY + pad; const r = tipEl.getBoundingClientRect(); if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad; if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad; tipEl.style.left = x + "px"; tipEl.style.top = y + "px"; }
function bindHover(cv, onMove, onClick) {
  cv.addEventListener("pointermove", ev => { const r = cv.getBoundingClientRect(); onMove(ev, ev.clientX - r.left, ev.clientY - r.top); });
  cv.addEventListener("pointerleave", ev => { tip(ev, null); onMove(ev, null, null); });
  if (onClick) { cv.style.cursor = "pointer"; cv.addEventListener("click", ev => { const r = cv.getBoundingClientRect(); onClick(ev, ev.clientX - r.left, ev.clientY - r.top); }); }
}

// ---------------------------------------------------------------- small math
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function nrm(a) { return Math.sqrt(dot(a, a)); }
function diff(x) { const o = new Float64Array(x.length - 1); for (let i = 0; i < o.length; i++) o[i] = x[i + 1] - x[i]; return o; }
function D1T(a) { const N = a.length + 1, o = new Float64Array(N); o[0] = -a[0]; for (let j = 1; j < N - 1; j++) o[j] = a[j - 1] - a[j]; o[N - 1] = a[N - 2]; return o; }
function cdiff(x) { const o = new Float64Array(x.length); for (let i = 1; i < x.length - 1; i++) o[i] = 0.5 * (x[i + 1] - x[i - 1]); return o; }
function sliceW(x, a, b) { return Array.from(x.slice(a, b)); }
function interp(x, t) { const i = Math.floor(t); if (i < 0 || i >= x.length - 1) return 0; const f = t - i; return x[i] * (1 - f) + x[i + 1] * f; }
function quantile(arr, q) { const a = Float64Array.from(arr).sort(); return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))]; }
function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }
function support(w, frac, pad) { let mx = 0; for (const v of w) mx = Math.max(mx, Math.abs(v)); let a = 0, b = w.length - 1; while (a < b && Math.abs(w[a]) < frac * mx) a++; while (b > a && Math.abs(w[b]) < frac * mx) b--; return [Math.max(0, a - pad), Math.min(w.length, b + pad)]; }

// ---------------------------------------------------------------- workers
function workerMain() {
  const K = makeCore(); let ctx = null, key = "";
  function ensure(p) { const k = JSON.stringify(p); if (k !== key) { ctx = K.setup(p); key = k; } }
  self.onmessage = function (ev) {
    const m = ev.data;
    if (m.cmd === "stack") {
      ensure(m.params);
      for (let j = 0; j < m.n; j++) { const ch = K.genChannels(ctx, K.seedFor(m.seed, j), true); const cube = K.dedisperse(ctx, K.chanFFT(ctx, ch.I)); self.postMessage({ type: "cube", gen: m.gen, j, cube }, [cube.buffer]); }
    } else if (m.cmd === "mc") {
      ensure(m.params); const flt = K.filters(ctx.N, m.kc, m.O);
      for (const j of m.js) { const r = K.mcOne(ctx, flt, K.seedFor(m.seed, j)); r.j = j; self.postMessage({ type: "mc", gen: m.gen, r }); }
      self.postMessage({ type: "mcdone", gen: m.gen });
    }
  };
}
const WORKER_SRC = makeCore.toString() + "\n(" + workerMain.toString() + ")();";
let inlineQ = [];
function inlinePump() {
  if (!inlineQ.length) return; const task = inlineQ.shift(); task(); setTimeout(inlinePump, 0);
}

function readParams() {
  const kcv = $("#c-kc").value.trim();
  return { amp: +$("#c-amp").value, selfnoise: $("#c-noise").value === "1", kcManual: kcv ? Math.max(5, Math.round(+kcv)) : null, O: +$("#c-O").value, mcN: +$("#c-mc").value, seed: Math.max(1, Math.round(+$("#c-seed").value || 1)) };
}
function setStatus() {
  const n = S.mc.length, N = S.mcN, have = S.stack.filter(Boolean).length;
  let t;
  if (have < NSTACK) t = `Dealing realisations ${have} / ${NSTACK}`;
  else if (n < N) t = `Monte Carlo ${n} / ${N}`;
  else t = `Monte Carlo complete · ${N} draws`;
  if (S.kc) t += ` · k_c = ${S.kc}${S.params.kcManual ? " (manual)" : " (get_kc)"}`;
  $("#status-txt").textContent = t;
  $("#status-bar").style.width = (100 * (have / NSTACK * 0.15 + n / N * 0.85)).toFixed(1) + "%";
}

function start() {
  S.workers.forEach(w => w.terminate()); S.workers = []; inlineQ = [];
  const gen = ++S.gen, p = readParams(); S.params = p; S.mcN = p.mcN;
  S.ctx = K.setup({ amp: p.amp, selfnoise: p.selfnoise });
  S.stack = new Array(NSTACK).fill(null); S.order = [...Array(NSTACK).keys()]; S.sel = 0; S.row = null;
  S.mc = []; S.mcDoneW = 0; S.kc = null; S.flt = null; S.sd = null; S.rd = null; S.st = null; S.heat = null;
  for (const el of $$('[data-show="self"]')) el.hidden = !p.selfnoise;
  buildDeck(); setStatus(); clearViews();
  const params = { amp: p.amp, selfnoise: p.selfnoise };
  const nW = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
  try {
    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    for (let i = 0; i < nW; i++) { const w = new Worker(url); w.onmessage = ev => onMsg(ev.data); S.workers.push(w); }
  } catch (e) { S.workers = []; }
  if (S.workers.length) S.workers[0].postMessage({ cmd: "stack", params, seed: p.seed, n: NSTACK, gen });
  else {
    const ctx = S.ctx;
    for (let j = 0; j < NSTACK; j++) inlineQ.push(() => { if (gen !== S.gen) return; const ch = K.genChannels(ctx, K.seedFor(p.seed, j), true); onMsg({ type: "cube", gen, j, cube: K.dedisperse(ctx, K.chanFFT(ctx, ch.I)) }); });
    setTimeout(inlinePump, 0);
  }
}
function dispatchMC() {
  const p = S.params, gen = S.gen, params = { amp: p.amp, selfnoise: p.selfnoise }, js = [...Array(S.mcN).keys()];
  if (S.workers.length) {
    const nW = S.workers.length;
    S.workers.forEach((w, i) => w.postMessage({ cmd: "mc", params, seed: p.seed, kc: S.kc, O: p.O, js: js.filter(j => j % nW === i), gen }));
  } else {
    for (const j of js) inlineQ.push(() => { if (gen !== S.gen) return; const r = K.mcOne(S.ctx, S.flt, K.seedFor(p.seed, j)); r.j = j; onMsg({ type: "mc", gen, r }); });
    inlineQ.push(() => onMsg({ type: "mcdone", gen, all: true }));
  }
}
function onMsg(m) {
  if (m.gen !== S.gen) return;
  if (m.type === "cube") onCube(m.j, m.cube);
  else if (m.type === "mc") { S.mc.push(m.r); schedule("mc"); }
  else if (m.type === "mcdone") { S.mcDoneW++; schedule("mc"); }
}
function onCube(j, cube) {
  const ctx = S.ctx, N = ctx.N;
  if (j === 0) {
    const C = new Float64Array(ctx.ND * N), row = new Float64Array(N);
    for (let m = 0; m < ctx.ND; m++) { for (let t = 0; t < N; t++) row[t] = cube[m * N + t]; C.set(ctx.dct(row), m * N); }
    S.kcAuto = K.getKc(C, ctx.ND, N); S.kc = S.params.kcManual || S.kcAuto; S.flt = K.filters(N, S.kc, S.params.O);
    setV("kc", String(S.kc));
    dispatchMC();
  }
  const A = K.analyse(ctx, cube, S.flt);
  if (!S.heat) { let mx = -Infinity; for (let m = 0; m < ctx.ND; m++) for (let t = TW0; t < TW1; t++) mx = Math.max(mx, cube[m * N + t]); S.heat = { vmin: -2 * A.sn[ctx.i0], vmax: 0.9 * mx }; }
  S.stack[j] = { j, cube, A };
  drawCard(j); draw("cv-sd"); setStatus();
  if (j === S.sel) selectRealisation(j);
  if (S.stack.every(Boolean)) schedule("mc");
}

// ---------------------------------------------------------------- selection
function selectRealisation(j) {
  const st = S.stack[j]; if (!st) return;
  const ctx = S.ctx, N = ctx.N, NCH = ctx.NCH, flt = S.flt, A = st.A;
  S.sel = j; S.order = [j, ...S.order.filter(x => x !== j)]; layoutDeck();
  const ch = K.genChannels(ctx, K.seedFor(S.params.seed, j), true);
  const FI = K.chanFFT(ctx, ch.I), FS = K.chanFFT(ctx, ch.Sg);
  const s0 = new Float64Array(N); for (let c = 0; c < NCH; c++) for (let t = 0; t < N; t++) s0[t] += ch.Sg[c * N + t];
  const cs0 = ctx.dct(s0);
  // noise-free structure ||a(d)|| = ||F C^T s_d|| at every trial DM, and its curvature at the true DM
  const sCube = K.dedisperse(ctx, FS), anorm = new Float64Array(ctx.ND), rowS = new Float64Array(N);
  for (let m = 0; m < ctx.ND; m++) { for (let t = 0; t < N; t++) rowS[t] = sCube[m * N + t]; const c = ctx.dct(rowS); let a2 = 0; for (let k = 0; k < N; k++) a2 += (flt.F[k] * c[k]) ** 2; anorm[m] = Math.sqrt(a2); }
  const axs = [], ays = []; for (let m = ctx.i0 - 3; m <= ctx.i0 + 3; m++) { axs.push(ctx.dms[m]); ays.push(anorm[m]); }
  const aq = K.quadfit(axs, ays, 0);
  // per-row sigma with local (self-noise) variance
  let sigL = null;
  if (ctx.selfnoise) {
    sigL = new Float64Array(ctx.ND);
    for (let m = 0; m < ctx.ND; m++) { const c = A.C.subarray(m * N, (m + 1) * N), w = K.weights(ctx, flt, c), lv = K.localVar(ctx, c); let s = 0; for (let t = 0; t < N; t++) s += w[t] * w[t] * lv[t]; sigL[m] = Math.sqrt(s); }
  }
  // stage 5: the data recipe at this draw's measured peak d-hat (derivation step 6h)
  const dm = K.sigmaDMdata(ctx, flt, FI, K.chanVar(ctx, ch.I), A.S, A.dhat); A.sigDM = dm.sigDM;
  const wHat = K.wRow(ctx, flt, FI, A.dhat), wdHat = cdiff(wHat).map(v => v / ctx.dt);
  const wpTop = K.wprimeChan(ctx, dm.wp, dm.wm, A.dhat, dm.delta, 0), wpBot = K.wprimeChan(ctx, dm.wp, dm.wm, A.dhat, dm.delta, NCH - 1);
  S.sd = { j, ch, FI, FS, A, cube: st.cube, s0, cs0, sigL, dm, wHat, wdHat, wpTop, wpBot, wCache: new Map(), ww: support(wHat, 0.15, 30), truth0: truthAt(cs0, A.sn[ctx.i0]), anorm, aq };
  setV("dmW", dm.W.toFixed(4)); setV("dmDelta", dm.delta.toFixed(4)); setV("sigDMsel", sig(dm.sigDM, 3));
  $("#s2-title").textContent = `Realisation #${j + 1}: I(t, ΔDM)`;
  if (S.row == null) S.row = A.dpk;
  setRow(S.row);
  draw("cv-sd"); draw("cv-cube"); draw("cv-sprof"); drawStep5(); schedule("mc");
}
function truthAt(cs, sn) {
  const { F, R } = S.flt, N = cs.length;
  let a = 0; for (let k = 0; k < N; k++) a += (F[k] * cs[k]) ** 2; a = Math.sqrt(a);
  let Fu2 = 0, T = 0, Rb = 0;
  for (let k = 0; k < N; k++) { const u = F[k] * cs[k] / a; Fu2 += (F[k] * u) ** 2; T += R[k] * R[k]; Rb += (R[k] * cs[k]) ** 2; }
  const Neff = T / Fu2, Gam = Rb / (sn * sn * T);
  return { a, Fu: Math.sqrt(Fu2), T, Neff, Gam };
}
function setRow(m) {
  const sd = S.sd; if (!sd) return;
  const ctx = S.ctx, N = ctx.N, flt = S.flt, A = sd.A;
  m = Math.max(0, Math.min(ctx.ND - 1, m)); S.row = m; const d = ctx.dms[m];
  const c = A.C.subarray(m * N, (m + 1) * N), i = Float64Array.from(sd.cube.subarray(m * N, (m + 1) * N));
  const s = K.dedisperseRow(ctx, sd.FS, d), cs = ctx.dct(s), n = i.map((v, t) => v - s[t]), nu = ctx.dct(n);
  const ti = ctx.idct(c.map((v, k) => v * flt.f[k])), D1ti = diff(ti);
  const aN = nrm(D1ti), ah = D1ti.map(v => v / aN);   // a-hat estimated as D1 S i / ||D1 S i||
  const d1t = D1T(ah), w = K.weights(ctx, flt, c);
  const eps = dot(w, n), sn = A.sn[m];
  let en = 0; for (let k = 0; k < N; k++) en += (flt.F[k] * nu[k]) ** 2; en = Math.sqrt(en);
  const tr = truthAt(cs, sn);
  const lv = ctx.selfnoise ? K.localVar(ctx, c) : null;
  const wsv = new Float64Array(N); for (let k = 0; k < N; k++) wsv[k] = flt.F[k] * flt.F[k] * cs[k] / tr.a;
  const wT = ctx.idct(wsv), epsT = dot(wT, n), second = (en * en - epsT * epsT) / (2 * tr.a);
  S.rd = { m, d, c, i, s, cs, n, nu, ti, D1ti, ah, d1t, w, wT, eps, epsT, second, en, tr, lv, sn, ww: support(w, 0.15, 30) };
  $("#r-dm").value = m;
  $("#s3-title").textContent = `Row ΔDM = ${fdm(d)}: the dynamic spectrum behind it`;
  setV("rowDM", fdm(d) + " pc cm⁻³"); setV("rowS", fx(A.S[m], 3)); setV("sn", fx(sn, 3));
  setV("rowSpread", fx(Math.abs(d) * (ctx.tau[ctx.NCH - 1] - ctx.tau[0]), 1) + " μs");
  setV("anorm", fx(tr.a, 3)); setV("enorm", fx(en, 3)); setV("eps", fx(epsT, 3)); setV("epsEst", fx(eps, 3)); setV("second", fx(second, 3)); setV("SmA", fx(A.S[m] - tr.a, 3)); setV("sigT", fx(lv ? Math.sqrt(dot(wT.map((v, t) => v * v), lv)) : sn * tr.Fu, 3));
  setV("sigRow", fx(ctx.selfnoise ? sd.sigL[m] : A.sig[m], 3)); setV("GammaRow", fx(tr.Gam, 2));
  draw("cv-cube"); draw("cv-sprof"); draw("cv-dyn"); draw("cv-row"); drawStep4();
}

// ---------------------------------------------------------------- deck
const deck = $("#deck");
function buildDeck() {
  deck.innerHTML = "";
  for (let j = 0; j < NSTACK; j++) {
    const b = document.createElement("button"); b.type = "button"; b.className = "card pending"; b.dataset.j = j; b.setAttribute("role", "option");
    b.setAttribute("aria-label", `Realisation ${j + 1}`);
    b.innerHTML = `<span class="tag"><b>#${j + 1}</b><span>dealing…</span></span><canvas></canvas>`;
    b.addEventListener("click", () => { if (S.stack[j]) selectRealisation(j); });
    b.addEventListener("pointerenter", () => { S.hoverCard = j; draw("cv-sd"); });
    b.addEventListener("pointerleave", () => { S.hoverCard = -1; draw("cv-sd"); });
    b.addEventListener("focus", () => { S.hoverCard = j; draw("cv-sd"); });
    b.addEventListener("blur", () => { S.hoverCard = -1; draw("cv-sd"); });
    deck.appendChild(b);
  }
  layoutDeck();
}
let deckGeom = null;
function layoutDeck() {
  const W = deck.clientWidth || 500, n = NSTACK, dx = Math.round(Math.min(22, W * 0.035)), dy = 24;
  const cw = W - (n - 1) * dx, ch = Math.round(Math.min(250, Math.max(150, cw * 0.5)));
  deck.style.height = (ch + (n - 1) * dy) + "px";
  const changed = !deckGeom || deckGeom.cw !== cw || deckGeom.ch !== ch; deckGeom = { cw, ch, dx, dy };
  for (const b of deck.children) {
    const j = +b.dataset.j, i = S.order.indexOf(j);
    b.style.width = cw + "px"; b.style.height = ch + "px";
    b.style.transform = `translate(${i * dx}px, ${(n - 1 - i) * dy}px)`; b.style.zIndex = String(n - i);
    b.classList.toggle("sel", j === S.sel); b.setAttribute("aria-selected", j === S.sel ? "true" : "false");
  }
  if (changed) for (let j = 0; j < NSTACK; j++) drawCard(j);
}
function drawCard(j) {
  const b = deck.children[j]; if (!b) return; const st = S.stack[j], cv = b.querySelector("canvas");
  if (!deckGeom) return;
  cv.dataset.h = deckGeom.ch - 23; cv.style.width = deckGeom.cw + "px";
  if (!st) return;
  b.classList.remove("pending");
  b.querySelector(".tag span").textContent = `d̂ = ${fdm(st.A.dhat)}`;
  const p = prep(cv); if (!p) return;
  const ctx = S.ctx, N = ctx.N, ND = ctx.ND, cube = st.cube;
  const img = heatImg((r, q) => cube[(ND - 1 - r) * N + TW0 + q], ND, TW, S.heat.vmin, S.heat.vmax);
  p.g.imageSmoothingEnabled = false; p.g.drawImage(img, 0, 0, p.w, p.h);
}

// ---------------------------------------------------------------- figures
const FIG = {};
function draw(id) { const cv = document.getElementById(id); if (!cv || !FIG[id]) return; if (!cv.clientWidth) return; try { FIG[id](cv); } catch (e) { console.error(id, e); } }
function clearViews() { for (const id of Object.keys(FIG)) { const cv = document.getElementById(id); if (cv && cv.clientWidth) { const p = prep(cv); if (p) { p.g.fillStyle = P.muted; p.g.font = `12px ${P.mono}`; p.g.textAlign = "center"; p.g.fillText("computing…", p.w / 2, p.h / 2); } } } }

// stage 1: S(d) per card
FIG["cv-sd"] = cv => {
  const have = S.stack.filter(Boolean); if (!have.length) return;
  const ctx = S.ctx, dms = ctx.dms; let lo = Infinity, hi = -Infinity;
  for (const st of have) for (const v of st.A.S) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = (hi - lo) * 0.08, ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  ax.xr(dms[0], dms[ctx.ND - 1]).yr(lo - pad, hi + pad).frame({ xlabel: "ΔDM (pc cm⁻³)", ylabel: "𝒮 " });
  ax.clip(); ax.vline(0, { c: P.muted, lw: 1, dash: [2, 3] });
  for (const st of have) if (st.j !== S.sel && st.j !== S.hoverCard) ax.line(dms, st.A.S, { c: P.mc, lw: 1, a: 0.7 });
  if (S.hoverCard >= 0 && S.stack[S.hoverCard] && S.hoverCard !== S.sel) ax.line(dms, S.stack[S.hoverCard].A.S, { c: P.accent, lw: 1.6 });
  if (S.stack[S.sel]) { const A = S.stack[S.sel].A; ax.line(dms, A.S, { c: P.ink, lw: 2 }); ax.dots([A.dhat], [A.S[A.dpk]], { c: P.ink, r: 3.5, ring: P.surface }); }
  // MC rug
  const g = ax.g; g.strokeStyle = P.mc; g.globalAlpha = 0.5; g.lineWidth = 1; g.beginPath();
  for (const r of S.mc) { const x = Math.round(ax.X(r.dhat)) + 0.5; g.moveTo(x, ax.py0); g.lineTo(x, ax.py0 - 6); } g.stroke(); g.globalAlpha = 1;
  ax.unclip();
  cv._ax = ax;
};
bindHover($("#cv-sd"), (ev, x, y) => {
  const ax = $("#cv-sd")._ax; if (!ax || x == null) { if (S.hoverCard !== -1 && x == null) { S.hoverCard = -1; draw("cv-sd"); } return; }
  const d = ax.iX(x), yv = ax.iY(y), m = Math.round((d - S.ctx.dms[0]) / 0.001); if (m < 0 || m >= S.ctx.ND) return tip(ev, null);
  let best = -1, bd = Infinity; for (const st of S.stack) if (st) { const dd = Math.abs(st.A.S[m] - yv); if (dd < bd) { bd = dd; best = st.j; } }
  if (best !== S.hoverCard) { S.hoverCard = best; draw("cv-sd"); }
  const A = S.stack[best].A; tip(ev, `card #${best + 1}<br>d̂ = ${fdm(A.dhat)}<br>𝒮 (ΔDM = ${fdm(S.ctx.dms[m])}) = ${fx(A.S[m], 3)}<br><span style="opacity:.7">click to select</span>`);
}, () => { if (S.hoverCard >= 0) selectRealisation(S.hoverCard); });

// MC aggregate
function mcStats() {
  const R = S.mc, n = R.length; if (n < 3) return null;
  const ctx = S.ctx, sn = ctx.selfnoise, i0 = ctx.i0, ND = ctx.ND;
  const g = k => R.map(r => r[k]);
  const ok = R.filter(r => r.lo != null && r.hi != null);
  let ref = S.sd ? S.sd.anorm : null;   // noise-free ||a(d)|| of the selected realisation; MC mean only as a fallback
  if (!ref) { ref = new Float64Array(ND); for (const r of R) for (let m = 0; m < ND; m++) ref[m] += r.S[m] / n; }
  const h = ctx.dms[i0 + 1] - ctx.dms[i0];
  const slopes = R.map(r => ((r.S[i0 + 1] - ref[i0 + 1]) - (r.S[i0 - 1] - ref[i0 - 1])) / (2 * h));
  const xs = [], ys = []; for (let m = i0 - 3; m <= i0 + 3; m++) { xs.push(ctx.dms[m]); ys.push(ref[m]); }
  const q = K.quadfit(xs, ys, 0);
  return {
    n, mcS: K.std(g("S0")), mcEps: K.std(g("eps")), sig: K.median(g(sn ? "sigL" : "sigW")), s23: K.median(g("s23abs")), s23rad: K.median(g("s23rad")), en: K.median(g("en")),
    mcDM: K.std(g("dhat")), mDM: mean(g("dhat")), sigDM: K.median(g("sigDM")), hw: ok.length ? K.median(ok.map(r => 0.5 * (r.hi - r.lo))) : null,
    lo: ok.length ? K.median(ok.map(r => r.lo)) : null, hi: ok.length ? K.median(ok.map(r => r.hi)) : null,
    ref, slopes, slopeMC: K.std(slopes), slopeData: K.median(g("slope")), slopeTrue: K.median(g("slopeT")), sigDMt: K.median(g("sigDMt")), SppT: K.median(g("SppT")), q, Sbpp: 2 * q[2]
  };
}
function chip(ratio) { if (!isFinite(ratio)) return ""; const good = ratio > 0.67 && ratio < 1.5; return `<span class="chip ${good ? "good" : "bad"}">×${ratio < 10 ? ratio.toFixed(2) : ratio.toFixed(1)}</span>`; }
function updateSummary() {
  const st = S.st = mcStats(); setStatus();
  $("#hist-n").textContent = S.mc.length;
  if (!st) return;
  $("#b-s-mc").textContent = sig(st.mcS, 3);
  $("#b-s-new").innerHTML = sig(st.sig, 3) + chip(st.sig / st.mcS);
  $("#b-s-s23").innerHTML = sig(st.s23, 3) + chip(st.s23 / st.mcS);
  $("#b-d-mc").textContent = sig(st.mcDM, 3);
  $("#b-d-new").innerHTML = sig(st.sigDM, 3) + chip(st.sigDM / st.mcDM);
  const edge = st.lo != null && (st.lo <= S.ctx.dms[0] + 1e-9 || st.hi >= S.ctx.dms[S.ctx.ND - 1] - 1e-9), ge = edge ? "≥ " : "";
  $("#b-d-s23").innerHTML = (st.hw != null ? ge + sig(st.hw, 3) + ` <span class='muted' style='font-family:var(--font-body);font-size:12px'>half-width${edge ? ", hits grid edge" : ""}</span>` + chip(st.hw / st.mcDM).replace("×", ge ? "≥×" : "×") : "–");
  setV("mcEps", fx(st.mcEps, 3)); setV("mcS", fx(st.mcS, 3)); setV("mcSig", fx(st.sig, 3));
  setV("mcDM", sig(st.mcDM, 3)); setV("sigDM", sig(st.sigDM, 3)); setV("sigDMt", sig(st.sigDMt, 3)); setV("slopeTrue", fx(st.slopeTrue, 2)); setV("s23hw", st.hw != null ? sig(st.hw, 3) : "–");
  setV("slopeMC", fx(st.slopeMC, 2)); setV("slopeData", fx(st.slopeData, 2));
  setV("Sbarpp", fx(st.Sbpp, 0));
  const sd = S.sd;
  if (sd) {
    const t0 = sd.truth0, A = sd.A, i0 = S.ctx.i0, normf = A.S[i0] / A.FLn[i0];
    setV("Neff0", fx(t0.Neff, 2)); setV("sNeff0", fx(Math.sqrt(t0.Neff), 2)); setV("sGam0", fx(Math.sqrt(1 + t0.Gam), 2)); setV("normf", fx(normf, 2));
    const pred = Math.sqrt(t0.Neff * (1 + t0.Gam)) * normf, meas = st.s23 / st.mcS;
    setV("predRatio", fx(pred, 1)); setV("measRatio", fx(meas, 1));
    $("#decomp").innerHTML = `S23's σ(𝒮 ) overshoot at the true DM: √N<sub>eff</sub> = ${fx(Math.sqrt(t0.Neff), 2)} (dimensionality) × √(1+Γ) = ${fx(Math.sqrt(1 + t0.Gam), 2)} (burst leakage) × ${fx(normf, 2)} (double-smoothed normalisation) = ×${fx(pred, 1)} predicted, ×${fx(meas, 1)} measured. The Eq. 11 DM interval inherits all three.`;
  }
  draw("cv-hist"); draw("cv-sd"); draw("cv-sprof");
  if (S.step4 === 4 || S.step4 === 5 || S.step4 === 6) drawStep4();
  drawStep5();
}

function drawHist(cv, big) {
  const st = S.st; const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  const R = S.mc, bw = 0.001, x0 = -0.04, x1 = 0.04, nb = Math.round((x1 - x0) / bw), cnt = new Float64Array(nb);
  for (const r of R) { const b = Math.floor((r.dhat - x0) / bw); if (b >= 0 && b < nb) cnt[b]++; }
  let ym = 1; for (const c of cnt) ym = Math.max(ym, c);
  const sigd = st ? st.sigDM : null, gauss = sigd ? R.length * bw / (Math.sqrt(2 * Math.PI) * sigd) : 0;
  ym = Math.max(ym, gauss) * 1.12;
  ax.xr(x0, x1).yr(0, ym).frame({ xlabel: "recovered ΔDM (pc cm⁻³)", ylabel: "draws" });
  ax.clip();
  if (st && st.lo != null) ax.rectD(st.lo, st.hi, 0, ym, { c: P.s23, a: 0.13 });
  ax.bars(Array.from(cnt, (_, b) => x0 + (b + 0.5) * bw), Array.from(cnt), { bw, c: P.mc, gap: 2 });
  ax.vline(0, { c: P.muted, lw: 1, dash: [2, 3] });
  if (sigd) { const xs = [], ys = []; for (let i = 0; i <= 400; i++) { const x = x0 + (x1 - x0) * i / 400; xs.push(x); ys.push(gauss * Math.exp(-0.5 * x * x / sigd / sigd)); } ax.line(xs, ys, { c: P.accent, lw: 2 }); }
  if (big && st) {
    const yb = ym * 0.93;
    ax.line([-st.mcDM, st.mcDM], [yb, yb], { c: P.ink, lw: 2 }); ax.text(`MC ±1σ ${sig(st.mcDM, 2)}`, st.mcDM, yb, { al: "left", c: P.ink }).g;
    ax.line([-sigd, sigd], [yb * 0.9, yb * 0.9], { c: P.accent, lw: 2 }); ax.text(` σ_{DM} ${sig(sigd, 2)}`, sigd, yb * 0.9, { c: P.accent });
    if (st.lo != null) ax.text(`S23 [${fdm(st.lo)}, ${fdm(st.hi)}]`, st.hi, ym * 0.6, { al: "right", c: P.s23 });
  }
  ax.unclip(); cv._ax = ax; cv._cnt = cnt;
}
FIG["cv-hist"] = cv => drawHist(cv, false);
bindHover($("#cv-hist"), (ev, x) => { const cv = $("#cv-hist"), ax = cv._ax; if (!ax || x == null) return tip(ev, null); const b = Math.floor((ax.iX(x) + 0.04) / 0.001); if (b < 0 || b >= 80) return tip(ev, null); tip(ev, `ΔDM ${fdm(-0.04 + b * 0.001)} to ${fdm(-0.04 + (b + 1) * 0.001)}<br>${cv._cnt[b]} draws`); });

// stage 2: cube + S profile
const RM = { t: 10, b: 36 };
FIG["cv-cube"] = cv => {
  const sd = S.sd; if (!sd) return; const ctx = S.ctx, N = ctx.N, ND = ctx.ND, cube = sd.cube;
  const ax = new Ax(cv, { l: 58, r: 18, t: RM.t, b: RM.b }); if (!ax.ok) return;
  ax.xr(TW0, TW1).yr(ctx.dms[0] - 0.0005, ctx.dms[ND - 1] + 0.0005);
  blit(ax, heatImg((r, q) => cube[(ND - 1 - r) * N + TW0 + q], ND, TW, S.heat.vmin, S.heat.vmax), false);
  ax.frame({ xlabel: "time (μs)", ylabel: "ΔDM (pc cm⁻³)", grid: false });
  const g = ax.g, rowBox = (m, c, lw) => { const y0 = ax.Y(ctx.dms[m] + 0.0005), y1 = ax.Y(ctx.dms[m] - 0.0005); g.strokeStyle = c; g.lineWidth = lw; g.strokeRect(ax.px0 + 0.5, y0, ax.px1 - ax.px0 - 1, y1 - y0); };
  if (S.hoverRow >= 0 && S.hoverRow !== S.row) rowBox(S.hoverRow, P.accent, 1.2);
  if (S.row != null) rowBox(S.row, P.ink, 2);
  // true-DM marker
  const yt = ax.Y(0); g.fillStyle = P.ink; g.beginPath(); g.moveTo(ax.px0 - 1, yt); g.lineTo(ax.px0 - 7, yt - 4); g.lineTo(ax.px0 - 7, yt + 4); g.closePath(); g.fill();
  cv._ax = ax;
};
FIG["cv-sprof"] = cv => {
  const sd = S.sd; if (!sd) return; const ctx = S.ctx, A = sd.A, dms = ctx.dms, ND = ctx.ND;
  const sg = ctx.selfnoise ? sd.sigL : A.sig, adj = A.S.map((v, m) => v * (1 + A.rel[m]));
  let lo = Infinity, hi = -Infinity; for (let m = 0; m < ND; m++) { lo = Math.min(lo, A.S[m] - sg[m]); hi = Math.max(hi, adj[m]); }
  const pad = (hi - lo) * 0.06;
  const ax = new Ax(cv, { l: 14, r: 34, t: RM.t, b: RM.b }); if (!ax.ok) return;
  ax.xr(lo - pad, hi + pad).yr(dms[0] - 0.0005, dms[ND - 1] + 0.0005).frame({ xlabel: "𝒮 ", noYlab: true, xgrid: true, grid: true, yt: niceTicks(-0.04, 0.04, 8) });
  ax.clip();
  // corrected band (x = S +- sigma, y = dm) drawn as polygon
  const g = ax.g; g.fillStyle = P.accent; g.globalAlpha = 0.18; g.beginPath();
  for (let m = 0; m < ND; m++) { const x = ax.X(A.S[m] + sg[m]), y = ax.Y(dms[m]); m ? g.lineTo(x, y) : g.moveTo(x, y); }
  for (let m = ND - 1; m >= 0; m--) g.lineTo(ax.X(A.S[m] - sg[m]), ax.Y(dms[m])); g.closePath(); g.fill(); g.globalAlpha = 1;
  ax.vline(A.S[A.dpk], { c: P.muted, lw: 1, dash: [2, 3] });
  ax.line(i => adj[i], Array.from(dms), { c: P.s23, lw: 1.6 });
  ax.line(i => A.S[i], Array.from(dms), { c: P.ink, lw: 2 });
  if (S.row != null) ax.hline(dms[S.row], { c: P.ink, lw: 1, a: 0.6 });
  ax.hline(0, { c: P.muted, lw: 1, dash: [2, 3], a: 0.8 });
  ax.unclip();
  // interval bars at right margin
  const xr1 = ax.px1 + 9, xr2 = ax.px1 + 21;
  g.lineCap = "butt";
  if (A.lo != null) { g.strokeStyle = P.s23; g.lineWidth = 5; g.beginPath(); g.moveTo(xr1, ax.Y(A.lo)); g.lineTo(xr1, ax.Y(A.hi)); g.stroke(); }
  const sdm = A.sigDM; g.strokeStyle = P.accent; g.lineWidth = 5; g.beginPath(); g.moveTo(xr2, ax.Y(A.dhat - sdm)); g.lineTo(xr2, ax.Y(A.dhat + sdm)); g.stroke();
  g.fillStyle = P.ink; g.fillRect(xr2 - 5, ax.Y(A.dhat) - 1, 10, 2);
  g.lineCap = "round";
  cv._ax = ax;
};
function rowFromY(ax, y) { const d = ax.iY(y); const m = Math.round((d - S.ctx.dms[0]) / 0.001); return m >= 0 && m < S.ctx.ND ? m : -1; }
for (const id of ["cv-cube", "cv-sprof"]) bindHover($("#" + id), (ev, x, y) => {
  const ax = $("#" + id)._ax; if (!ax || y == null || !S.sd) { if (S.hoverRow !== -1) { S.hoverRow = -1; draw("cv-cube"); } return; }
  const m = rowFromY(ax, y); if (m !== S.hoverRow) { S.hoverRow = m; draw("cv-cube"); }
  if (m < 0) return tip(ev, null);
  const A = S.sd.A, ctx = S.ctx;
  let extra = ""; if (id === "cv-cube" && x >= ax.px0 && x <= ax.px1) { const t = Math.round(ax.iX(x)); extra = `<br>t = ${t} μs · I = ${fx(S.sd.cube[m * ctx.N + t], 1)}`; }
  tip(ev, `ΔDM = ${fdm(ctx.dms[m])}<br>𝒮  = ${fx(A.S[m], 3)} · σ = ${fx((ctx.selfnoise ? S.sd.sigL : A.sig)[m], 3)}<br>S23 𝒮 (1+r) = ${fx(A.S[m] * (1 + A.rel[m]), 3)}${extra}<br><span style="opacity:.7">click to open this row</span>`);
}, (ev, x, y) => { const ax = $("#" + id)._ax; if (!ax) return; const m = rowFromY(ax, y); if (m >= 0) { setRow(m); scrollToEl($("#s3")); } });
function scrollToEl(el) { const r = el.getBoundingClientRect(); if (r.top > innerHeight * 0.85 || r.bottom < 0) el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }); }

// stage 3
FIG["cv-dyn"] = cv => {
  const sd = S.sd, rd = S.rd; if (!sd || !rd) return; const ctx = S.ctx, NCH = ctx.NCH;
  const raw = K.dynspec(ctx, sd.FI, rd.d, TW0, TW1), smooth = $("#c-smooth").checked;
  let data = raw;
  if (smooth) { data = new Float32Array(raw.length); for (let c = 0; c < NCH; c++) for (let t = 0; t < TW; t++) { let s = 0, n = 0; for (let dc = -1; dc <= 2; dc++) for (let dt = -1; dt <= 1; dt++) { const cc = c + dc, tt = t + dt; if (cc < 0 || cc >= NCH || tt < 0 || tt >= TW) continue; s += raw[cc * TW + tt]; n++; } data[c * TW + t] = s / n; } }
  const sdv = Math.sqrt(mean(Array.from(data.slice(0, NCH * 40)).map(v => v * v)));
  const vmax = quantile(data, 0.997);
  const ax = new Ax(cv, { l: 58, r: 12, t: 10, b: 30 }); if (!ax.ok) return;
  const fhi = ctx.freqs[0] + 2, flo = ctx.freqs[NCH - 1] - 2;
  ax.xr(TW0, TW1).yr(flo, fhi);
  blit(ax, heatImg((r, q) => data[r * TW + q], NCH, TW, -2 * sdv, vmax), false);
  ax.frame({ ylabel: "ν (MHz)", grid: false, noXlab: false });
  // delay track of the first burst component
  const xs = [], ys = []; for (let c = 0; c < NCH; c++) { xs.push(1000 - rd.d * ctx.tau[c]); ys.push(ctx.freqs[c]); }
  ax.clip(); ax.line(xs, ys, { c: P.deriv, lw: 1.6, dash: [5, 4] }); ax.unclip();
  cv._ax = ax; cv._raw = raw;
};
bindHover($("#cv-dyn"), (ev, x, y) => {
  const cv = $("#cv-dyn"), ax = cv._ax; if (!ax || x == null || x < ax.px0 || x > ax.px1 || y < ax.py1 || y > ax.py0) return tip(ev, null);
  const ctx = S.ctx, c = Math.min(ctx.NCH - 1, Math.max(0, Math.floor((ax.py0 - y) / (ax.py0 - ax.py1) * ctx.NCH))), ch = ctx.NCH - 1 - c, t = Math.floor(ax.iX(x));
  tip(ev, `channel ${ch} · ν = ${fx(ctx.freqs[ch], 0)} MHz<br>τ<sub>ν</sub> = ${fx(ctx.tau[ch], 0)} μs per pc cm⁻³<br>advanced here by ΔDM·τ<sub>ν</sub> = ${fx(S.rd.d * ctx.tau[ch], 1)} μs<br>t = ${t} μs · I = ${fx(cv._raw[ch * TW + (t - TW0)], 2)}`);
});
FIG["cv-row"] = cv => {
  const rd = S.rd; if (!rd) return;
  const ax = new Ax(cv, { l: 58, r: 12, t: 8, b: 34 }); if (!ax.ok) return;
  const iw = sliceW(rd.i, TW0, TW1), sw = sliceW(rd.s, TW0, TW1);
  let lo = Math.min(...iw), hi = Math.max(...iw);
  ax.xr(TW0, TW1).yr(lo, hi).frame({ xlabel: "time (μs)", ylabel: "I" });
  ax.clip(); ax.line(i => TW0 + i, iw, { c: P.mc, lw: 1 }); ax.line(i => TW0 + i, sw, { c: P.sig, lw: 2 }); ax.unclip();
};

// ---------------------------------------------------------------- stage 4
function drawStep4() { const k = S.step4; ({ 0: ["f40"], 1: ["f41"], 2: ["f42a", "f42b"], 3: ["f43a", "f43b", "f43c", "f43d"], 4: ["f44a", "f44b", "f44c"], 5: ["f45"], 6: ["f46"] })[k].forEach(draw); }
const kArr = n => Array.from({ length: n }, (_, j) => j + 1);
FIG.f40 = cv => {
  const rd = S.rd; if (!rd) return; const N = S.ctx.N;
  const ac = Array.from(rd.c, Math.abs), as = Array.from(rd.cs, Math.abs);
  const ymin = rd.sn * 2e-3, ymax = Math.max(...ac) * 2;
  const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  ax.xr(1, N, true).yr(ymin, ymax, true).frame({ xlabel: "DCT mode k", ylabel: "|c_{k}|", xgrid: true });
  ax.clip();
  ax.rectD(N / 4, N, ymin, ymax, { c: P.accent, a: 0.07 });
  ax.line(j => j + 1, ac, { c: P.mc, lw: 1, a: 0.85 });
  // noiseless burst split by parity: odd k filled, even k hollow
  { const ko = [], ao = []; for (let j = 0; j < N; j += 2) { ko.push(j + 1); ao.push(as[j]); } ax.dots(ko, ao, { c: P.sig, r: 1.8 });
    const g = ax.g; g.strokeStyle = P.sig; g.lineWidth = 1.1;
    for (let j = 1; j < N; j += 2) { const v = as[j]; if (!(v > ymin)) continue; g.beginPath(); g.arc(ax.X(j + 1), ax.Y(v), 2.1, 0, 2 * Math.PI); g.stroke(); } }
  ax.hline(rd.sn, { c: P.accent, lw: 1.6, dash: [6, 4] });
  ax.vline(S.kc, { c: P.muted, lw: 1.2, dash: [2, 3] }); ax.text(` k_{c} = ${S.kc}`, S.kc, ymax / 3, { c: P["ink-2"] });
  ax.unclip(); cv._ax = ax;
};
bindHover($("#f40"), (ev, x) => { const cv = $("#f40"), ax = cv._ax; if (!ax || x == null || !S.rd) return tip(ev, null); const k = Math.round(ax.iX(x)); if (k < 1 || k > S.ctx.N) return tip(ev, null); tip(ev, `k = ${k}<br>|c<sub>k</sub>| = ${sig(Math.abs(S.rd.c[k - 1]), 3)}<br>|c<sub>s,k</sub>| = ${sig(Math.abs(S.rd.cs[k - 1]), 3)}<br>f<sub>k</sub> = ${sig(S.flt.f[k - 1], 3)} · h<sub>k</sub>f<sub>k</sub> = ${sig(S.flt.F[k - 1], 3)}`); });
FIG.f41 = cv => {
  if (!S.flt) return; const N = S.ctx.N, { f, h, F, R } = S.flt;
  const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  ax.xr(1, N, true).yr(1e-6, 4, true).frame({ xlabel: "DCT mode k", ylabel: "response", xgrid: true });
  ax.clip();
  ax.line(j => j + 1, f, { c: P.sig, lw: 2 }); ax.line(j => j + 1, h, { c: P.deriv, lw: 2 });
  ax.line(j => j + 1, F, { c: P.accent, lw: 2.4 }); ax.line(j => j + 1, R, { c: P.s23, lw: 1.8, dash: [6, 4] });
  ax.vline(S.kc, { c: P.muted, lw: 1.2, dash: [2, 3] }); ax.text(` k_{c}`, S.kc, 2, { c: P["ink-2"] });
  ax.unclip();
};
function tplot(cv, series, o) {
  o = o || {}; const ax = new Ax(cv, Object.assign({ l: 58, r: 12, t: 8, b: o.noX ? 12 : 34 }, o.m || {})); if (!ax.ok) return null;
  const a = o.t0 || TW0, b = o.t1 || TW1; let lo = Infinity, hi = -Infinity;
  for (const s of series) if (!s.noscale) for (let t = a; t < b; t++) { const v = s.y[t]; if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  if (o.sym) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  const pad = (hi - lo) * (o.head || 0.06); ax.xr(a, b).yr(lo - pad, hi + pad);
  ax.frame({ xlabel: o.noX ? null : (o.xlabel || "time (μs)"), nx: o.nx, ylabel: o.ylabel, noXlab: !!o.noX, noYlab: !!o.noYlab, ny: o.ny });
  ax.clip(); if (o.zero) ax.hline(0, { c: P.rule, lw: 1 });
  for (const s of series) { if (s.fill) ax.band(i => a + i, new Float64Array(b - a), Array.from(s.y.slice(a, b)), { c: s.c, a: s.fill }); ax.line(i => a + i, Array.from(s.y.slice(a, b)), s); }
  ax.unclip(); return ax;
}
FIG.f42a = cv => { const rd = S.rd; if (!rd) return; tplot(cv, [{ y: rd.i, c: P.mc, lw: 1 }, { y: rd.ti, c: P.ink, lw: 2 }, { y: rd.s, c: P.sig, lw: 1.6, dash: [5, 4] }], { ylabel: "I", noX: true }); };
FIG.f42b = cv => { const rd = S.rd; if (!rd) return; tplot(cv, [{ y: rd.D1ti, c: P.ink, lw: 1.8, fill: 0.12 }], { ylabel: "D₁ĩ", zero: true, sym: true }); };
FIG.f43a = cv => { const rd = S.rd; if (!rd) return; tplot(cv, [{ y: rd.ah, c: P.ink, lw: 1.8 }], { t0: rd.ww[0], t1: rd.ww[1], xlabel: "t (μs)", nx: 3, zero: true, sym: true, m: { l: 50, r: 10, t: 8, b: 32 }, ny: 3, ylabel: "â" }); };
FIG.f43b = cv => { const rd = S.rd; if (!rd) return; tplot(cv, [{ y: rd.d1t, c: P.deriv, lw: 1.8 }], { t0: rd.ww[0], t1: rd.ww[1], xlabel: "t (μs)", nx: 3, zero: true, sym: true, m: { l: 50, r: 10, t: 8, b: 32 }, ny: 3, ylabel: "âᵀD₁" }); };
FIG.f43c = cv => { const rd = S.rd; if (!rd) return; tplot(cv, [{ y: rd.w, c: P.w, lw: 1.8 }], { t0: rd.ww[0], t1: rd.ww[1], xlabel: "t (μs)", nx: 3, zero: true, sym: true, m: { l: 50, r: 10, t: 8, b: 32 }, ny: 3, ylabel: "w" }); };
FIG.f43d = cv => {
  const rd = S.rd; if (!rd) return;
  const [a, b] = rd.ww; let wm = 0, sm = 0; for (let t = a; t < b; t++) { wm = Math.max(wm, Math.abs(rd.w[t])); sm = Math.max(sm, Math.abs(rd.s[t])); }
  const sS = rd.s.map(v => v / sm * wm);
  const ax = tplot(cv, [{ y: rd.w, c: P.w, lw: 2.2, fill: 0.13 }, { y: rd.wT, c: P.w, lw: 1.3, dash: [2, 3] }, { y: sS, c: P.sig, lw: 1.5, dash: [5, 4], noscale: true }], { t0: a, t1: b, ylabel: "w", zero: true, sym: true, head: 0.2 });
  if (!ax) return;
  let tmax = a, tmin = a, m = 0; for (let t = a; t < b; t++) { if (rd.w[t] > rd.w[tmax]) tmax = t; if (rd.w[t] < rd.w[tmin]) tmin = t; m = Math.max(m, Math.abs(rd.w[t]), Math.abs(rd.wT[t])); }
  // labels sit in the headroom above/below every curve, clamped inside the plot, with a short leader to the extremum
  const lab = (txt, t, yv, ylab) => {
    const g = ax.g, wpx = richWidth(g, txt), x = Math.min(Math.max(ax.X(t), ax.px0 + wpx / 2 + 4), ax.px1 - wpx / 2 - 4), y = ax.Y(ylab);
    g.strokeStyle = P.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(ax.X(t), ax.Y(yv) + (yv > 0 ? -4 : 4)); g.lineTo(x, y + (yv > 0 ? 7 : -7)); g.stroke();
    ax.text(txt, x, y, { px: true, al: "center", c: P["ink-2"] });
  };
  lab("noise here raises 𝒮 ", tmax, rd.w[tmax], 1.24 * m);
  lab("noise here lowers 𝒮 ", tmin, rd.w[tmin], -1.24 * m);
};
FIG.f44a = cv => {
  const rd = S.rd; if (!rd) return; const [a, b] = rd.ww; let nm = 0, wm = 0; for (let t = a; t < b; t++) { nm = Math.max(nm, Math.abs(rd.n[t])); wm = Math.max(wm, Math.abs(rd.wT[t])); }
  tplot(cv, [{ y: rd.n, c: P.mc, lw: 1 }, { y: rd.wT.map(v => v / wm * nm * 0.9), c: P.w, lw: 2, noscale: true }], { t0: a, t1: b, ylabel: "n", noX: true, zero: true, sym: true });
};
FIG.f44b = cv => {
  const rd = S.rd; if (!rd) return; const N = S.ctx.N, prod = new Float64Array(N), cum = new Float64Array(N); let s = 0;
  for (let t = 0; t < N; t++) { prod[t] = rd.wT[t] * rd.n[t]; s += prod[t]; cum[t] = s; }
  const ax = new Ax(cv, { l: 58, t: 8, b: 34 }); if (!ax.ok) return;
  const [a, b] = rd.ww; let lo = 0, hi = 0; for (let t = a; t < b; t++) { lo = Math.min(lo, prod[t], cum[t]); hi = Math.max(hi, prod[t], cum[t]); }
  const pad = (hi - lo) * 0.12; ax.xr(a, b).yr(lo - pad, hi + pad).frame({ xlabel: "time (μs)", ylabel: "𝒮  units" });
  ax.clip(); ax.hline(0, { c: P.rule, lw: 1 });
  ax.bars(Array.from({ length: b - a }, (_, i) => a + i), Array.from(prod.slice(a, b)), { bw: 1, c: P.w, a: 0.6, gap: 0 });
  ax.line(i => a + i, Array.from(cum.slice(a, b)), { c: P.ink, lw: 2 });
  ax.hline(rd.epsT, { c: P.ink, lw: 1, dash: [2, 3] }); ax.text(`âᵀe = ${fx(rd.epsT, 3)}`, a + 4, rd.epsT, { al: "left", bl: rd.epsT >= 0 ? "bottom" : "top", c: P.ink });
  ax.unclip();
};
FIG.f44c = cv => {
  const rd = S.rd; if (!rd || !rd.lv) return; const [a, b] = rd.ww, w2 = rd.w.map(v => v * v); let lm = 0, wm = 0; for (let t = a; t < b; t++) { lm = Math.max(lm, rd.lv[t]); wm = Math.max(wm, w2[t]); }
  tplot(cv, [{ y: rd.lv, c: P.mc, lw: 1.8 }, { y: w2.map(v => v / wm * lm), c: P.w, lw: 1.6, noscale: true, dash: [5, 3] }], { t0: a, t1: b, ylabel: "σ_{t}^{2}" });
};
FIG.f45 = cv => {
  const st = S.st; const ax = new Ax(cv, { l: 50, r: 14, t: 12, b: 36 }); if (!ax.ok) return;
  if (!st) { ax.text("waiting for Monte-Carlo draws…", ax.w / 2, ax.h / 2, { px: true, al: "center" }); return; }
  const R = S.mc, md = K.median, perp = (n, p) => Math.sqrt(Math.max(n * n - p * p, 0));
  const ex = R.map(r => r.eps), ey = R.map(r => perp(r.en, r.eps));          // true noise e = D1 S n
  const rx = R.map(r => r.rPar), ry = R.map(r => perp(r.s23rad, r.rPar));   // S23 residual r = D1 S Delta_i
  const bx = md(R.map(r => r.bPar)), by = md(R.map(r => perp(r.bN, r.bPar)));  // burst part of r, fixed
  const ymax = Math.max(st.en, st.s23rad, ...ey, ...ry) * 1.1, ymin = -0.26 * ymax;
  const unit = (ax.py0 - ax.py1) / (ymax - ymin), xh = (ax.px1 - ax.px0) / 2 / unit;
  ax.xr(-xh, xh).yr(ymin, ymax).frame({ xlabel: "component along â", ylabel: "length ⊥ â", xgrid: true, yt: niceTicks(0, ymax, 4) });
  ax.clip();
  const g = ax.g;
  ax.rectD(-st.sig, st.sig, 0, ymax, { c: P.accent, a: 0.13 });
  { const X = [], Y = []; for (let i = 0; i <= 180; i++) { const th = Math.PI * i / 180; X.push(st.en * Math.cos(th)); Y.push(st.en * Math.sin(th)); } ax.line(X, Y, { c: P.muted, lw: 1.2, dash: [5, 4] }); }
  ax.vline(0, { c: P.rule, lw: 1, y0: 0, y1: ymax });
  ax.dots(ex, ey, { c: P.mc, r: 2.5, a: 0.7 });
  ax.dots(rx, ry, { c: P.s23, r: 2.3, a: 0.45 });
  // a-hat axis
  { const Y = Math.round(ax.Y(0)) + 0.5; g.strokeStyle = P["ink-2"]; g.lineWidth = 1.2; g.setLineDash([]); g.beginPath(); g.moveTo(ax.px0, Y); g.lineTo(ax.px1 - 2, Y); g.stroke();
    g.fillStyle = P["ink-2"]; g.beginPath(); g.moveTo(ax.px1 - 1, Y); g.lineTo(ax.px1 - 9, Y - 4); g.lineTo(ax.px1 - 9, Y + 4); g.closePath(); g.fill();
    g.font = `12px ${P.mono}`; g.textAlign = "right"; g.textBaseline = "bottom"; g.fillText("â", ax.px1 - 4, Y - 4); g.font = `11px ${P.mono}`; }
  const arrow = (x0, y0, x1, y1, c, lw, dash) => {
    const X0 = ax.X(x0), Y0 = ax.Y(y0), X1 = ax.X(x1), Y1 = ax.Y(y1), th = Math.atan2(Y1 - Y0, X1 - X0), hl = 9;
    g.strokeStyle = c; g.fillStyle = c; g.lineWidth = lw; g.setLineDash(dash || []);
    g.beginPath(); g.moveTo(X0, Y0); g.lineTo(X1 - hl * 0.8 * Math.cos(th), Y1 - hl * 0.8 * Math.sin(th)); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.moveTo(X1, Y1); g.lineTo(X1 - hl * Math.cos(th - 0.38), Y1 - hl * Math.sin(th - 0.38)); g.lineTo(X1 - hl * Math.cos(th + 0.38), Y1 - hl * Math.sin(th + 0.38)); g.closePath(); g.fill();
  };
  // labels: clamped inside the plot, surface-coloured halo; dy in px (negative = above)
  const lab = (txt, x, y, c, dx, dy, al, short) => {
    let wpx = richWidth(g, txt);
    if (short && wpx > ax.px1 - ax.px0 - 8) { txt = short; wpx = richWidth(g, txt); }
    let X = ax.X(x) + (dx || 0); if (al === "right") X -= wpx; X = Math.min(Math.max(X, ax.px0 + 4), ax.px1 - 4 - wpx);
    const Y = ax.Y(y) + (dy || 0);
    g.textBaseline = "middle"; g.fillStyle = c; richText(g, txt, X, Y, "left", 11, P.surface);
  };
  // fixed burst part d_s of the residual, and the median S23 residual swung onto a-hat
  const rmx = md(rx), rmy = md(ry), rlen = st.s23rad, th0 = Math.atan2(rmy, rmx);
  arrow(0, 0, bx, by, P.s23, 1.2, [3, 3]);
  g.strokeStyle = P.s23; g.lineWidth = 1.6; g.beginPath(); g.arc(ax.X(bx), ax.Y(by), 5, 0, 2 * Math.PI); g.stroke();
  { const X = [], Y = []; for (let i = 0; i <= 60; i++) { const th = th0 * (1 - i / 60); X.push(rlen * Math.cos(th)); Y.push(rlen * Math.sin(th)); } ax.line(X, Y, { c: P.s23, lw: 1.4, dash: [5, 4] }); }
  arrow(0, 0, rmx, rmy, P.s23, 2.2);
  // typical noise vector and its projection
  const exm = st.mcEps, eym = md(ey);
  ax.line([exm, exm], [eym, 0], { c: P.ink, lw: 1, dash: [2, 3], a: 0.7 });
  arrow(0, 0, exm, eym, P.ink, 2.2);
  // along a-hat: what S23 use, and what should be used
  arrow(0, 0, rlen, 0, P.s23, 3);
  arrow(0, 0, st.sig, 0, P.accent, 3);
  lab("typical e", exm, eym, P.ink, 8, -6);
  lab("median D₁SΔ_{i}", rmx, rmy, P.s23, 10, -4);
  lab("d_{s} (fixed)", bx, by, P.s23, 10, 9);
  lab(`σ along â = ${fx(st.sig, 3)} · should be used`, 0, 0, P.accent, 2, 11, null, `σ = ${fx(st.sig, 3)} · should be used`);
  lab(`S23 1σ = ‖D₁SΔ_{i}‖ = ${fx(rlen, 2)} (×${fx(rlen / st.mcEps, 1)}) · currently used`, 0, 0, P.s23, 2, 26, null, `S23 1σ = ${fx(rlen, 2)} (×${fx(rlen / st.mcEps, 1)}) · currently used`);
  ax.unclip(); cv._ax = ax;
};
bindHover($("#f45"), (ev, x, y) => {
  const ax = $("#f45")._ax; if (!ax || x == null) return tip(ev, null);
  const perp = (n, p) => Math.sqrt(Math.max(n * n - p * p, 0));
  let best = null, kind = "", bd = 64;
  for (const r of S.mc) {
    const d1 = (ax.X(r.eps) - x) ** 2 + (ax.Y(perp(r.en, r.eps)) - y) ** 2; if (d1 < bd) { bd = d1; best = r; kind = "e"; }
    const d2 = (ax.X(r.rPar) - x) ** 2 + (ax.Y(perp(r.s23rad, r.rPar)) - y) ** 2; if (d2 < bd) { bd = d2; best = r; kind = "r"; }
  }
  if (!best) return tip(ev, null);
  if (kind === "e") tip(ev, `draw #${best.j + 1}: true noise e = D₁Sn<br>along â = ${fx(best.eps, 3)} · ‖e‖ = ${fx(best.en, 3)}<br>angle to â = ${fx(Math.acos(best.eps / best.en) * 180 / Math.PI, 1)}°`);
  else tip(ev, `draw #${best.j + 1}: S23 residual D₁SΔᵢ<br>along â = ${fx(best.rPar, 3)} · ‖D₁SΔᵢ‖ = ${fx(best.s23rad, 3)}<br>angle to â = ${fx(Math.acos(best.rPar / best.s23rad) * 180 / Math.PI, 1)}°`);
});
FIG.f46 = cv => {
  const rd = S.rd; if (!rd) return; const { F, R } = S.flt, kmax = Math.min(S.ctx.N, Math.ceil(3 * S.kc)), sn2 = rd.sn * rd.sn;
  const ks = [], nz = [], bu = [], sg = [];
  for (let k = 1; k <= kmax; k++) { const j = k - 1; ks.push(k); nz.push(sn2 * R[j] * R[j]); bu.push(R[j] * R[j] * rd.cs[j] * rd.cs[j]); sg.push(sn2 * (F[j] * F[j] * rd.cs[j] / rd.tr.a) ** 2); }
  const ym = Math.max(...nz.map((v, i) => v + bu[i]), ...sg) * 1.08;
  const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  ax.xr(0.5, kmax + 0.5).yr(0, ym).frame({ xlabel: "DCT mode k", ylabel: "per-mode power" });
  ax.clip();
  ax.bars(ks, nz, { bw: 1, c: P.mc, a: 0.75, gap: kmax > 80 ? 0.5 : 1 });
  // burst part split by parity: odd k solid, even k hollow (alternate modes sample cos and sin of each component's phase)
  const gap = kmax > 80 ? 0.5 : 1, odd = i => ks[i] % 2 === 1;
  ax.bars(ks.filter((_, i) => odd(i)), bu.filter((_, i) => odd(i)), { bw: 1, c: P.s23, a: 0.85, b0: nz.filter((_, i) => odd(i)), gap });
  { const g = ax.g; g.strokeStyle = P.s23; g.lineWidth = 1.2; g.setLineDash([]);
    ks.forEach((k, i) => { if (odd(i) || bu[i] <= 0) return;
      const x0 = ax.X(k - 0.5) + gap / 2 + 0.6, x1 = ax.X(k + 0.5) - gap / 2 - 0.6, y0 = ax.Y(nz[i]), y1 = ax.Y(nz[i] + bu[i]);
      if (y0 - y1 < 0.8) return; g.strokeRect(x0, y1, Math.max(x1 - x0, 0.8), y0 - y1); }); }
  ax.line(ks, sg, { c: P.accent, lw: 2 });
  ax.vline(S.kc, { c: P.muted, lw: 1.2, dash: [2, 3] });
  ax.unclip();
};

// ---------------------------------------------------------------- stage 5
function drawStep5() { const k = S.step5; ({ 0: ["f50"], 1: ["f51a", "f51b"], 2: ["f52"], 3: ["f53"], 4: ["f54"] })[k].forEach(draw); updateStep5Vals(); }
function updateStep5Vals() {
  const ctx = S.ctx; if (!ctx) return; const ch = +$("#r-ch").value, dd = +$("#r-dd").value;
  setV("chFreq", fx(ctx.freqs[ch], 0) + " MHz"); setV("ddVal", (dd >= 0 ? "+" : MINUS) + Math.abs(dd).toFixed(4));
  setV("tauCh", fx(ctx.tau[ch] - ctx.tauMean, 0)); setV("shiftCh", fx((ctx.tau[ch] - ctx.tauMean) * dd, 2)); setV("tauBar", fx(ctx.tauMean, 1));
  const st = S.st, tv = +$("#r-tilt").value; setV("tiltVal", (tv >= 0 ? "+" : MINUS) + Math.abs(tv).toFixed(1) + " σ");
  if (st) setV("tiltShift", fdm(-tv * st.slopeMC / st.Sbpp) + " pc cm⁻³");
}
// weight row at an arbitrary trial DM for the selected draw (cached)
function wAt(d) { const sd = S.sd, k = d.toFixed(6); if (!sd.wCache.has(k)) sd.wCache.set(k, K.wRow(S.ctx, S.flt, sd.FI, d)); return sd.wCache.get(k); }
FIG.f50 = cv => {
  const sd = S.sd; if (!sd) return; const ctx = S.ctx, N = ctx.N, ch = +$("#r-ch").value, dd = +$("#r-dd").value, dh = sd.A.dhat;
  const x = sd.ch.X.subarray(ch * N, (ch + 1) * N), w0 = sd.wHat, w1 = wAt(dh + dd);
  const s0 = dh * ctx.tau[ch] / ctx.dt, s1 = (dh + dd) * ctx.tau[ch] / ctx.dt, net = dd * (ctx.tau[ch] - ctx.tauMean) / ctx.dt;
  const [a, b] = sd.ww; let xm = 0, wm = 0; for (let t = a; t < b; t++) { xm = Math.max(xm, Math.abs(x[t])); wm = Math.max(wm, Math.abs(w0[t])); }
  const sc = xm * 0.85 / wm, y0 = new Float64Array(N), y1 = new Float64Array(N);
  for (let t = 0; t < N; t++) { y0[t] = interp(w0, t - s0) * sc; y1[t] = interp(w1, t - s1) * sc; }
  const ax = tplot(cv, [{ y: x, c: P.mc, lw: 1 }, { y: y0, c: P.w, lw: 2.2, noscale: true }, { y: y1, c: P.w, lw: 2.2, dash: [6, 4], noscale: true }], { t0: a, t1: b, ylabel: "x_{ν} (normalised)", zero: true, sym: true });
  if (ax) ax.text(`net slide ${fx(Math.abs(net), 1)} μs ${net > 0 ? "→" : "←"}`, b - 4, ax.ya + (ax.yb - ax.ya) * 0.94, { al: "right", c: P.w });
};
FIG.f51a = cv => {
  const sd = S.sd; if (!sd) return; const ctx = S.ctx, N = ctx.N, dh = sd.A.dhat;
  // same-profile form (visualisation only): -(tau_nu - tau-bar) w-dot(t - d tau_nu)
  const sp = c => { const s = dh * ctx.tau[c] / ctx.dt, k = -(ctx.tau[c] - ctx.tauMean), o = new Float64Array(N); for (let t = 0; t < N; t++) o[t] = k * interp(sd.wdHat, t - s); return o; };
  tplot(cv, [{ y: sd.wpTop, c: P.w, lw: 2 }, { y: sd.wpBot, c: P.deriv, lw: 2 }, { y: sp(0), c: P.ink, lw: 1.2, dash: [1.5, 3] }, { y: sp(ctx.NCH - 1), c: P.ink, lw: 1.2, dash: [1.5, 3] }],
    { t0: sd.ww[0], t1: sd.ww[1], ylabel: "w′_{ν} (per pc cm⁻³)", zero: true, sym: true });
};
FIG.f51b = cv => {
  const ctx = S.ctx; if (!ctx) return; const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  const d = Array.from(ctx.tau, v => v - ctx.tauMean), m = Math.max(...d.map(Math.abs)) * 1.1;
  ax.xr(ctx.freqs[ctx.NCH - 1] - 4, ctx.freqs[0] + 4).yr(-m, m).frame({ xlabel: "ν (MHz)", ylabel: "τ_{ν} (μs)", ny: 4 });
  ax.clip(); ax.rectD(ax.xa, ax.xb, -ctx.sigTau, ctx.sigTau, { c: P.accent, a: 0.13 });
  ax.bars(Array.from(ctx.freqs), d, { bw: 336 / ctx.NCH, c: P.ink, a: 0.75, gap: 1 }); ax.hline(0, { c: P.rule }); ax.unclip();
};
FIG.f52 = cv => {
  const st = S.st; const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  if (!st) { ax.text("waiting for Monte-Carlo draws…", ax.w / 2, ax.h / 2, { px: true, al: "center" }); return; }
  const ctx = S.ctx, i0 = ctx.i0, W = 12, dm = [], R = S.mc.slice(0, 60);
  for (let m = i0 - W; m <= i0 + W; m++) dm.push(ctx.dms[m]);
  const curve = r => dm.map((_, i) => (r.S[i0 - W + i] - st.ref[i0 - W + i]) - (r.S[i0] - st.ref[i0]));
  let ym = 0; const cs = R.map(curve); for (const c of cs) for (const v of c) ym = Math.max(ym, Math.abs(v));
  // S23 Eq. 11 allowance for S(d) - S(d0): the radius ||D1 S (Delta_d - Delta_d0)|| <-> ||F(1-f)(c_d - c_d0)||, median over the eight cards
  const N = ctx.N, { F, f } = S.flt, cards = S.stack.filter(Boolean);
  const env = dm.map((_, i) => { const m = i0 - W + i, v = cards.map(st2 => { const C = st2.A.C; let s2 = 0; for (let k = 0; k < N; k++) { const q = F[k] * (1 - f[k]) * (C[m * N + k] - C[i0 * N + k]); s2 += q * q; } return Math.sqrt(s2); }); return K.median(v); });
  const h = ctx.dms[i0 + 1] - ctx.dms[i0]; setV("s23slope", fx(0.5 * (env[W + 1] + env[W - 1]) / h, 1));
  ym = Math.max(ym, ...env);
  ax.xr(dm[0], dm[dm.length - 1]).yr(-ym * 1.08, ym * 1.08).frame({ xlabel: "ΔDM (pc cm⁻³)", ylabel: "e_{∥} (𝒮  units)" });
  ax.clip();
  const fan = st.slopeData, xe = dm[dm.length - 1];
  ax.band([dm[0], 0, xe], [fan * -dm[0] * 1, 0, -fan * xe], [-fan * -dm[0], 0, fan * xe], { c: P.accent, a: 0.12 });
  ax.band(dm, env.map(v => -v), env, { c: P.s23, a: 0.07 });
  ax.line(dm, env, { c: P.s23, lw: 1.8 }); ax.line(dm, env.map(v => -v), { c: P.s23, lw: 1.8 });
  ax.text("S23 Eq. 11 allowance", dm[1], env[1], { al: "left", bl: "bottom", c: P.s23 });
  cs.forEach((c, i) => { if (R[i].j !== S.sel) ax.line(dm, c, { c: P.mc, lw: 1, a: 0.55 }); });
  const me = S.mc.find(r => r.j === S.sel); if (me) ax.line(dm, curve(me), { c: P.ink, lw: 2.2 });
  ax.line([dm[0], xe], [fan * dm[0], fan * xe], { c: P.accent, lw: 1.4, dash: [6, 4] }); ax.line([dm[0], xe], [-fan * dm[0], -fan * xe], { c: P.accent, lw: 1.4, dash: [6, 4] });
  ax.text("predicted ±1σ slope", xe, fan * xe, { al: "right", bl: "bottom", c: P.accent });
  ax.unclip();
};
FIG.f53 = cv => {
  const st = S.st; const ax = new Ax(cv, { b: 36 }); if (!ax.ok) return;
  if (!st) { ax.text("waiting for Monte-Carlo draws…", ax.w / 2, ax.h / 2, { px: true, al: "center" }); return; }
  const ctx = S.ctx, i0 = ctx.i0, W = 20, dm = [], sb = []; for (let m = i0 - W; m <= i0 + W; m++) { dm.push(ctx.dms[m]); sb.push(st.ref[m]); }
  const q = st.q, fit = dm.map(x => q[0] + q[1] * x + q[2] * x * x), tv = +$("#r-tilt").value, sl = tv * st.slopeMC;
  const tilt = dm.map((x, i) => sb[i] + sl * x);
  const lo = Math.min(...sb, ...tilt), hi = Math.max(...sb, ...tilt), pad = (hi - lo) * 0.08;
  ax.xr(dm[0], dm[dm.length - 1]).yr(lo - pad, hi + pad).frame({ xlabel: "ΔDM (pc cm⁻³)", ylabel: "𝒮 " });
  ax.clip();
  ax.line(dm, fit, { c: P.ink, lw: 1.2, dash: [1.5, 3] }); ax.line(dm, sb, { c: P.ink, lw: 2 }); ax.line(dm, tilt, { c: P.accent, lw: 2.2 });
  const x0 = -q[1] / (2 * q[2]), xt = -(q[1] + sl) / (2 * q[2]);
  const yt = q[0] + (q[1] + sl) * xt + q[2] * xt * xt, y0 = q[0] + q[1] * x0 + q[2] * x0 * x0;
  ax.vline(0, { c: P.muted, lw: 1, dash: [2, 3] });
  ax.dots([x0], [y0], { c: P.ink, r: 4, ring: P.surface }); ax.dots([xt], [yt], { c: P.accent, r: 4.5, ring: P.surface });
  ax.vline(x0, { c: P.ink, lw: 1, dash: [2, 3], y0: lo - pad, y1: y0 }); ax.vline(xt, { c: P.accent, lw: 1, dash: [2, 3], y0: lo - pad, y1: yt });
  const g = ax.g, yA = ax.py0 - 30; g.strokeStyle = P.accent; g.fillStyle = P.accent; g.lineWidth = 1.6; g.beginPath(); g.moveTo(ax.X(x0), yA); g.lineTo(ax.X(xt), yA); g.stroke();
  if (Math.abs(ax.X(xt) - ax.X(x0)) > 6) { const s = Math.sign(xt - x0); g.beginPath(); g.moveTo(ax.X(xt), yA); g.lineTo(ax.X(xt) - s * 7, yA - 4); g.lineTo(ax.X(xt) - s * 7, yA + 4); g.closePath(); g.fill(); }
  ax.text(` peak shift ${fdm(xt - x0)} pc cm⁻³ `, xt, ax.iY(yA), { al: xt >= x0 ? "left" : "right", c: P.accent });
  ax.unclip();
};
FIG.f54 = cv => drawHist(cv, true);

// ---------------------------------------------------------------- steppers
function stepper(root, key, onShow) {
  const tabs = Array.from(root.querySelectorAll(".steps-nav button")), steps = Array.from(root.querySelectorAll(".step")), pos = root.querySelector("[data-pos]");
  const prev = root.querySelector('[data-nav="-1"]'), next = root.querySelector('[data-nav="1"]'), n = steps.length;
  function show(k, focus) {
    k = Math.max(0, Math.min(n - 1, k)); S[key] = k;
    tabs.forEach((b, i) => { b.setAttribute("aria-selected", i === k ? "true" : "false"); b.tabIndex = i === k ? 0 : -1; });
    steps.forEach((s, i) => s.hidden = i !== k); pos.textContent = `Step ${k + 1} of ${n}`;
    prev.disabled = k === 0; next.disabled = k === n - 1; if (focus) tabs[k].focus();
    requestAnimationFrame(() => onShow(k));
  }
  tabs.forEach((b, i) => { b.addEventListener("click", () => show(i)); b.addEventListener("keydown", e => { if (e.key === "ArrowRight") show(S[key] + 1, true); if (e.key === "ArrowLeft") show(S[key] - 1, true); }); });
  prev.addEventListener("click", () => show(S[key] - 1)); next.addEventListener("click", () => show(S[key] + 1));
  show(0);
}

// ---------------------------------------------------------------- scheduling / events
const pending = new Set(); let raf = 0;
function schedule(what) { pending.add(what); if (!raf) raf = requestAnimationFrame(flush); }
function flush() { raf = 0; const p = new Set(pending); pending.clear(); if (p.has("mc")) updateSummary(); if (p.has("all")) redrawAll(); }
function redrawAll() { layoutDeck(); for (let j = 0; j < NSTACK; j++) drawCard(j); for (const id of ["cv-sd", "cv-hist", "cv-cube", "cv-sprof", "cv-dyn", "cv-row"]) draw(id); drawStep4(); drawStep5(); }

function init() {
  readPalette();
  stepper($("#st4"), "step4", () => drawStep4());
  stepper($("#st5"), "step5", () => drawStep5());
  let t = null;
  $("#controls").addEventListener("change", e => { if (e.target.id === "c-smooth") return; clearTimeout(t); t = setTimeout(start, 150); });
  $("#controls").addEventListener("submit", e => e.preventDefault());
  $("#b-seed").addEventListener("click", () => { $("#c-seed").value = (+$("#c-seed").value || 1) + 1; start(); });
  $("#r-dm").addEventListener("input", e => setRow(+e.target.value));
  $("#b-dm-prev").addEventListener("click", () => setRow(S.row - 1));
  $("#b-dm-next").addEventListener("click", () => setRow(S.row + 1));
  $("#b-dm-peak").addEventListener("click", () => S.sd && setRow(S.sd.A.dpk));
  $("#b-dm-true").addEventListener("click", () => setRow(S.ctx.i0));
  $("#c-smooth").addEventListener("change", () => draw("cv-dyn"));
  for (const id of ["r-ch", "r-dd"]) $("#" + id).addEventListener("input", () => { updateStep5Vals(); drawStep5(); });
  $("#r-tilt").addEventListener("input", () => { updateStep5Vals(); draw("f53"); });
  const ro = new ResizeObserver(() => schedule("all")); ro.observe($(".wrap"));
  const mq = matchMedia("(prefers-color-scheme: dark)"); const onTheme = () => { readPalette(); schedule("all"); };
  mq.addEventListener ? mq.addEventListener("change", onTheme) : mq.addListener(onTheme);
  new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => schedule("all"));
  start();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
