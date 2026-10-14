function makeCore() {
  "use strict";
  // ---------- RNG ----------
  function hash32(a) { a = (a ^ 61) ^ (a >>> 16); a = a + (a << 3) | 0; a = a ^ (a >>> 4); a = Math.imul(a, 0x27d4eb2d); a = a ^ (a >>> 15); return a >>> 0; }
  function mulberry32(a) {
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function gaussian(seed) {
    const u = mulberry32(seed); let spare = 0, has = false;
    return function () {
      if (has) { has = false; return spare; }
      let a = u(); while (a <= 1e-300) a = u();
      const b = u(), r = Math.sqrt(-2 * Math.log(a));
      spare = r * Math.sin(2 * Math.PI * b); has = true; return r * Math.cos(2 * Math.PI * b);
    };
  }
  function seedFor(base, j) { return hash32((hash32(base >>> 0) + Math.imul(j + 1, 0x9E3779B1)) >>> 0); }

  // ---------- FFT (radix-2, in place) ----------
  function makeFFT(N) {
    const L = Math.round(Math.log2(N)), rev = new Uint32Array(N);
    for (let i = 0; i < N; i++) { let r = 0, x = i; for (let b = 0; b < L; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
    const ct = new Float64Array(N >> 1), st = new Float64Array(N >> 1);
    for (let i = 0; i < N >> 1; i++) { ct[i] = Math.cos(2 * Math.PI * i / N); st[i] = Math.sin(2 * Math.PI * i / N); }
    return function (re, im, inv) {
      for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (let size = 2; size <= N; size <<= 1) {
        const half = size >> 1, step = N / size;
        for (let i = 0; i < N; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half, c = ct[k], s = inv ? st[k] : -st[k];
            const tr = re[l] * c - im[l] * s, ti = re[l] * s + im[l] * c;
            re[l] = re[j] - tr; im[l] = im[j] - ti; re[j] += tr; im[j] += ti;
          }
        }
      }
      if (inv) for (let i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
    };
  }

  // ---------- orthonormal DCT-II / DCT-III (scipy norm="ortho") ----------
  function makeDCT(N, fft) {
    const vr = new Float64Array(N), vi = new Float64Array(N), wc = new Float64Array(N), ws = new Float64Array(N);
    for (let k = 0; k < N; k++) { wc[k] = Math.cos(Math.PI * k / (2 * N)); ws[k] = Math.sin(Math.PI * k / (2 * N)); }
    const a0 = Math.sqrt(1 / N), a = Math.sqrt(2 / N), h = N >> 1;
    function dct(x, out) {
      out = out || new Float64Array(N);
      for (let n = 0; n < h; n++) { vr[n] = x[2 * n]; vr[N - 1 - n] = x[2 * n + 1]; }
      vi.fill(0); fft(vr, vi, false);
      for (let k = 0; k < N; k++) out[k] = (vr[k] * wc[k] + vi[k] * ws[k]) * (k ? a : a0);
      return out;
    }
    function idct(c, out) {
      out = out || new Float64Array(N);
      for (let k = 0; k < N; k++) {
        const yk = c[k] / (k ? a : a0), ynk = k ? c[N - k] / a : 0;
        vr[k] = yk * wc[k] + ynk * ws[k]; vi[k] = yk * ws[k] - ynk * wc[k];
      }
      fft(vr, vi, true);
      for (let n = 0; n < h; n++) { out[2 * n] = vr[n]; out[2 * n + 1] = vr[N - 1 - n]; }
      return out;
    }
    return { dct, idct };
  }

  // ---------- small helpers ----------
  function norm(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s); }
  function median(arr) { const a = Float64Array.from(arr).sort(); const n = a.length; return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]); }
  function std(arr) { let m = 0; for (const v of arr) m += v; m /= arr.length; let s = 0; for (const v of arr) s += (v - m) * (v - m); return Math.sqrt(s / arr.length); }
  function peakRefine(y, x) {
    let k = 0; for (let i = 1; i < y.length; i++) if (y[i] > y[k]) k = i;
    k = Math.min(Math.max(k, 1), y.length - 2);
    const a = y[k - 1], b = y[k], c = y[k + 1], den = a - 2 * b + c;
    return x[k] + (den !== 0 ? 0.5 * (a - c) / den : 0) * (x[1] - x[0]);
  }
  function argmax(y) { let k = 0; for (let i = 1; i < y.length; i++) if (y[i] > y[k]) k = i; return k; }
  function quadfit(x, y, x0) { // least squares y = c0 + c1 (x-x0) + c2 (x-x0)^2
    let S = [0, 0, 0, 0, 0], T = [0, 0, 0];
    for (let i = 0; i < x.length; i++) { const u = x[i] - x0; let p = 1; for (let j = 0; j < 5; j++) { S[j] += p; if (j < 3) T[j] += p * y[i]; p *= u; } }
    const M = [[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]];
    const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const D = det(M), c = [];
    for (let j = 0; j < 3; j++) { const Mj = M.map((r, i) => r.map((v, k) => k === j ? T[i] : v)); c.push(det(Mj) / D); }
    return c;
  }

  // ---------- model setup (mirrors error_validation/sim.py) ----------
  function setup(p) {
    const N = p.N || 2048, NCH = p.NCH || 84, dt = p.dt || 1.0, KDM = 4.149e9;
    const freqs = new Float64Array(NCH);
    for (let i = 0; i < NCH; i++) freqs[i] = 1440 - i * (336 / NCH) - 2.0;
    const fref = median(freqs);
    const tau = new Float64Array(NCH);
    for (let i = 0; i < NCH; i++) tau[i] = KDM * (Math.pow(freqs[i], -2) - Math.pow(fref, -2));
    const dms = []; for (let m = -40; m <= 40; m++) dms.push(Math.round(m) / 1000);
    const ND = dms.length, i0 = 40;
    const amp = p.amp, prof = new Float64Array(N);
    for (let t = 0; t < N; t++) prof[t] = amp * (Math.exp(-0.5 * ((t * dt - 1000) / 12) ** 2) + 0.7 * Math.exp(-0.5 * ((t * dt - 1050) / 20) ** 2));
    const fft = makeFFT(N), D = makeDCT(N, fft);
    let tm = 0; for (const v of tau) tm += v; tm /= NCH;
    let tv = 0; for (const v of tau) tv += (v - tm) ** 2; const sigTau = Math.sqrt(tv / NCH);
    const omega = new Float64Array(N); for (let k = 0; k < N; k++) omega[k] = Math.PI * k / (N * dt);
    return { N, NCH, dt, freqs, fref, tau, tauMean: tm, sigTau, dms: Float64Array.from(dms), ND, i0, prof, amp, fft, dct: D.dct, idct: D.idct, omega, selfnoise: !!p.selfnoise };
  }

  // channel intensities, channel-normalised exactly as sim.realise
  function genChannels(ctx, seed, withSignal) {
    const { N, NCH, prof } = ctx, g = gaussian(seed);
    const I = new Float64Array(NCH * N), X = new Float64Array(NCH * N), Sg = new Float64Array(NCH * N);
    const mean = new Float64Array(NCH), sd = new Float64Array(NCH);
    for (let ch = 0; ch < NCH; ch++) {
      const o = ch * N; let m = 0;
      for (let t = 0; t < N; t++) {
        const S = withSignal ? prof[t] : 0;
        const a = g(), b = g(), c = g(), d = g();
        const q = 0.25 * (a * a + b * b + c * c + d * d);
        const v = ctx.selfnoise ? q * (1 + S) : q + S;
        I[o + t] = v; m += v;
      }
      m /= N; let s = 0;
      for (let t = 0; t < N; t++) s += (I[o + t] - m) ** 2;
      s = Math.sqrt(s / N); mean[ch] = m; sd[ch] = s;
      for (let t = 0; t < N; t++) {
        const S = withSignal ? prof[t] : 0;
        X[o + t] = (I[o + t] - (1 + S)) / s;      // true noise part
        Sg[o + t] = (1 + S - m) / s;              // true signal part
        I[o + t] = (I[o + t] - m) / s;            // data = signal + noise
      }
    }
    return { I, X, Sg, mean, sd };
  }

  function chanFFT(ctx, A) {
    const { N, NCH, fft } = ctx, re = new Float64Array(NCH * N), im = new Float64Array(NCH * N);
    const r = new Float64Array(N), i = new Float64Array(N);
    for (let ch = 0; ch < NCH; ch++) {
      r.set(A.subarray(ch * N, ch * N + N)); i.fill(0); fft(r, i, false);
      re.set(r, ch * N); im.set(i, ch * N);
    }
    return { re, im };
  }

  // I(DM,t) cube: channel ch shifted by d*tau_ch (Fourier), summed. jmax limits rfft bins (band-limited fast mode).
  function dedisperse(ctx, F, jmax) {
    const { N, NCH, ND, dms, tau, dt, fft } = ctx, H = N >> 1;
    const NK = Math.min(jmax == null ? H : jmax, H) + 1;
    const Pr = new Float64Array(NCH * NK), Pi = new Float64Array(NCH * NK), Rr = new Float64Array(NCH * NK), Ri = new Float64Array(NCH * NK);
    const step = dms[1] - dms[0];
    for (let ch = 0; ch < NCH; ch++) for (let k = 0; k < NK; k++) {
      const nu = k / (N * dt), ph0 = -2 * Math.PI * nu * tau[ch] * dms[0], ph = -2 * Math.PI * nu * tau[ch] * step, o = ch * NK + k;
      Pr[o] = Math.cos(ph0); Pi[o] = Math.sin(ph0); Rr[o] = Math.cos(ph); Ri[o] = Math.sin(ph);
    }
    const out = new Float32Array(ND * N), xr = new Float64Array(N), xi = new Float64Array(N);
    for (let m = 0; m < ND; m++) {
      xr.fill(0); xi.fill(0);
      for (let ch = 0; ch < NCH; ch++) {
        const o = ch * NK, fo = ch * N;
        for (let k = 0; k < NK; k++) {
          const fr = F.re[fo + k], fi = F.im[fo + k], pr = Pr[o + k], pi = Pi[o + k];
          xr[k] += fr * pr - fi * pi; xi[k] += fr * pi + fi * pr;
          const rr = Rr[o + k], ri = Ri[o + k];
          Pr[o + k] = pr * rr - pi * ri; Pi[o + k] = pr * ri + pi * rr;
        }
      }
      xi[0] = 0; if (NK > H) xi[H] = 0;
      for (let k = 1; k < H; k++) { xr[N - k] = xr[k]; xi[N - k] = -xi[k]; }
      fft(xr, xi, true);
      out.set(xr, m * N);
    }
    return out;
  }

  function dedisperseRow(ctx, F, d) {
    const { N, NCH, tau, dt, fft } = ctx, H = N >> 1;
    const xr = new Float64Array(N), xi = new Float64Array(N);
    for (let ch = 0; ch < NCH; ch++) {
      const fo = ch * N;
      for (let k = 0; k <= H; k++) {
        const ph = -2 * Math.PI * (k / (N * dt)) * tau[ch] * d, c = Math.cos(ph), s = Math.sin(ph);
        const fr = F.re[fo + k], fi = F.im[fo + k];
        xr[k] += fr * c - fi * s; xi[k] += fr * s + fi * c;
      }
    }
    xi[0] = 0; xi[H] = 0;
    for (let k = 1; k < H; k++) { xr[N - k] = xr[k]; xi[N - k] = -xi[k]; }
    fft(xr, xi, true);
    return xr;
  }

  // per-channel shifted series in a time window [t0, t1) — the dynamic spectrum behind one row
  function dynspec(ctx, F, d, t0, t1) {
    const { N, NCH, tau, dt, fft } = ctx, H = N >> 1, W = t1 - t0;
    const out = new Float32Array(NCH * W), xr = new Float64Array(N), xi = new Float64Array(N);
    for (let ch = 0; ch < NCH; ch++) {
      const fo = ch * N; xr.fill(0); xi.fill(0);
      for (let k = 0; k <= H; k++) {
        const ph = -2 * Math.PI * (k / (N * dt)) * tau[ch] * d, c = Math.cos(ph), s = Math.sin(ph);
        const fr = F.re[fo + k], fi = F.im[fo + k];
        xr[k] = fr * c - fi * s; xi[k] = fr * s + fi * c;
      }
      xi[0] = 0; xi[H] = 0;
      for (let k = 1; k < H; k++) { xr[N - k] = xr[k]; xi[N - k] = -xi[k]; }
      fft(xr, xi, true);
      for (let t = 0; t < W; t++) out[ch * W + t] = xr[t0 + t];
    }
    return out;
  }

  // ---------- SHRINE pieces (struct_max_core.py) ----------
  function getKc(Crows, ND, N) {
    const mx = new Float64Array(N);
    for (let m = 0; m < ND; m++) for (let k = 0; k < N; k++) { const v = Math.abs(Crows[m * N + k]); if (v > mx[k]) mx[k] = v; }
    const nm = Math.floor(N / 2); let top = 0; for (let k = nm; k < N; k++) top += mx[k]; top /= (N - nm);
    for (let i = 5; i < N; i++) { let s = 0; for (let j = i - 5; j <= i; j++) s += mx[j]; if (s / 6 <= top) return i; }
    return 40;
  }
  function filters(N, kc, O) {
    const f = new Float64Array(N), h = new Float64Array(N), F = new Float64Array(N), R = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      const k = j + 1; f[j] = 1 / (1 + Math.pow(k / kc, 2 * O));
      h[j] = Math.sqrt(2 - 2 * Math.cos(j * Math.PI / N)); F[j] = h[j] * f[j]; R[j] = F[j] * (1 - f[j]);
    }
    return { f, h, F, R, kc, O };
  }
  function localVar(ctx, c) { // sigma_t^2 from the k > N/4 band (struct_max_errors.local_noise_var)
    const { N, idct } = ctx, kmin = Math.floor(N / 4), cc = new Float64Array(N);
    for (let k = kmin; k < N; k++) cc[k] = c[k];
    const hi = idct(cc), out = new Float64Array(N), W = 31, h = 15;
    const sq = hi.map(v => v * v); let run = 0;
    for (let t = 0; t < Math.min(h, N); t++) run += sq[t];
    for (let t = 0; t < N; t++) {
      if (t + h < N) run += sq[t + h];
      if (t - h - 1 >= 0) run -= sq[t - h - 1];
      out[t] = run / W / (1 - kmin / N);
    }
    return out;
  }
  function sigmaHatN(c, N) { const kmin = Math.floor(N / 4), a = new Float64Array(N - kmin); for (let k = kmin; k < N; k++) a[k - kmin] = Math.abs(c[k]); return 1.4826 * median(a); }

  // Full analysis of one I(DM,t) cube. full=true keeps per-row DCTs for the UI.
  function analyse(ctx, cube, flt, opt) {
    opt = opt || {};
    const { N, ND, dms, dct } = ctx, { f, F } = flt;
    const C = new Float64Array(ND * N), S = new Float64Array(ND), FLn = new Float64Array(ND), unc = new Float64Array(ND), sig = new Float64Array(ND), sn = new Float64Array(ND);
    const row = new Float64Array(N), c = new Float64Array(N);
    for (let m = 0; m < ND; m++) {
      for (let t = 0; t < N; t++) row[t] = cube[m * N + t];
      dct(row, c); C.set(c, m * N);
      let s1 = 0, s2 = 0, s3 = 0, s4 = 0;
      for (let k = 0; k < N; k++) {
        const Fc = F[k] * c[k], FLc = Fc * f[k];
        s1 += Fc * Fc; s2 += FLc * FLc; s3 += (F[k] * (1 - f[k]) * c[k]) ** 2; s4 += (F[k] * FLc) ** 2;
      }
      S[m] = Math.sqrt(s1); FLn[m] = Math.sqrt(s2); unc[m] = Math.sqrt(s3) / FLn[m];
      if (!opt.bandLimited) { sn[m] = sigmaHatN(c, N); sig[m] = sn[m] * Math.sqrt(s4) / FLn[m]; }
    }
    const dpk = argmax(S), dhat = peakRefine(S, dms);
    const rel = new Float64Array(ND);
    for (let m = 0; m < ND; m++) {
      let s = 0; for (let k = 0; k < N; k++) { const v = F[k] * (1 - f[k]) * (C[m * N + k] - C[dpk * N + k]); s += v * v; }
      rel[m] = Math.sqrt(s) / FLn[m];
    }
    // S23 range (dm_range_from_relative_uncertainty)
    const mx = S[dpk], ranges = []; let cont = false;
    for (let m = 0; m < ND; m++) { const adj = S[m] + S[m] * rel[m]; if (!cont && adj >= mx) { cont = true; ranges.push([m]); } else if (cont && adj < mx) { cont = false; ranges[ranges.length - 1].push(m); } }
    let lo = null, hi = null;
    if (ranges.length) { lo = dms[ranges[0][0]]; const last = ranges[ranges.length - 1]; hi = last.length === 2 ? dms[last[1]] : dms[ND - 1]; }
    // curvature from a quadratic fit over peak +-3 trials
    const a = Math.max(dpk - 3, 0), b = Math.min(dpk + 4, ND), xs = [], ys = [];
    for (let m = a; m < b; m++) { xs.push(dms[m]); ys.push(S[m]); }
    const q = quadfit(xs, ys, dms[dpk]), Spp = 2 * q[2];
    // direction u at the peak row and ||Omega F u||
    let uNorm2 = 0; const u = new Float64Array(N);
    for (let k = 0; k < N; k++) { u[k] = F[k] * f[k] * C[dpk * N + k]; uNorm2 += u[k] * u[k]; }
    const un = Math.sqrt(uNorm2); let oFu = 0, Fu = 0;
    for (let k = 0; k < N; k++) { u[k] /= un; oFu += (ctx.omega[k] * F[k] * u[k]) ** 2; Fu += (F[k] * u[k]) ** 2; }
    oFu = Math.sqrt(oFu); Fu = Math.sqrt(Fu);
    return { C, S, FLn, unc, rel, sig, sn, dpk, dhat, lo, hi, quad: { x0: dms[dpk], c: q, a, b }, Spp, u, oFu, Fu };
  }

  // time-domain weights w = C F u  (= IDCT(F^2 L c)/||F L c||) for a row's DCT c
  function weights(ctx, flt, c) {
    const { N, idct } = ctx, { f, F } = flt, v = new Float64Array(N); let n2 = 0;
    for (let k = 0; k < N; k++) { const x = F[k] * f[k] * c[k]; n2 += x * x; v[k] = F[k] * x; }
    const n = Math.sqrt(n2); for (let k = 0; k < N; k++) v[k] /= n;
    return idct(v);
  }

  // One Monte-Carlo realisation, reduced to the numbers the explorer needs.
  function mcOne(ctx, flt, seed, cubeFull, chans) {
    const { N, NCH, i0, dct, sigTau, ND } = ctx, { f, F } = flt;
    const ch = chans || genChannels(ctx, seed, true);
    const cube = cubeFull || dedisperse(ctx, chanFFT(ctx, ch.I));
    const A = analyse(ctx, cube, flt);
    // full-band d = 0 row: no shift, so it is the plain channel sum
    const i0r = new Float64Array(N), n0 = new Float64Array(N), s0 = new Float64Array(N);
    for (let c = 0; c < NCH; c++) for (let t = 0; t < N; t++) { i0r[t] += ch.I[c * N + t]; n0[t] += ch.X[c * N + t]; s0[t] += ch.Sg[c * N + t]; }
    const c0 = dct(i0r), nu = dct(n0), cs = dct(s0);
    let S0 = 0, FL = 0, R = 0, F2L = 0;
    for (let k = 0; k < N; k++) { const Fc = F[k] * c0[k]; S0 += Fc * Fc; FL += (Fc * f[k]) ** 2; R += (F[k] * (1 - f[k]) * c0[k]) ** 2; F2L += (F[k] * Fc * f[k]) ** 2; }
    S0 = Math.sqrt(S0); FL = Math.sqrt(FL); R = Math.sqrt(R); F2L = Math.sqrt(F2L);
    const sn = sigmaHatN(c0, N), sigW = sn * F2L / FL;
    const w = weights(ctx, flt, c0), lv = localVar(ctx, c0);
    let sl = 0, wn = 0; for (let t = 0; t < N; t++) { sl += w[t] * w[t] * lv[t]; wn += w[t] * w[t]; }
    const sigL = Math.sqrt(sl), sEff = ctx.selfnoise ? Math.sqrt(sl / wn) : sn;
    // true decomposition: eps = u_s^T F nu, ||e|| = ||F nu||
    let us = 0; for (let k = 0; k < N; k++) us += (F[k] * cs[k]) ** 2; us = Math.sqrt(us);
    let eps = 0, en = 0; for (let k = 0; k < N; k++) { eps += (F[k] * cs[k] / us) * F[k] * nu[k]; en += (F[k] * nu[k]) ** 2; }
    en = Math.sqrt(en);
    // S23 residual r = D1 S Delta_i <-> F(1-f)c0 and its fixed burst part b = D1 S (I-S) s <-> F(1-f)c_s, projected on the true a-hat
    let rPar = 0, bPar = 0, bN = 0;
    for (let k = 0; k < N; k++) { const Rk = F[k] * (1 - f[k]), uk = F[k] * cs[k] / us; rPar += uk * Rk * c0[k]; bPar += uk * Rk * cs[k]; bN += (Rk * cs[k]) ** 2; }
    bN = Math.sqrt(bN);
    const sigDM = sEff * sigTau * A.oFu / Math.abs(A.Spp);
    return { oFu: A.oFu, S: Float32Array.from(A.S), dhat: A.dhat, lo: A.lo, hi: A.hi, S0, s23abs: R / FL * S0, s23rad: R, sigW, sigL, sigDM, eps, en, rPar, bPar, bN, A: us, sn, Spp: A.Spp };
  }

  return { setup, genChannels, chanFFT, dedisperse, dedisperseRow, dynspec, getKc, filters, analyse, weights, localVar, sigmaHatN, mcOne, seedFor, norm, median, std, quadfit, peakRefine, argmax, gaussian };
}
if (typeof module !== "undefined") module.exports = makeCore;
