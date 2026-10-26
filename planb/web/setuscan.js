/* SetuScan in-browser pipeline: OpenCV 5.0.0 (opencv.js, WebAssembly).
 * A port of the Python pipeline (setuscan/*.py) to the subset of OpenCV 5 that the opencv.js build exposes:
 * imgproc, objdetect (ArUco), calib3d. NOT ported (not in the opencv.js build or not needed):
 *  - ximgproc thinning -> Zhang-Suen implemented here in plain JS
 *  - SIFT visit comparison -> Python build only
 * Pure functions of (cv, ImageData-like {data,width,height}); runs in a browser or in Node (tests).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SetuScan = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------ constants (same as the Python build)
  const MARKER_SIZES_MM = { 0: 40, 1: 40, 2: 20, 3: 20 };
  const CARD_MARGIN = 0.45, CARD_GUARD_MM = 6, WORK_SIDE = 1400, ZOOM_SIDE = 2000;
  const MAX_CRACK_MM = 8, MIN_CRACK_LEN_MM = 12, MIN_RELIABLE_WIDTH_PX = 3.0, MAX_EDGE_BLUR_PX = 2.5;
  const QTH = { min_sharpness: 2.0, min_mean: 45, max_mean: 225, max_dark_frac: 0.35, max_bright_frac: 0.25, max_glare_frac: 0.04 };
  const POLICY = {
    label: "EXAMPLE policy (illustrative thresholds, not a code-certified standard)",
    bands: [
      { name: "hairline", max_mm: 0.3, action: "monitor", reinspect_days: 365 },
      { name: "fine", max_mm: 1.0, action: "monitor", reinspect_days: 180 },
      { name: "moderate", max_mm: 2.0, action: "review", reinspect_days: 30 },
      { name: "wide", max_mm: Infinity, action: "urgent_review", reinspect_days: 7 },
    ],
    edge_margin_frac: 0.15,
  };
  const bandFor = w => POLICY.bands.find(b => w < b.max_mm) || POLICY.bands[POLICY.bands.length - 1];
  const edges = () => POLICY.bands.filter(b => isFinite(b.max_mm)).map(b => b.max_mm);
  function nearestEdge(w) { const e = edges().reduce((a, b) => (Math.abs(b - w) < Math.abs(a - w) ? b : a)); return [e, Math.abs(w - e) / e]; }
  function isAmbiguous(w, mmpp) { const [e, rel] = nearestEdge(w); return rel <= POLICY.edge_margin_frac || Math.abs(w - e) <= mmpp; }

  // ------------------------------------------------------------ small helpers
  const percentile = (arr, p) => { const a = Float64Array.from(arr).sort(); if (!a.length) return NaN;
    const k = (a.length - 1) * p / 100, f = Math.floor(k), c = Math.min(f + 1, a.length - 1); return a[f] + (a[c] - a[f]) * (k - f); };
  const median = a => percentile(a, 50);
  const round = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? v : Math.round(v * 10 ** d) / 10 ** d);

  function u8Mat(cv, h, w, arr) { const m = new cv.Mat(h, w, cv.CV_8U); m.data.set(arr); return m; }
  function Scope() { const l = []; return { add(m) { l.push(m); return m; }, free() { l.forEach(m => { try { m.delete(); } catch (e) {} }); l.length = 0; } }; }

  function grayMat(cv, imageData, S) {
    const rgba = S.add(cv.matFromImageData(imageData));
    const g = S.add(new cv.Mat());
    cv.cvtColor(rgba, g, cv.COLOR_RGBA2GRAY);
    return g;
  }
  function resizeMax(cv, src, side, S) {
    const s = Math.min(1, side / Math.max(src.rows, src.cols));
    if (s >= 1) return src;
    const d = S.add(new cv.Mat());
    cv.resize(src, d, new cv.Size(Math.round(src.cols * s), Math.round(src.rows * s)), 0, 0, cv.INTER_AREA);
    return d;
  }
  const blur32 = (cv, src, sigma, S) => { const d = S.add(new cv.Mat()); cv.GaussianBlur(src, d, new cv.Size(0, 0), sigma); return d; };
  function to32(cv, g8, S) { const d = S.add(new cv.Mat()); g8.convertTo(d, cv.CV_32F); return d; }

  // ------------------------------------------------------------ ArUco markers (objdetect)
  function detectMarkers(cv, gray8) {
    const S = Scope();
    try {
      const dict = S.add(cv.getPredefinedDictionary(cv.DICT_4X4_50));
      const params = S.add(new cv.aruco_DetectorParameters());
      try { params.cornerRefinementMethod = cv.CORNER_REFINE_SUBPIX; params.cornerRefinementWinSize = 5; } catch (e) {}
      const refine = S.add(new cv.aruco_RefineParameters(10, 3, true));
      const det = S.add(new cv.aruco_ArucoDetector(dict, params, refine));
      const corners = S.add(new cv.MatVector()), ids = S.add(new cv.Mat()), rej = S.add(new cv.MatVector());
      det.detectMarkers(gray8, corners, ids, rej);
      const out = [];
      for (let i = 0; i < ids.rows * ids.cols; i++) {
        const id = ids.intAt(i, 0), c = corners.get(i), d = c.data32F;
        if (id in MARKER_SIZES_MM) out.push({ id, quad: [[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]] });
        c.delete();
      }
      return out;
    } finally { S.free(); }
  }
  const sideLengths = q => q.map((p, i) => Math.hypot(p[0] - q[(i + 1) % 4][0], p[1] - q[(i + 1) % 4][1]));
  const polyArea = q => Math.abs(q.reduce((s, p, i) => s + p[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * p[1], 0)) / 2;

  function bilinear(g, w, h, x, y) { // g: Float32Array/Uint8Array, BORDER_REPLICATE
    x = Math.min(Math.max(x, 0), w - 1); y = Math.min(Math.max(y, 0), h - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1), fx = x - x0, fy = y - y0;
    return (g[y0 * w + x0] * (1 - fx) + g[y0 * w + x1] * fx) * (1 - fy) + (g[y1 * w + x0] * (1 - fx) + g[y1 * w + x1] * fx) * fy;
  }

  /* Blur sigma from the marker's outer black->white edges (10-90 % rise of a Gaussian-blurred step = 2.563 sigma). */
  function edgeBlurSigma(gray8, quad) {
    const w = gray8.cols, h = gray8.rows, g = gray8.data;
    const side = sideLengths(quad).reduce((a, b) => a + b, 0) / 4;
    if (side < 24) return null;
    const reach = Math.min(0.12 * side, 12);
    const c = [0, 1].map(k => quad.reduce((s, p) => s + p[k], 0) / 4);
    const t = []; for (let v = -reach; v <= reach + 0.05; v += 0.1) t.push(v);
    const sig = [];
    for (let i = 0; i < 4; i++) {
      const a = quad[i], b = quad[(i + 1) % 4], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const d = [(b[0] - a[0]) / L, (b[1] - a[1]) / L]; let n = [-d[1], d[0]];
      if (n[0] * ((a[0] + b[0]) / 2 - c[0]) + n[1] * ((a[1] + b[1]) / 2 - c[1]) < 0) n = [-n[0], -n[1]];
      for (let k = 0; k < 9; k++) {
        const f = 0.2 + 0.6 * k / 8, p = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
        const prof = t.map(tt => bilinear(g, w, h, p[0] + tt * n[0], p[1] + tt * n[1]));
        const q4 = Math.floor(prof.length / 4);
        const lo = median(prof.slice(0, q4)), hi = median(prof.slice(-q4));
        if (hi - lo < 40) continue;
        const nn = prof.map(v => (v - lo) / (hi - lo));
        const i10 = nn.findIndex(v => v >= 0.1), i90 = nn.findIndex(v => v >= 0.9);
        if (i10 < 0 || i90 <= i10) continue;
        sig.push((i90 - i10) * 0.1 / 2.563);
      }
    }
    return sig.length >= 6 ? median(sig) : null;
  }

  function estimateScale(cv, gray8) {
    const found = detectMarkers(cv, gray8);
    if (!found.length) return { found: false, note: "no SetuScan marker detected" };
    const best = found.reduce((a, b) => (polyArea(b.quad) > polyArea(a.quad) ? b : a));
    const mm = MARKER_SIZES_MM[best.id], sides = sideLengths(best.quad);
    const native = sides.reduce((a, b) => a + b, 0) / 4 / mm, tilt = Math.max(...sides) / Math.max(Math.min(...sides), 1e-6);
    const S = Scope();
    try {
      const src = S.add(cv.matFromArray(4, 1, cv.CV_32FC2, best.quad.flat()));
      const dst = S.add(cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, mm, 0, mm, mm, 0, mm]));
      const Hm = S.add(cv.findHomography(src, dst, 0));
      const H = Array.from(Hm.data64F);
      const w = gray8.cols, h = gray8.rows;
      const warp = (x, y) => { const z = H[6] * x + H[7] * y + H[8]; return [(H[0] * x + H[1] * y + H[2]) / z, (H[3] * x + H[4] * y + H[5]) / z]; };
      const wp = [warp(0, 0), warp(w, 0), warp(w, h), warp(0, h)];
      const lim = 0.75 * Math.max(h, w) / native, cx = mm / 2;
      const xs = wp.map(p => p[0]), ys = wp.map(p => p[1]);
      const view = [Math.max(Math.min(...xs), cx - lim), Math.max(Math.min(...ys), cx - lim), Math.min(Math.max(...xs), cx + lim), Math.min(Math.max(...ys), cx + lim)];
      const blur = edgeBlurSigma(gray8, best.quad);
      return { found: true, marker_id: best.id, marker_mm: mm, corners: best.quad, px_per_mm_native: native, tilt_ratio: tilt,
        H_mm: H, view_mm: view, n_markers: found.length, edge_blur_px: blur, edge_blur_mm: blur == null ? null : blur / native, note: "" };
    } finally { S.free(); }
  }

  // ------------------------------------------------------------ quality gate
  function assessQuality(cv, gray8full) {
    const S = Scope();
    try {
      const gray = resizeMax(cv, gray8full, 1024, S);
      const g32 = to32(cv, gray, S);
      const sm = blur32(cv, g32, 1.0, S), lap = S.add(new cv.Mat());
      cv.Laplacian(sm, lap, cv.CV_32F, 3);
      const mean = S.add(new cv.Mat()), std = S.add(new cv.Mat());
      cv.meanStdDev(lap, mean, std);
      const sharp = std.data64F[0] ** 2;
      const gm = S.add(new cv.Mat()), gs = S.add(new cv.Mat());
      cv.meanStdDev(gray, gm, gs);
      const meanG = gm.data64F[0], N = gray.rows * gray.cols;
      const d = gray.data; let dark = 0, bright = 0;
      for (let i = 0; i < N; i++) { if (d[i] < 25) dark++; if (d[i] > 245) bright++; }
      // glare: saturated pixels in flat neighbourhoods, minus the (bright, flat by design) marker card
      const m1 = blur32(cv, g32, 3, S), sq = S.add(new cv.Mat()); cv.multiply(g32, g32, sq);
      const m2 = blur32(cv, sq, 3, S);
      const a1 = m1.data32F, a2 = m2.data32F, mask = new Uint8Array(N);
      for (let i = 0; i < N; i++) mask[i] = d[i] >= 250 && Math.sqrt(Math.max(a2[i] - a1[i] * a1[i], 0)) < 4 ? 1 : 0;
      const mm = S.add(u8Mat(cv, gray.rows, gray.cols, mask));
      const k3 = S.add(cv.Mat.ones(3, 3, cv.CV_8U)), op = S.add(new cv.Mat());
      cv.morphologyEx(mm, op, cv.MORPH_OPEN, k3);
      const sc = gray.cols / gray8full.cols;
      for (const { quad } of detectMarkers(cv, gray)) {
        const c = [0, 1].map(k => quad.reduce((s, p) => s + p[k], 0) / 4), f = 1 + 2 * CARD_MARGIN + 0.3;
        const pts = quad.map(p => [c[0] + (p[0] - c[0]) * f, c[1] + (p[1] - c[1]) * f]);
        const pm = S.add(cv.matFromArray(4, 1, cv.CV_32SC2, pts.flat().map(Math.round)));
        const pv = S.add(new cv.MatVector()); pv.push_back(pm);
        cv.fillPoly(op, pv, new cv.Scalar(0));
      }
      const lab = S.add(new cv.Mat()), st = S.add(new cv.Mat()), ce = S.add(new cv.Mat());
      const n = cv.connectedComponentsWithStats(op, lab, st, ce, 8, cv.CV_32S);
      let big = 0; for (let i = 1; i < n; i++) { const a = st.intAt(i, 4); if (a >= 30) big += a; }
      const glare = big / N;
      const issues = [], advice = [];
      if (sharp < QTH.min_sharpness) { issues.push("blurry"); advice.push("Hold the phone steady, tap to focus on the crack, and retake."); }
      if (meanG < QTH.min_mean || dark / N > QTH.max_dark_frac) { issues.push("too_dark"); advice.push("Add light (torch at an angle) or move to daylight and retake."); }
      if (meanG > QTH.max_mean || bright / N > QTH.max_bright_frac) { issues.push("overexposed"); advice.push("Avoid direct sun on the surface; shade it or lower exposure."); }
      if (glare > QTH.max_glare_frac) { issues.push("glare"); advice.push("Change the angle so the light does not reflect straight into the camera."); }
      return { sharpness: sharp, mean: meanG, dark_frac: dark / N, bright_frac: bright / N, glare_frac: glare, ok: !issues.length, issues, advice };
    } finally { S.free(); }
  }

  // ------------------------------------------------------------ views (rectified working windows)
  function mulH(A, B) { const R = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) R[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j]; return R; }
  function makeView(H, unit, box, scale, native) {
    const M = mulH([scale, 0, -scale * box[0], 0, scale, -scale * box[1], 0, 0, 1], H);
    return { H_base: H, unit, box, scale, native_scale: native, M,
      size: [Math.max(1, Math.ceil((box[2] - box[0]) * scale)), Math.max(1, Math.ceil((box[3] - box[1]) * scale))],
      toBase(x, y) { return [box[0] + x / scale, box[1] + y / scale]; } };
  }
  function baseView(w, h, scale) {
    if (scale.found) {
      const b = scale.view_mm, span = Math.max(b[2] - b[0], b[3] - b[1]);
      return makeView(scale.H_mm, "mm", b, Math.min(scale.px_per_mm_native, WORK_SIDE / span), scale.px_per_mm_native);
    }
    return makeView([1, 0, 0, 0, 1, 0, 0, 0, 1], "px", [0, 0, w, h], Math.min(1, WORK_SIDE / Math.max(w, h)), 1);
  }
  function zoomView(v, bbox) {
    const [x, y, bw, bh] = bbox, pad = 0.15 * Math.max(bw, bh) + 10;
    let [bx0, by0] = v.toBase(x - pad, y - pad), [bx1, by1] = v.toBase(x + bw + pad, y + bh + pad);
    bx0 = Math.max(bx0, v.box[0]); by0 = Math.max(by0, v.box[1]); bx1 = Math.min(bx1, v.box[2]); by1 = Math.min(by1, v.box[3]);
    const span = Math.max(bx1 - bx0, by1 - by0);
    return makeView(v.H_base, v.unit, [bx0, by0, bx1, by1], Math.min(v.native_scale, ZOOM_SIDE / span), v.native_scale);
  }
  function render(cv, src8, v, S, interp, border) {
    const M = S.add(cv.matFromArray(3, 3, cv.CV_64F, v.M)), out = S.add(new cv.Mat());
    cv.warpPerspective(src8, out, M, new cv.Size(v.size[0], v.size[1]), interp ?? (v.scale < 1 ? cv.INTER_AREA : cv.INTER_LINEAR), border ?? cv.BORDER_REPLICATE, new cv.Scalar(0, 0, 0, 0));
    return out;
  }
  function viewBlurSigma(cv, gray8, v, scale) {
    if (v.unit !== "mm" || !scale.found) return null;
    const S = Scope();
    try {
      const mm = scale.marker_mm, pad = 0.3 * mm;
      const mv = makeView(v.H_base, "mm", [-pad, -pad, mm + pad, mm + pad], v.scale, v.native_scale);
      const patch = render(cv, gray8, mv, S);
      const quad = [[0, 0], [mm, 0], [mm, mm], [0, mm]].map(p => [(p[0] + pad) * v.scale, (p[1] + pad) * v.scale]);
      return edgeBlurSigma(patch, quad);
    } finally { S.free(); }
  }
  function segParamsFor(v) {
    const clip = (x, a, b) => Math.min(Math.max(x, a), b);
    return v.unit === "mm"
      ? { max_width_px: clip(Math.trunc(MAX_CRACK_MM * v.scale), 11, 61), min_len_px: Math.max(25, MIN_CRACK_LEN_MM * v.scale) }
      : { max_width_px: clip(Math.trunc(21 * v.scale), 11, 41), min_len_px: Math.max(25, 40 * v.scale) };
  }

  // ------------------------------------------------------------ segmentation (imgproc)
  const odd = k => Math.max(3, k | 1);
  /* Black-hat depth (closing minus image). Returns both variants from ONE closing: smooth (detection) and raw (width). */
  function blackHat(cv, gray8, maxW, S) {
    const g = S.add(new cv.Mat()); cv.GaussianBlur(gray8, g, new cv.Size(3, 3), 0);
    const closed = S.add(new cv.Mat()), k = odd(maxW + 4);
    const ker = S.add(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k)));
    cv.morphologyEx(g, closed, cv.MORPH_CLOSE, ker, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    const N = g.rows * g.cols, smooth = new Float32Array(N), raw = new Float32Array(N), c = closed.data, gd = g.data, r = gray8.data;
    for (let i = 0; i < N; i++) { smooth[i] = Math.max(c[i] - gd[i], 0); raw[i] = Math.max(c[i] - r[i], 0); }
    return { smooth, raw };
  }
  function ridgeStrength(cv, gray8, sigmas, S) {
    const f = to32(cv, gray8, S), N = f.rows * f.cols, best = new Float32Array(N);
    for (const s of sigmas) {
      const g = blur32(cv, f, s, S), dxx = S.add(new cv.Mat()), dyy = S.add(new cv.Mat()), dxy = S.add(new cv.Mat());
      cv.Sobel(g, dxx, cv.CV_32F, 2, 0, 3); cv.Sobel(g, dyy, cv.CV_32F, 0, 2, 3); cv.Sobel(g, dxy, cv.CV_32F, 1, 1, 3);
      const a = dxx.data32F, b = dyy.data32F, c = dxy.data32F, s2 = s * s;
      for (let i = 0; i < N; i++) {
        const tmp = Math.sqrt((a[i] - b[i]) ** 2 + 4 * c[i] * c[i]);
        const l1 = 0.5 * (a[i] + b[i] + tmp), l2 = 0.5 * (a[i] + b[i] - tmp);
        const r = Math.max(l1 - Math.abs(l2), 0) * s2;
        if (r > best[i]) best[i] = r;
      }
    }
    return best;
  }
  function robustNoise(x, valid) {
    const v = []; const step = Math.max(1, Math.floor(x.length / 200000));
    for (let i = 0; i < x.length; i += step) if (valid[i]) v.push(x[i]);
    if (!v.length) return 1;
    const med = median(v), mad = median(v.map(a => Math.abs(a - med))) * 1.4826;
    return Math.max(mad, 0.5);
  }
  /* Zhang-Suen thinning on a Uint8Array 0/1 (w*h). Returns new 0/1 array. */
  function thin(bin, w, h) {
    const img = Uint8Array.from(bin), rm = [];
    let changed = true;
    const P = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : img[y * w + x]);
    while (changed) {
      changed = false;
      for (let step = 0; step < 2; step++) {
        rm.length = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          if (!img[y * w + x]) continue;
          const p2 = P(x, y - 1), p3 = P(x + 1, y - 1), p4 = P(x + 1, y), p5 = P(x + 1, y + 1), p6 = P(x, y + 1), p7 = P(x - 1, y + 1), p8 = P(x - 1, y), p9 = P(x - 1, y - 1);
          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9; if (B < 2 || B > 6) continue;
          const seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]; let A = 0; for (let i = 0; i < 8; i++) if (!seq[i] && seq[i + 1]) A++;
          if (A !== 1) continue;
          if (step === 0 ? (p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0) : (p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0)) rm.push(y * w + x);
        }
        if (rm.length) { changed = true; for (const i of rm) img[i] = 0; }
      }
    }
    return img;
  }
  function labelComponents(cv, bin8, S) { // bin8: Mat CV_8U 0/255 -> {n, labels Int32Array, stats}
    const lab = S.add(new cv.Mat()), st = S.add(new cv.Mat()), ce = S.add(new cv.Mat());
    const n = cv.connectedComponentsWithStats(bin8, lab, st, ce, 8, cv.CV_32S);
    return { n, labels: lab.data32S, stats: i => [st.intAt(i, 0), st.intAt(i, 1), st.intAt(i, 2), st.intAt(i, 3), st.intAt(i, 4)] };
  }
  function maxFeret(pts) { // pts: array of [x,y]; hull via monotone chain
    if (pts.length < 2) return 1;
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    const hull = lo.slice(0, -1).concat(up.slice(0, -1)); let m = 0;
    for (let i = 0; i < hull.length; i++) for (let j = i + 1; j < hull.length; j++) m = Math.max(m, Math.hypot(hull[i][0] - hull[j][0], hull[i][1] - hull[j][1]));
    return m;
  }

  function segment(cv, gray8, valid, P) {
    const S = Scope();
    try {
      const w = gray8.cols, h = gray8.rows, N = w * h;
      const bh = blackHat(cv, gray8, P.max_width_px, S), depth = bh.smooth;
      const ridge = ridgeStrength(cv, gray8, [1.0, 2.0, 3.5], S);
      for (let i = 0; i < N; i++) if (!valid[i]) { depth[i] = 0; ridge[i] = 0; }
      const nd = robustNoise(depth, valid), nr = robustNoise(ridge, valid);
      const kHigh = 5.0, kLow = 3.0, minContrast = 6.0, minElong = 10.0, minFill = 0.5, maxAsym = 0.13;
      const score = new Float32Array(N), weak = new Uint8Array(N);
      for (let i = 0; i < N; i++) { score[i] = Math.sqrt((depth[i] / nd) * (ridge[i] / nr)); weak[i] = score[i] >= kLow ? 255 : 0; }
      const weakM = S.add(u8Mat(cv, h, w, weak));
      const L0 = labelComponents(cv, weakM, S), keep = new Uint8Array(L0.n);
      for (let i = 0; i < N; i++) if (score[i] >= kHigh && depth[i] >= minContrast) keep[L0.labels[i]] = 1;
      keep[0] = 0;
      const cand = new Uint8Array(N); for (let i = 0; i < N; i++) cand[i] = keep[L0.labels[i]] ? 255 : 0;
      const candM = S.add(u8Mat(cv, h, w, cand));
      const gs = blur32(cv, to32(cv, gray8, S), 1.5, S), gx = S.add(new cv.Mat()), gy = S.add(new cv.Mat());
      cv.Sobel(gs, gx, cv.CV_32F, 1, 0, 3); cv.Sobel(gs, gy, cv.CV_32F, 0, 1, 3);
      const gX = gx.data32F, gY = gy.data32F;
      const L = labelComponents(cv, candM, S), mask = new Uint8Array(N); let rejected = 0; const comps = [];
      // bucket pixels per label once
      const pix = Array.from({ length: L.n }, () => []);
      for (let i = 0; i < N; i++) if (L.labels[i]) pix[L.labels[i]].push(i);
      for (let id = 1; id < L.n; id++) {
        const [x, y, bw, bh, area] = L.stats(id);
        if (Math.max(bw, bh) < P.min_len_px * 0.5) { rejected++; continue; }
        const idx = pix[id], pts = idx.map(i => [i % w, (i / w) | 0]);
        const extent = maxFeret(pts), elong = extent * extent / Math.max(idx.length, 1);
        if (extent < P.min_len_px || elong < minElong) { rejected++; continue; }
        const sub = new Uint8Array(bw * bh); for (const i of idx) sub[((i / w | 0) - y) * bw + (i % w) - x] = 1;
        const sk = thin(sub, bw, bh), dsk = [], gsk = [];
        for (let yy = 0; yy < bh; yy++) for (let xx = 0; xx < bw; xx++) if (sk[yy * bw + xx]) { const gi = (y + yy) * w + x + xx; dsk.push(depth[gi]); gsk.push(Math.hypot(gX[gi], gY[gi]) / 8); }
        let asym = null;
        if (dsk.length) { asym = median(gsk.map((g, i) => g * 1.5 / Math.max(dsk[i], 1))); if (asym > maxAsym) { rejected++; continue; } }
        const core = percentile(idx.map(i => depth[i]), 90);
        for (const i of idx) if (depth[i] >= minFill * core) mask[i] = 255;
        comps.push({ bbox: [x, y, bw, bh], extent_px: round(extent, 1), elongation: round(elong, 1), core_contrast: round(core, 1), asymmetry: round(asym, 3) });
      }
      const depthRaw = bh.raw;
      return { mask, depthRaw, comps, rejected, noise: nd };
    } finally { S.free(); }
  }

  // ------------------------------------------------------------ measurement
  function deblurWidth(ratio, sigma) {
    if (sigma <= 0) return ratio;
    const k = 2 * Math.SQRT2 * sigma;
    if (ratio <= Math.sqrt(2 * Math.PI) * sigma * 1.02) return null;
    const erf = x => { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
      return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); };
    let lo = 0, hi = ratio;
    for (let i = 0; i < 50; i++) { const mid = 0.5 * (lo + hi); if (mid - ratio * erf(mid / k) < 0) lo = mid; else hi = mid; }
    return 0.5 * (lo + hi);
  }
  function profileWidth(depth, w, h, x, y, angDeg, halfLen) {
    const a = (angDeg + 90) * Math.PI / 180, ta = angDeg * Math.PI / 180, step = 0.1, n = Math.floor((2 * halfLen) / step + 1e-9) + 1;
    const prof = new Float64Array(n);
    for (let j = 0; j < n; j++) { const t = -halfLen + j * step; let s = 0;
      for (let o = -2; o <= 2; o++) { const off = o * 0.75; s += bilinearR(depth, w, h, x + off * Math.cos(ta) + t * Math.cos(a), y + off * Math.sin(ta) + t * Math.sin(a)); }
      prof[j] = s / 5; }
    const q = Math.max(3, Math.floor(n / 8)), base = median([...prof.slice(0, q), ...prof.slice(-q)]);
    for (let j = 0; j < n; j++) prof[j] -= base;
    const c = Math.floor(n / 2), win = Math.max(1, Math.floor(1.5 / step));
    let k = c - win, peak = -Infinity; for (let j = c - win; j <= Math.min(c + win, n - 1); j++) if (prof[j] > peak) { peak = prof[j]; k = j; }
    if (peak <= 0) return null;
    const half = peak / 2; let left = k, right = k;
    while (left > 0 && prof[left] > half) left--;
    while (right < n - 1 && prof[right] > half) right++;
    if (left === 0 || right === n - 1) return null;
    const cross = (i0, i1) => { const v0 = prof[i0], v1 = prof[i1]; return v1 !== v0 ? i0 + (half - v0) / (v1 - v0) : i0; };
    const tl = cross(left, left + 1), tr = cross(right - 1, right), ext = 0.5 * (tr - tl);
    const i0 = Math.max(0, Math.floor(tl - ext)), i1 = Math.min(n - 1, Math.ceil(tr + ext));
    let area = 0; for (let j = i0; j <= i1; j++) area += Math.max(prof[j], 0); area *= step;
    return area / peak;
  }
  function bilinearR(g, w, h, x, y) { // BORDER_REFLECT_101-ish via clamp-reflect
    const rf = (v, m) => { if (m === 1) return 0; v = Math.abs(v); const p = 2 * (m - 1); v = v % p; return v > m - 1 ? p - v : v; };
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const at = (xx, yy) => g[rf(yy, h) * w + rf(xx, w)];
    return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
  }
  function localDirection(xs, ys, i, radius = 6) {
    let n = 0, mx = 0, my = 0; const sel = [];
    for (let j = 0; j < xs.length; j++) if (Math.abs(xs[j] - xs[i]) <= radius && Math.abs(ys[j] - ys[i]) <= radius) { sel.push(j); mx += xs[j]; my += ys[j]; n++; }
    if (n < 3) return 0; mx /= n; my /= n;
    let sxx = 0, syy = 0, sxy = 0; for (const j of sel) { const dx = xs[j] - mx, dy = ys[j] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
    return 0.5 * Math.atan2(2 * sxy, sxx - syy) * 180 / Math.PI;
  }
  function skeletonLength(cv, skel, w, h) { // half the perimeter of the simplified outer contour (as in Python)
    const S = Scope();
    try {
      const m = S.add(u8Mat(cv, h, w, skel)), cs = S.add(new cv.MatVector()), hi = S.add(new cv.Mat());
      cv.findContours(m, cs, hi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
      let total = 0;
      for (let i = 0; i < cs.size(); i++) { const c = cs.get(i); if (c.rows >= 2) { const ap = new cv.Mat(); cv.approxPolyDP(c, ap, 1.0, true); total += cv.arcLength(ap, true) / 2; ap.delete(); } c.delete(); }
      return total;
    } finally { S.free(); }
  }
  function measure(cv, seg, w, h, opts) {
    const S = Scope();
    try {
      if (!seg.mask.some(v => v)) return [];
      const maskM = S.add(u8Mat(cv, h, w, seg.mask)), dtM = S.add(new cv.Mat());
      cv.distanceTransform(maskM, dtM, cv.DIST_L2, 0);
      const dt = dtM.data32F, skel = thin(Uint8Array.from(seg.mask, v => (v ? 1 : 0)), w, h);
      const sk8 = Uint8Array.from(skel, v => v * 255), skM = S.add(u8Mat(cv, h, w, sk8));
      const L = labelComponents(cv, skM, S), out = [];
      for (let id = 1; id < L.n; id++) {
        const xs = [], ys = [];
        for (let i = 0; i < w * h; i++) if (L.labels[i] === id) { xs.push(i % w); ys.push((i / w) | 0); }
        const comp = new Uint8Array(w * h); xs.forEach((x, k) => (comp[ys[k] * w + x] = 255));
        const length = skeletonLength(cv, comp, w, h); if (length < opts.min_len_px) continue;
        const rangeX = Math.max(...xs) - Math.min(...xs), rangeY = Math.max(...ys) - Math.min(...ys);
        const order = xs.map((_, k) => k).sort((a, b) => (rangeX >= rangeY ? xs[a] - xs[b] || ys[a] - ys[b] : ys[a] - ys[b] || xs[a] - xs[b]));
        const stride = Math.max(1, Math.round(opts.sample_every_px)), samples = []; let below = 0;
        for (let o = 0; o < order.length; o += stride) {
          const j = order[o], x = xs[j], y = ys[j], dtw = 2 * dt[y * w + x], ang = localDirection(xs, ys, j);
          const raw = profileWidth(seg.depthRaw, w, h, x, y, ang, Math.max(4, 3 * dtw)); if (raw === null) continue;
          let pw = raw;
          if (opts.blur_sigma_px) { const d = deblurWidth(raw, opts.blur_sigma_px); if (d === null) below++; else pw = d; }
          samples.push({ x, y, width_px: pw, dt_width_px: dtw, raw });
        }
        if (!samples.length) continue;
        const ww = samples.map(s => s.width_px);
        const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
        let sxx = 0, syy = 0, sxy = 0; xs.forEach((x, k) => { const dx = x - mx, dy = ys[k] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; });
        const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), x0 = Math.min(...xs), y0 = Math.min(...ys);
        out.push({ length_px: length, width_max_px: percentile(ww, 95), width_median_px: median(ww), orientation_deg: Math.abs(th) * 180 / Math.PI,
          bbox: [x0, y0, rangeX + 1, rangeY + 1], n_samples: samples.length, n_below_blur: below,
          raw_width_median_px: median(samples.map(s => s.raw)), pts: samples.map(s => [s.x, s.y, s.width_px]) });
      }
      out.sort((a, b) => b.length_px - a.length_px); out.forEach((c, k) => (c.id = k + 1));
      return out;
    } finally { S.free(); }
  }

  // ------------------------------------------------------------ analyse one view
  function analyzeView(cv, gray8, v, scale) {
    const S = Scope(), t0 = performance.now();
    try {
      const work = render(cv, gray8, v, S), onesM = S.add(cv.Mat.ones(gray8.rows, gray8.cols, cv.CV_8U));
      cv.convertScaleAbs(onesM, onesM, 255, 0);
      const validM = render(cv, onesM, v, S, cv.INTER_NEAREST, cv.BORDER_CONSTANT), er = S.add(new cv.Mat());
      cv.erode(validM, er, S.add(cv.Mat.ones(7, 7, cv.CV_8U)), new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
      const w = work.cols, h = work.rows;
      if (scale.found && v.unit === "mm") {
        const mm = scale.marker_mm, m = CARD_MARGIN * mm + CARD_GUARD_MM;
        const poly = [[-m, -m], [mm + m, -m], [mm + m, mm + m], [-m, mm + m]].map(p => [Math.round((p[0] - v.box[0]) * v.scale), Math.round((p[1] - v.box[1]) * v.scale)]);
        const pm = S.add(cv.matFromArray(4, 1, cv.CV_32SC2, poly.flat())), pv = S.add(new cv.MatVector()); pv.push_back(pm);
        cv.fillPoly(er, pv, new cv.Scalar(0));
      }
      const valid = Uint8Array.from(er.data), t1 = performance.now(), P = segParamsFor(v);
      const seg = segment(cv, work, valid, P), sigma = viewBlurSigma(cv, gray8, v, scale), t2 = performance.now();
      const cracks = measure(cv, seg, w, h, { sample_every_px: Math.max(2, v.unit === "mm" ? 0.5 * v.scale : 3), min_len_px: P.min_len_px * 0.6, blur_sigma_px: sigma });
      const t3 = performance.now(), mmpp = v.unit === "mm" ? 1 / v.scale : null;
      const summ = cracks.map(c => {
        const d = { id: c.id, length_px: round(c.length_px, 1), width_max_px: round(c.width_max_px, 2), width_median_px: round(c.width_median_px, 2),
          orientation_deg: round(c.orientation_deg, 1), bbox_px: c.bbox, n_width_samples: c.n_samples,
          width_reliable: c.width_max_px >= MIN_RELIABLE_WIDTH_PX && c.n_below_blur <= 0.2 * c.n_samples,
          raw_width_median_px: round(c.raw_width_median_px, 2), samples_below_blur_limit: c.n_below_blur };
        if (mmpp) { d.length_mm = round(c.length_px * mmpp, 1); d.width_max_mm = round(c.width_max_px * mmpp, 3); d.width_median_mm = round(c.width_median_px * mmpp, 3);
          const [e] = nearestEdge(d.width_max_mm); d.nearest_edge_mm = e; d.ambiguous = isAmbiguous(d.width_max_mm, mmpp); }
        else { d.nearest_edge_mm = null; d.ambiguous = false; }
        d.zoomable = v.scale < 0.9 * v.native_scale; d.zoomed = false; d.width_source = "working view"; return d; });
      // overlay (RGBA) for display: crack pixels orange on the rectified view
      const rgba = new Uint8ClampedArray(w * h * 4), wd = work.data;
      for (let i = 0; i < w * h; i++) { const m = seg.mask[i]; rgba[4 * i] = m ? 255 : wd[i]; rgba[4 * i + 1] = m ? 140 : wd[i]; rgba[4 * i + 2] = m ? 0 : wd[i]; rgba[4 * i + 3] = 255; }
      return { view: v, w, h, cracks: summ, raw_cracks: cracks, rejected: seg.rejected, mm_per_px: mmpp, blur_sigma_px: sigma,
        overlay: { data: rgba, width: w, height: h }, timings_ms: { render: round(t1 - t0, 0), segment: round(t2 - t1, 0), measure: round(t3 - t2, 0) } };
    } finally { S.free(); }
  }

  // ------------------------------------------------------------ the agent (rules decider; same policy as setuscan/deciders.py)
  /* Runs observe -> decide -> act until a terminal state. `decide` is pluggable: the default is the deterministic rules. */
  function runAgent(cv, imageData, opts = {}) {
    const trace = [], t_start = performance.now(), S = Scope();
    const state = { quality: null, scale: null, analysis: null, cracks: [], classification: null, zooms: 0, status: "open", reshoot: null };
    let gray;
    const log = (action, args, rationale, result, decider = "rules (in-browser)") => {
      trace.push({ step: trace.length + 1, decider, action, args, rationale, result, status: state.status });
      if (opts.onStep) opts.onStep(trace[trace.length - 1]);
    };
    try {
      gray = grayMat(cv, imageData, S);
      const allowed = () => {
        const a = { request_reshoot: {} };
        if (!state.quality) a.check_quality = {};
        else if (state.quality.ok && !state.scale) a.detect_scale = {};
        else if (state.quality.ok && state.scale && !state.analysis) a.segment_measure = {};
        if (state.analysis && state.zooms < 3) { const ids = state.cracks.filter(c => c.zoomable && !c.zoomed).map(c => c.id); if (ids.length) a.zoom_measure = { crack_ids: ids }; }
        if (state.analysis && !state.classification) a.classify = {};
        if (state.classification) a[state.classification.needs_engineer ? "request_approval" : "close_case"] = {};
        else if (state.analysis && !state.cracks.length) a.close_case = {};
        return a;
      };
      const reshoot = (reasons, why, crack) => {
        const ADV = { blurry_marker: "The photo is out of focus or shaken: tap to focus on the crack, hold steady (or brace the phone), and retake.",
          no_marker: "Stick the printed SetuScan marker flat on the surface next to the crack and retake.", too_far: "Move closer (about half the distance) keeping the marker and the crack in view." };
        state.reshoot = { reasons, advice: [...(state.quality && !state.quality.ok ? state.quality.advice : []), ...reasons.filter(r => ADV[r]).map(r => ADV[r])], crack_id: crack ?? null };
        state.status = "needs_reshoot"; log("request_reshoot", { visit: "current", reasons }, why, state.reshoot);
      };
      for (let step = 0; step < 24 && state.status === "open"; step++) {
        const al = allowed(), q = state.quality, sc = state.scale;
        if (!q) { state.quality = assessQuality(cv, gray); log("check_quality", { visit: "current" }, "No quality check yet for the current photo.", state.quality); continue; }
        if (!q.ok) { reshoot(q.issues, `Current photo failed the quality gate (${q.issues.join(", ")}); measuring it would give unreliable widths.`); break; }
        if (!sc) { state.scale = estimateScale(cv, gray); const r = { ...state.scale }; delete r.H_mm; log("detect_scale", { visit: "current" }, "Need px-to-mm scale for the current photo.", r); continue; }
        if (sc.found && sc.edge_blur_px != null && sc.edge_blur_px > MAX_EDGE_BLUR_PX) {
          reshoot(["blurry_marker"], `The marker's printed edges are blurred by sigma ${sc.edge_blur_px.toFixed(1)} px (${sc.edge_blur_mm.toFixed(2)} mm), above the ${MAX_EDGE_BLUR_PX} px limit: crack widths would be over-read.`); break; }
        if (!state.analysis) {
          const v = baseView(gray.cols, gray.rows, sc); const a = analyzeView(cv, gray, v, sc); state.analysis = a; state.cracks = a.cracks;
          log("segment_measure", { visit: "current" }, "Segment and measure cracks in the current photo.", { n_cracks: a.cracks.length, mm_per_px: round(a.mm_per_px, 4), view: { unit: v.unit, scale_px_per_unit: round(v.scale, 4), size: v.size }, rejected_components: a.rejected, timings_ms: a.timings_ms }); continue; }
        if (!sc.found) {
          if (state.cracks.length) { reshoot(["no_marker"], `${state.cracks.length} crack(s) found but no scale marker, so widths are only in pixels and cannot be compared with the mm policy.`); break; }
          state.status = "closed"; log("close_case", { outcome: "no_crack_detected" }, "No crack found and no marker present; nothing to measure.", { outcome: "no_crack_detected" }); break; }
        if (al.zoom_measure) {
          const c = [...state.cracks].sort((a, b) => b.width_max_mm - a.width_max_mm).find(c => al.zoom_measure.crack_ids.includes(c.id) && (!c.width_reliable || c.ambiguous));
          if (c) {
            const why = []; if (!c.width_reliable) why.push(`only ${c.width_max_px.toFixed(1)} px wide at the working scale`);
            if (c.ambiguous) why.push(`${c.width_max_mm.toFixed(2)} mm is close to the ${c.nearest_edge_mm} mm band edge`);
            state.zooms++; const z = zoomView(state.analysis.view, c.bbox_px), az = analyzeView(cv, gray, z, sc); c.zoomed = true;
            let res = { crack_id: c.id, found_in_zoom: false, note: "crack not re-detected at native resolution; keeping working-view values" };
            if (az.cracks.length) {
              const [bx, by, bw, bh] = c.bbox_px, [cx, cy] = state.analysis.view.toBase(bx + bw / 2, by + bh / 2), zx = (cx - z.box[0]) * z.scale, zy = (cy - z.box[1]) * z.scale;
              const best = az.cracks.reduce((a, b) => { const d = m => Math.hypot(m.bbox_px[0] + m.bbox_px[2] / 2 - zx, m.bbox_px[1] + m.bbox_px[3] / 2 - zy) - 0.5 * m.length_px; return d(b) < d(a) ? b : a; });
              const before = { width_max_mm: c.width_max_mm, width_max_px: c.width_max_px };
              c.width_max_mm = best.width_max_mm; c.width_max_px = best.width_max_px; c.width_reliable = best.width_reliable; c.width_source = `zoom view at ${(1 / az.mm_per_px).toFixed(2)} px/mm`;
              c.nearest_edge_mm = nearestEdge(c.width_max_mm)[0]; c.ambiguous = isAmbiguous(c.width_max_mm, az.mm_per_px);
              res = { crack_id: c.id, found_in_zoom: true, before, after: { width_max_mm: c.width_max_mm, width_max_px: c.width_max_px }, width_reliable: c.width_reliable, ambiguous: c.ambiguous };
              state.zoomOverlay = az.overlay;
            }
            log("zoom_measure", { visit: "current", crack_id: c.id }, `Crack #${c.id}: ${why.join(" and ")}; re-measuring it at the photo's native resolution.`, res); continue; }
        }
        const tf = state.cracks.find(c => c.zoomed && c.width_max_px < MIN_RELIABLE_WIDTH_PX && c.ambiguous);
        if (tf) { reshoot(["too_far"], `Even at native resolution crack #${tf.id} is ${tf.width_max_px.toFixed(1)} px wide and its ${tf.width_max_mm.toFixed(2)} mm reading sits on a band edge: a closer photo is needed.`, tf.id); break; }
        if (!state.classification) {
          if (!state.cracks.length) state.classification = { band: "no_crack_detected", width_max_mm: null, needs_engineer: false, growth: false, reinspect_days: 365, policy: POLICY.label };
          else { const worst = state.cracks.reduce((a, b) => (b.width_max_mm > a.width_max_mm ? b : a)), band = bandFor(worst.width_max_mm);
            state.classification = { band: band.name, action: band.action, width_max_mm: worst.width_max_mm, worst_crack: worst.id, width_source: worst.width_source, growth: false,
              needs_engineer: band.action !== "monitor", reinspect_days: band.reinspect_days, policy: POLICY.label }; }
          log("classify", {}, "All measurements are in; apply the severity policy.", state.classification); continue; }
        const cl = state.classification;
        if (cl.needs_engineer) { state.status = "awaiting_approval"; log("request_approval", {}, `Policy result '${cl.band}': an engineer must approve before a ticket is raised.`, { state: "pending" }); }
        else { state.status = "closed"; log("close_case", { outcome: cl.band }, `Policy result '${cl.band}' needs no engineer action; schedule re-inspection in ${cl.reinspect_days} days.`, { outcome: cl.band, reinspect_days: cl.reinspect_days }); }
        break;
      }
      if (state.status === "open") state.status = "error";
      return { status: state.status, trace, quality: state.quality, scale: state.scale && (({ H_mm, ...r }) => r)(state.scale), cracks: state.cracks, classification: state.classification,
        reshoot: state.reshoot, analysis: state.analysis, zoomOverlay: state.zoomOverlay || null, agent_total_ms: round(performance.now() - t_start, 0), opencv: cv.getBuildInformation().match(/OpenCV (\d+\.\d+\.\d+)/)?.[1] };
    } finally { S.free(); }
  }

  return { runAgent, assessQuality, estimateScale, analyzeView, baseView, POLICY, MARKER_SIZES_MM, version: "planb-1" };
});
