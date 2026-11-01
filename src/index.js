// In-browser PP-OCRv6 Tiny (det.onnx + rec.onnx + dict.txt) on onnxruntime-web.
// Pre/post-processing follows byrizki/rusto-rs, the engine behind imagor's ocr() metadata filter.

const DET_MEAN = [0.485, 0.456, 0.406]; // applied to B,G,R in that order, as rusto does
const DET_STD = [0.229, 0.224, 0.225];
const REC_H = 48;
const REC_MIN_W = 320;

const DEFAULTS = {
  maxSide: 960,       // detection long-side cap; probe() callers usually pass ~480
  thresh: 0.3,        // DB binarisation threshold
  boxThresh: 0.6,     // minimum mean probability inside a box
  unclip: 1.8,        // box expansion (rusto v6 preset: 2.0)
  dilate: true,
  minTextScore: 0.5,  // read(): drop recognised regions below this
  // probe() decision. These are starting points; tune them on your own photos.
  minBoxes: 2,
  minCoverage: 0.01,  // fraction of the frame covered by text boxes
};

const sizeOf = s => [s.videoWidth || s.naturalWidth || s.displayWidth || s.width, s.videoHeight || s.naturalHeight || s.displayHeight || s.height];

function defaultCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
}

export async function createTinyOcr({ modelBase = './models/', ort, canvasFactory = defaultCanvas, numThreads, executionProviders = ['wasm'] } = {}) {
  ort ??= globalThis.ort ?? await import('onnxruntime-web');
  if (numThreads) ort.env.wasm.numThreads = numThreads;
  else if (ort.env?.wasm && !globalThis.crossOriginIsolated) ort.env.wasm.numThreads = 1;
  const base = modelBase.endsWith('/') ? modelBase : modelBase + '/';
  const sessionOpts = { executionProviders, graphOptimizationLevel: 'all' };

  const [det, rec, dictText] = await Promise.all([
    ort.InferenceSession.create(base + 'det.onnx', sessionOpts),
    ort.InferenceSession.create(base + 'rec.onnx', sessionOpts),
    fetchText(base + 'dict.txt'),
  ]);
  const lines = dictText.split('\n').map(l => l.replace(/\r$/, ''));
  if (lines[lines.length - 1] === '') lines.pop();
  const chars = ['blank', ...lines, ' ']; // CTC: blank first, space appended

  const ctx2d = c => c.getContext('2d', { willReadFrequently: true });

  // ort sessions are not re-entrant across overlapping calls; serialise them.
  let chain = Promise.resolve();
  let pending = 0;
  const serial = fn => { pending++; const p = chain.then(fn).finally(() => pending--); chain = p.catch(() => {}); return p; };

  async function toSource(input) {
    if (typeof Blob !== 'undefined' && input instanceof Blob) return createImageBitmap(input, { imageOrientation: 'from-image' });
    return input;
  }

  async function detect(src, o) {
    const [w, h] = sizeOf(src);
    const ratio = Math.min(1, o.maxSide / Math.max(w, h));
    const rh = Math.max(32, Math.round(h * ratio / 32) * 32), rw = Math.max(32, Math.round(w * ratio / 32) * 32);
    const t = canvasFactory(rw, rh), tg = ctx2d(t);
    tg.fillStyle = '#fff'; tg.fillRect(0, 0, rw, rh);
    tg.imageSmoothingQuality = 'high'; tg.drawImage(src, 0, 0, rw, rh);
    const px = tg.getImageData(0, 0, rw, rh).data, plane = rw * rh;
    const input = new Float32Array(3 * plane);
    let lapSum = 0, lapSq = 0, lapN = 0;
    const lum = new Float32Array(plane);
    for (let i = 0; i < plane; i++) {
      const r = px[4 * i], g = px[4 * i + 1], b = px[4 * i + 2];
      lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      input[i] = (b / 255 - DET_MEAN[0]) / DET_STD[0];
      input[plane + i] = (g / 255 - DET_MEAN[1]) / DET_STD[1];
      input[2 * plane + i] = (r / 255 - DET_MEAN[2]) / DET_STD[2];
    }
    for (let y = 1; y < rh - 1; y++) for (let x = 1; x < rw - 1; x++) { // Laplacian variance = cheap blur measure
      const i = y * rw + x, l = lum[i - 1] + lum[i + 1] + lum[i - rw] + lum[i + rw] - 4 * lum[i];
      lapSum += l; lapSq += l * l; lapN++;
    }
    const sharpness = lapN ? lapSq / lapN - (lapSum / lapN) ** 2 : 0;

    const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, rh, rw]) });
    const od = out[det.outputNames[0]], prob = od.data, H = od.dims[od.dims.length - 2], W = od.dims[od.dims.length - 1];

    let mask = new Uint8Array(W * H);
    for (let i = 0; i < mask.length; i++) mask[i] = prob[i] > o.thresh ? 1 : 0;
    if (o.dilate) { // 2x2 kernel
      const d = new Uint8Array(mask);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (mask[i] || (x + 1 < W && mask[i + 1]) || (y + 1 < H && mask[i + W]) || (x + 1 < W && y + 1 < H && mask[i + W + 1])) d[i] = 1;
      }
      mask = d;
    }

    // connected components (8-neighbour) -> axis-aligned boxes
    const seen = new Uint8Array(W * H), stack = new Int32Array(W * H), boxes = [];
    for (let s = 0; s < mask.length; s++) {
      if (!mask[s] || seen[s]) continue;
      let sp = 0, x0 = W, y0 = H, x1 = 0, y1 = 0;
      stack[sp++] = s; seen[s] = 1;
      while (sp) {
        const i = stack[--sp], x = i % W, y = (i - x) / W;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const j = ny * W + nx;
          if (mask[j] && !seen[j]) { seen[j] = 1; stack[sp++] = j; }
        }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      if (Math.min(bw, bh) < 3) continue;
      let sum = 0;
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) sum += prob[y * W + x];
      const score = sum / (bw * bh);
      if (score < o.boxThresh) continue;
      const dist = (bw * bh * o.unclip) / (2 * (bw + bh)); // DB unclip, rectangle case
      boxes.push({
        x0: Math.max(0, (x0 - dist) * w / W), y0: Math.max(0, (y0 - dist) * h / H),
        x1: Math.min(w, (x1 + 1 + dist) * w / W), y1: Math.min(h, (y1 + 1 + dist) * h / H), score,
      });
    }
    return { boxes, width: w, height: h, sharpness };
  }

  function crop(full, b) {
    const x = Math.floor(b.x0), y = Math.floor(b.y0), cw = Math.max(1, Math.ceil(b.x1) - x), ch = Math.max(1, Math.ceil(b.y1) - y);
    let c = canvasFactory(cw, ch);
    ctx2d(c).drawImage(full, x, y, cw, ch, 0, 0, cw, ch);
    if (ch / cw >= 1.5) { // sideways text: rotate 90° counter-clockwise
      const r = canvasFactory(ch, cw), g = ctx2d(r);
      g.translate(0, cw); g.rotate(-Math.PI / 2); g.drawImage(c, 0, 0);
      c = r;
    }
    return c;
  }

  async function recognize(c) {
    const ratio = c.width / c.height;
    const imgW = Math.round(REC_H * Math.max(REC_MIN_W / REC_H, ratio));
    const rw = Math.min(imgW, Math.ceil(REC_H * ratio));
    const t = canvasFactory(rw, REC_H), tg = ctx2d(t);
    tg.imageSmoothingQuality = 'high'; tg.drawImage(c, 0, 0, rw, REC_H);
    const px = tg.getImageData(0, 0, rw, REC_H).data, plane = REC_H * imgW;
    const input = new Float32Array(3 * plane); // zero padding on the right
    for (let y = 0; y < REC_H; y++) for (let x = 0; x < rw; x++) {
      const s = 4 * (y * rw + x), d = y * imgW + x;
      input[d] = (px[s + 2] / 255 - 0.5) / 0.5; input[plane + d] = (px[s + 1] / 255 - 0.5) / 0.5; input[2 * plane + d] = (px[s] / 255 - 0.5) / 0.5;
    }
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, REC_H, imgW]) });
    const od = out[rec.outputNames[0]], T = od.dims[1], C = od.dims[2], d = od.data;
    let text = '', sum = 0, n = 0, last = -1;
    for (let t2 = 0; t2 < T; t2++) {
      let best = 0, bv = -Infinity;
      for (let k = 0; k < C; k++) { const v = d[t2 * C + k]; if (v > bv) { bv = v; best = k; } }
      if (best !== 0 && best !== last) { text += chars[best] ?? ''; sum += bv; n++; }
      last = best;
    }
    return { text, score: n ? sum / n : 0 };
  }

  /** Detection only. Cheap enough for a live camera loop. Accepts video, canvas, image, ImageBitmap or Blob. */
  function probe(input, opts = {}) {
    const o = { ...DEFAULTS, maxSide: 480, ...opts };
    return serial(async () => {
      const t0 = performance.now(), src = await toSource(input);
      const { boxes, width, height, sharpness } = await detect(src, o);
      if (src !== input && src.close) src.close();
      const area = boxes.reduce((a, b) => a + (b.x1 - b.x0) * (b.y1 - b.y0), 0);
      const coverage = Math.min(1, area / (width * height));
      const score = boxes.length ? boxes.reduce((a, b) => a + b.score, 0) / boxes.length : 0;
      return {
        hasText: boxes.length >= o.minBoxes && coverage >= o.minCoverage,
        count: boxes.length, coverage, score, sharpness, width, height, boxes, ms: performance.now() - t0,
      };
    });
  }

  /** Detection + recognition. Returns an object shaped like imagor's `ocr` metadata. */
  function read(input, opts = {}) {
    const o = { ...DEFAULTS, ...opts };
    return serial(async () => {
      const t0 = performance.now(), src = await toSource(input);
      const [w, h] = sizeOf(src);
      let full = src;
      const isCanvas = (typeof OffscreenCanvas !== 'undefined' && src instanceof OffscreenCanvas) || (typeof HTMLCanvasElement !== 'undefined' && src instanceof HTMLCanvasElement);
      if (!isCanvas) { full = canvasFactory(w, h); const g = ctx2d(full); g.fillStyle = '#fff'; g.fillRect(0, 0, w, h); g.drawImage(src, 0, 0, w, h); }
      const { boxes } = await detect(full, o), t1 = performance.now();
      const regions = [];
      for (const b of boxes) {
        const r = await recognize(crop(full, b));
        if (r.text.trim() && r.score >= o.minTextScore) regions.push({ ...b, ...r });
      }
      if (src !== input && src.close) src.close();

      regions.sort((a, b) => a.y0 - b.y0); // reading order: group by vertical overlap
      const rows = [];
      for (const r of regions) {
        const cy = (r.y0 + r.y1) / 2, row = rows.find(l => cy > l.y0 && cy < l.y1);
        if (row) { row.items.push(r); row.y0 = Math.min(row.y0, r.y0); row.y1 = Math.max(row.y1, r.y1); } else rows.push({ y0: r.y0, y1: r.y1, items: [r] });
      }
      rows.forEach(l => l.items.sort((a, b) => a.x0 - b.x0));
      const ordered = rows.flatMap(l => l.items);
      const charCount = ordered.reduce((a, r) => a + r.text.length, 0);
      return {
        engine: 'ppocrv6-tiny-onnxruntime-web', model: 'tiny', coordinate_space: 'source_image',
        width: w, height: h,
        text: rows.map(l => l.items.map(r => r.text).join(' ')).join('\n'),
        text_score: charCount ? ordered.reduce((a, r) => a + r.score * r.text.length, 0) / charCount : 0,
        char_count: charCount,
        regions: ordered.map(r => ({ text: r.text, score: +r.score.toFixed(3), box: [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]].map(q => q.map(Math.round)) })),
        timing: { detect_ms: t1 - t0, total_ms: performance.now() - t0 },
      };
    });
  }

  return { probe, read, get busy() { return pending > 0; }, dispose: async () => { await det.release?.(); await rec.release?.(); } };
}

async function fetchText(url) {
  if (typeof process !== 'undefined' && process.versions?.node && !/^https?:/.test(url)) return (await import('node:fs/promises')).readFile(url, 'utf8');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`tiny-ocr: ${url} → ${r.status}`);
  return r.text();
}
