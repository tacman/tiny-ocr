# @tacman1123/tiny-ocr

In-browser OCR using PP-OCRv6 **Tiny** on [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web). The same
`det.onnx` / `rec.onnx` / `dict.txt` that imagor's optional `ocr()` filter (rusto) uses on the server, so a browser
result and a server result have the same shape and can be compared directly.

Two calls:

- `probe(frame)` is detection only. It is cheap enough for a live camera loop and answers "does this look like it has readable text?"
- `read(frame)` is detection + recognition. It returns an object shaped like imagor's `ocr` metadata.

```js
import { createTinyOcr } from '@tacman1123/tiny-ocr';

const ocr = await createTinyOcr({ modelBase: '/models/' }); // where det.onnx, rec.onnx, dict.txt are served

const p = await ocr.probe(videoElement);   // { hasText, count, coverage, score, sharpness, boxes, ms }
if (p.hasText) {
  const r = await ocr.read(photoBlob);     // { text, text_score, char_count, regions: [{ text, score, box }], timing }
}
```

Inputs can be a `<video>`, `<canvas>`, `OffscreenCanvas`, `<img>`, `ImageBitmap`, or a `Blob`/`File` (EXIF orientation is applied).
Box coordinates are in the source image's pixels.

## Models

The weights are not in this repo or the package (6.3 MB). Serve them yourself with long cache headers. Get them from
[byrizki/rusto-rs](https://github.com/byrizki/rusto-rs):

```sh
bash scripts/download_models.sh --model ppocrv6 --tier tiny --output-dir models
cd models && cp tiny/* . && shasum -a 256 -c SHA256SUMS
```

`models/SHA256SUMS` lists the checksums of the Tiny files this was built and tested against.

## The text gate

`probe()` returns `hasText` from two thresholds, `minBoxes` (default 2) and `minCoverage` (default 0.01 of the frame).
Those defaults are starting points, not a validated classifier. Tune them on your own photos (product boxes, envelopes,
no-text shots) before relying on them. `sharpness` (Laplacian variance) is returned for a blur check but is not part of the decision.

## Options

`createTinyOcr({ modelBase, ort, canvasFactory, numThreads, executionProviders })`. `ort` defaults to a global `ort`
(script tag) or `import('onnxruntime-web')`. Threaded wasm needs COOP/COEP headers; without them it runs single-threaded.
`probe()`/`read()` take `{ maxSide, thresh, boxThresh, unclip, dilate, minTextScore, minBoxes, minCoverage }`.

## Known limits

- Boxes are axis-aligned. Tilted text gets looser crops than rusto's rotated boxes.
- Tall boxes (height/width ≥ 1.5) are rotated 90° counter-clockwise before recognition. Upside-down text is not corrected.
- No orientation detection, no multi-page handling, no table or layout.
- Tiny's text-presence recall has not been validated on dense or unusual scripts.

## Develop

```sh
npm i && npm test        # Node: onnxruntime-node + canvas shim, needs models/ populated
python3 -m http.server   # then open /demo/ (camera needs HTTPS or localhost)
```

MIT. Model weights are PaddleOCR's; see their license.

`test/sample.png` is from [byrizki/rusto-rs](https://github.com/byrizki/rusto-rs) (MIT); the pre/post-processing here follows its implementation.
