// Runs the library in Node (onnxruntime-node + a canvas shim) against the sample invoice.
// Needs models/ populated (see README) and: npm i
import assert from 'node:assert/strict';
import * as ort from 'onnxruntime-node';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { createTinyOcr } from '../src/index.js';

const here = new URL('.', import.meta.url).pathname;
const ocr = await createTinyOcr({ modelBase: here + '../models/', ort, canvasFactory: createCanvas, executionProviders: ['cpu'] });

const img = await loadImage(here + 'sample.png');
const canvas = createCanvas(img.width, img.height);
canvas.getContext('2d').drawImage(img, 0, 0);

const p = await ocr.probe(canvas, { maxSide: 960 });
assert.equal(p.hasText, true, 'sample invoice should pass the text gate');
assert.ok(p.count > 20, `expected many boxes, got ${p.count}`);

const blank = createCanvas(400, 300);
const bctx = blank.getContext('2d'); bctx.fillStyle = '#ccc'; bctx.fillRect(0, 0, 400, 300);
const pb = await ocr.probe(blank);
assert.equal(pb.hasText, false, 'blank frame must not pass the text gate');

const r = await ocr.read(canvas);
for (const needle of ['INVOICE', 'ad4example@gmail.com', 'Invoice Date']) assert.ok(r.text.includes(needle), `missing "${needle}" in:\n${r.text}`);
assert.ok(r.text_score > 0.85);
console.log(`ok: probe ${p.count} boxes cov=${p.coverage.toFixed(2)} sharp=${p.sharpness.toFixed(0)} ${p.ms.toFixed(0)}ms | blank ${pb.count} boxes | read ${r.regions.length} regions, score ${r.text_score.toFixed(2)}, ${r.timing.total_ms.toFixed(0)}ms`);
