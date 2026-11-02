#!/bin/sh
# Regenerate results/*.json. Usage: RUSTO=/path/to/rusto-rs MODELS_TINY=dir MODELS_SMALL=dir LAYOUT=layout_cdla.onnx scripts/gen.sh
set -e
cd "$(dirname "$0")/.."
cc -I"$RUSTO/include" scripts/dump.c -L"$RUSTO/target/release" -lrusto -Wl,-rpath,"$RUSTO/target/release" -o /tmp/dump
for img in clipping.jpg yearbook.jpg chronam-ad-page.png; do
  b=${img%.*}
  for tier in tiny small; do
    eval M=\$MODELS_$(echo $tier | tr a-z A-Z)
    base="\"detModelPath\":\"$M/det.onnx\",\"recModelPath\":\"$M/rec.onnx\",\"dictPath\":\"$M/dict.txt\""
    /tmp/dump "{$base}" img/$img > results/$b.$tier.json
    /tmp/dump "{$base,\"layoutModelPath\":\"$LAYOUT\",\"useLayout\":true}" img/$img '{"layout":true}' > results/$b.$tier.layout.json
  done
done
