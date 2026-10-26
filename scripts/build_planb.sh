#!/usr/bin/env bash
# Stage the Plan B static site into dist-planb/ (no AWS calls). Pinned OpenCV 5 WebAssembly from npm.
set -euo pipefail
cd "$(dirname "$0")/.."
PKG="@techstark/opencv-js@5.0.0-release.1"
SHA512="PIm+eB0MFtieXoNC2GRao0dv/02sehG+Nv2nSW5D6pQm6J/4WqvDHm0RyoqOmGYQm67jdGiaOdIeTShCY3PIUg=="   # npm dist.integrity (sha512, base64)
JS_SHA256="b873c8211421da7b9bf41ae157a923f05a46a0b8d3e5904c44c6f3ad6d39a1bd"
OUT=dist-planb; CACHE=.scratch/opencv-js
rm -rf "$OUT"; mkdir -p "$OUT/vendor" "$OUT/data" "$CACHE"
if [ ! -f "$CACHE/opencv.js" ]; then
  tmp=$(mktemp -d); (cd "$tmp" && npm pack "$PKG" >/dev/null 2>&1 && tar xzf ./*.tgz)
  got=$(openssl dgst -sha512 -binary "$tmp"/*.tgz | base64 -w0)
  [ "$got" = "$SHA512" ] || { echo "npm tarball integrity mismatch" >&2; exit 1; }
  cp "$tmp/package/dist/opencv.js" "$CACHE/opencv.js"; cp "$tmp/package/LICENSE" "$CACHE/LICENSE"; rm -rf "$tmp"
fi
echo "$JS_SHA256  $CACHE/opencv.js" | sha256sum -c - >/dev/null || { echo "opencv.js checksum mismatch" >&2; exit 1; }
cp planb/web/index.html planb/web/worker.js planb/web/setuscan.js "$OUT/"
cp "$CACHE/opencv.js" "$OUT/vendor/opencv.js"; cp "$CACHE/LICENSE" "$OUT/vendor/OPENCV-JS-LICENSE.txt"
cp -r web/samples "$OUT/samples"
cp planb/data/python_reference.json planb/data/bedrock_status.json "$OUT/data/"
[ -d planb/data/traces ] && [ -n "$(ls -A planb/data/traces 2>/dev/null)" ] && cp -r planb/data/traces "$OUT/data/traces" || true
echo "staged $OUT ($(du -sh "$OUT" | cut -f1))"
