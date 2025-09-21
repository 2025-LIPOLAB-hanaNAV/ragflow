#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-dist/offline-bundle}
OUTPUT_DIR=${2:-dist}

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Target directory '$ROOT_DIR' does not exist" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

compress() {
  local name=$1
  local src="$ROOT_DIR/$name"
  local dest="$OUTPUT_DIR/${name}.tar.gz"

  if [[ -d "$src" ]]; then
    echo "Compressing $src -> $dest"
    tar -czf "$dest" -C "$ROOT_DIR" "$name"
  fi
}

compress bundle
compress images
compress metadata
compress volumes

echo "Compression complete. Archives located in $OUTPUT_DIR"
