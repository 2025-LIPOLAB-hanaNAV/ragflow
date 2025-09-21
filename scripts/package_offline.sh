#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: package_offline.sh [options]

Build the current RAGFlow backend image, gather docker-compose dependencies,
and assemble an offline bundle that can run in an isolated environment.

Options:
  -o, --output DIR            Output directory (default: dist/offline-bundle)
  -t, --tag TAG               Tag for the built RAGFlow image (default: ragflow-offline:YYYYMMDDHHMM)
      --profiles PROFILES     Override COMPOSE_PROFILES when rendering docker compose config
      --pull-missing-images   Attempt to docker pull dependency images that are missing locally
      --skip-dependency-save  Skip saving dependency images (only save the built backend image)
      --export-volumes        Export named docker volumes referenced by docker-compose into tar.gz files
      --volume-exporter-image IMAGE
                               Image to use while exporting volumes (default: alpine:3.20)
  -h, --help                  Show this help message and exit

Environment overrides:
  LIGHTEN=0/1, NEED_MIRROR=0/1 are forwarded to docker build as build args.
USAGE
}

if [[ ${1-} == "-h" || ${1-} == "--help" ]]; then
  usage
  exit 0
fi

OUTPUT_DIR="dist/offline-bundle"
IMAGE_TAG=""
COMPOSE_PROFILES_OVERRIDE=""
PULL_MISSING=0
SAVE_DEP_IMAGES=1
EXPORT_VOLUMES=0
EXPORTER_IMAGE="alpine:3.20"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -t|--tag)
      [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; exit 1; }
      IMAGE_TAG="$2"
      shift 2
      ;;
    --profiles)
      [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; exit 1; }
      COMPOSE_PROFILES_OVERRIDE="$2"
      shift 2
      ;;
    --pull-missing-images)
      PULL_MISSING=1
      shift
      ;;
    --skip-dependency-save)
      SAVE_DEP_IMAGES=0
      shift
      ;;
    --export-volumes)
      EXPORT_VOLUMES=1
      shift
      ;;
    --volume-exporter-image)
      [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; exit 1; }
      EXPORTER_IMAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker CLI is required but not found on PATH" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found on PATH" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is required" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -z "$IMAGE_TAG" ]]; then
  IMAGE_TAG="ragflow-offline:$(date +%Y%m%d%H%M)"
fi

LIGHTEN_VAL=${LIGHTEN:-0}
NEED_MIRROR_VAL=${NEED_MIRROR:-0}

OUTPUT_DIR_ABS=$(python3 - <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
"$OUTPUT_DIR")

WORKTREE_DIR="$OUTPUT_DIR_ABS/bundle"
IMAGES_DIR="$OUTPUT_DIR_ABS/images"
METADATA_DIR="$OUTPUT_DIR_ABS/metadata"
VOLUME_DIR="$OUTPUT_DIR_ABS/volumes"

mkdir -p "$OUTPUT_DIR_ABS" "$IMAGES_DIR" "$METADATA_DIR"

if [[ -d "$WORKTREE_DIR" ]]; then
  rm -rf "$WORKTREE_DIR"
fi
mkdir -p "$WORKTREE_DIR"

# -----------------------------------------------------------------------------
# Build the backend image
# -----------------------------------------------------------------------------
echo "[1/6] Building backend image $IMAGE_TAG"

docker build \
  --tag "$IMAGE_TAG" \
  --build-arg "LIGHTEN=$LIGHTEN_VAL" \
  --build-arg "NEED_MIRROR=$NEED_MIRROR_VAL" \
  -f "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT"

SAFE_TAG=$(echo "$IMAGE_TAG" | sed 's#[^A-Za-z0-9_.-]#_#g')
TARGET_IMAGE_ARCHIVE="$IMAGES_DIR/${SAFE_TAG}.tar.gz"

echo "[2/6] Saving backend image to $TARGET_IMAGE_ARCHIVE"
docker save "$IMAGE_TAG" | gzip > "$TARGET_IMAGE_ARCHIVE"

echo "[3/6] Rendering docker-compose configuration"
COMPOSE_TMP=$(mktemp)

if [[ -n "$COMPOSE_PROFILES_OVERRIDE" ]]; then
  COMPOSE_ENV=("env" "COMPOSE_PROFILES=$COMPOSE_PROFILES_OVERRIDE" "RAGFLOW_IMAGE=$IMAGE_TAG")
else
  COMPOSE_ENV=("env" "RAGFLOW_IMAGE=$IMAGE_TAG")
fi
(
  cd "$REPO_ROOT/docker"
  "${COMPOSE_ENV[@]}" docker compose -f docker-compose.yml config --format json
) > "$COMPOSE_TMP"

python3 - <<'PY' "$REPO_ROOT" "$COMPOSE_TMP" "$METADATA_DIR"
import json
import pathlib
from collections import defaultdict
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
compose_path = pathlib.Path(sys.argv[2])
metadata_dir = pathlib.Path(sys.argv[3])
metadata_dir.mkdir(parents=True, exist_ok=True)

compose_cfg = json.loads(compose_path.read_text())
images = set()
bind_info = defaultdict(lambda: {"services": set(), "targets": set(), "source_raw": None})
volume_info = defaultdict(lambda: {"services": set()})

services = compose_cfg.get("services", {})
for svc_name, svc in services.items():
    image = svc.get("image")
    if image:
        images.add(image)
    for vol in svc.get("volumes", []):
        vtype = vol.get("type")
        if vtype == "bind":
            source_raw = vol.get("source")
            if not source_raw:
                continue
            source_path = pathlib.Path(source_raw)
            if source_path.is_absolute():
                abs_path = source_path.resolve()
            else:
                abs_path = (repo_root / source_path).resolve()
            key = str(abs_path)
            entry = bind_info[key]
            entry["services"].add(svc_name)
            target = vol.get("target")
            if target:
                entry["targets"].add(target)
            if entry["source_raw"] is None:
                entry["source_raw"] = source_raw
        elif vtype == "volume":
            logical = vol.get("source")
            if logical:
                volume_info[logical]["services"].add(svc_name)

manifest = {
    "images": sorted(images),
    "host_mounts": [],
    "named_volumes": [],
}

for abs_path_str in sorted(bind_info.keys()):
    data = bind_info[abs_path_str]
    abs_path = pathlib.Path(abs_path_str)
    try:
        relative = str(abs_path.relative_to(repo_root))
    except ValueError:
        relative = None
    manifest["host_mounts"].append({
        "source": abs_path_str,
        "source_raw": data["source_raw"],
        "relative": relative,
        "services": sorted(data["services"]),
        "targets": sorted(data["targets"]),
        "exists": abs_path.exists(),
        "is_dir": abs_path.is_dir(),
        "is_file": abs_path.is_file(),
    })

volumes_section = compose_cfg.get("volumes", {})
for logical_name in sorted(volume_info.keys()):
    resolved_name = volumes_section.get(logical_name, {}).get("name", logical_name)
    manifest["named_volumes"].append({
        "source": logical_name,
        "resolved_name": resolved_name,
        "services": sorted(volume_info[logical_name]["services"]),
    })

manifest_path = metadata_dir / "compose_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

images_path = metadata_dir / "images.txt"
images_path.write_text("\n".join(manifest["images"]) + ("\n" if manifest["images"] else ""), encoding="utf-8")

binds_path = metadata_dir / "bind_mounts.tsv"
with binds_path.open("w", encoding="utf-8") as fh:
    fh.write("source\tsource_raw\trelative\ttype\texists\tservices\ttargets\n")
    for mount in manifest["host_mounts"]:
        if mount["is_dir"]:
            mtype = "dir"
        elif mount["is_file"]:
            mtype = "file"
        else:
            mtype = "missing"
        fh.write(
            f"{mount['source']}\t{mount['source_raw'] or ''}\t{mount['relative'] or ''}\t{mtype}\t{int(mount['exists'])}\t"
            f"{','.join(mount['services'])}\t{','.join(mount['targets'])}\n"
        )

volumes_path = metadata_dir / "named_volumes.tsv"
with volumes_path.open("w", encoding="utf-8") as fh:
    fh.write("source\tresolved_name\tservices\n")
    for item in manifest["named_volumes"]:
        fh.write(f"{item['source']}\t{item['resolved_name']}\t{','.join(item['services'])}\n")
PY

rm -f "$COMPOSE_TMP"

# -----------------------------------------------------------------------------
# Copy required host files into bundle
# -----------------------------------------------------------------------------
echo "[4/6] Copying docker configuration and host mounts"
cp -a "$REPO_ROOT/docker" "$WORKTREE_DIR/"

if [[ -d "$REPO_ROOT/history_data_agent" ]]; then
  cp -a "$REPO_ROOT/history_data_agent" "$WORKTREE_DIR/"
else
  mkdir -p "$WORKTREE_DIR/history_data_agent"
fi

ENV_COPY="$WORKTREE_DIR/docker/.env"
if [[ -f "$ENV_COPY" ]]; then
  python3 - <<'PY' "$ENV_COPY" "$IMAGE_TAG"
import sys, re
env_path = sys.argv[1]
image_tag = sys.argv[2]
text = open(env_path, encoding='utf-8').read()
pattern = re.compile(r'^RAGFLOW_IMAGE=.*$', re.MULTILINE)
replacement = f'RAGFLOW_IMAGE={image_tag}'
if pattern.search(text):
    text = pattern.sub(replacement, text, count=1)
else:
    if not text.endswith('\n'):
        text += '\n'
    text += replacement + '\n'
open(env_path, 'w', encoding='utf-8').write(text)
PY
fi

# -----------------------------------------------------------------------------
# Persist metadata about the built image
# -----------------------------------------------------------------------------
docker image inspect "$IMAGE_TAG" > "$METADATA_DIR/${SAFE_TAG}_inspect.json"

# -----------------------------------------------------------------------------
# Save dependency images (excluding the freshly built backend image)
# -----------------------------------------------------------------------------
readarray -t ALL_IMAGES < "$METADATA_DIR/images.txt"

DEPENDENCY_IMAGES=()
for image in "${ALL_IMAGES[@]}"; do
  [[ -z "$image" ]] && continue
  if [[ "$image" != "$IMAGE_TAG" ]]; then
    DEPENDENCY_IMAGES+=("$image")
  fi
done

EXPORTED_IMAGES=()
MISSING_IMAGES=()

if [[ $SAVE_DEP_IMAGES -eq 1 && ${#DEPENDENCY_IMAGES[@]} -gt 0 ]]; then
  echo "[5/6] Saving dependency images"
  for image in "${DEPENDENCY_IMAGES[@]}"; do
    if docker image inspect "$image" >/dev/null 2>&1; then
      EXPORTED_IMAGES+=("$image")
      continue
    fi
    if [[ $PULL_MISSING -eq 1 ]]; then
      if docker pull "$image" >/dev/null; then
        EXPORTED_IMAGES+=("$image")
      else
        MISSING_IMAGES+=("$image")
      fi
    else
      MISSING_IMAGES+=("$image")
    fi
  done

  if [[ ${#EXPORTED_IMAGES[@]} -gt 0 ]]; then
    DEP_ARCHIVE="$IMAGES_DIR/dependency-images.tar.gz"
    docker save "${EXPORTED_IMAGES[@]}" | gzip > "$DEP_ARCHIVE"
  fi
else
  echo "[5/6] Skipping dependency image export"
fi

if [[ ${#MISSING_IMAGES[@]} -gt 0 ]]; then
  printf '%s\n' "${MISSING_IMAGES[@]}" > "$METADATA_DIR/missing_images.txt"
fi

# -----------------------------------------------------------------------------
# Export named docker volumes if requested
# -----------------------------------------------------------------------------
if [[ $EXPORT_VOLUMES -eq 1 ]]; then
  echo "[6/6] Exporting named volumes"
  mkdir -p "$VOLUME_DIR"
  if ! docker image inspect "$EXPORTER_IMAGE" >/dev/null 2>&1; then
    if [[ $PULL_MISSING -eq 1 ]]; then
      docker pull "$EXPORTER_IMAGE"
    else
      echo "Exporter image $EXPORTER_IMAGE not present locally; rerun with --pull-missing-images or pull manually" >&2
      exit 1
    fi
  fi

  while IFS=$'\t' read -r source resolved services; do
    [[ "$source" == "source" ]] && continue
    if docker volume inspect "$resolved" >/dev/null 2>&1; then
      archive="$VOLUME_DIR/${source}.tar.gz"
      docker run --rm -v "$resolved:/volume" -v "$VOLUME_DIR:/backup" "$EXPORTER_IMAGE" \
        sh -c "cd /volume && tar -czf /backup/${source}.tar.gz ."
    else
      echo "$resolved" >> "$METADATA_DIR/missing_volumes.txt"
    fi
  done < "$METADATA_DIR/named_volumes.tsv"
else
  echo "[6/6] Volume export skipped (use --export-volumes to enable)"
fi

echo "Bundle generated at $OUTPUT_DIR_ABS"
