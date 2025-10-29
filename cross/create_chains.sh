#!/usr/bin/env bash
# script adapted from mcstitzer
set -euo pipefail

usage() {
  echo "Usage: $0 -i <MAF_DIR> -o <OUT_DIR> [-j JOBS]"
  echo
  echo "  -i, --input   Directory containing .maf or .maf.gz files"
  echo "  -o, --output  Directory to save .chain files"
  echo "  -j, --jobs    Number of parallel jobs (default: 8)"
  exit 1
}

# default values
JOBS=8
MAF_DIR=""
OUT_DIR=""

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)  MAF_DIR="$2"; shift 2 ;;
    -o|--output) OUT_DIR="$2"; shift 2 ;;
    -j|--jobs)   JOBS="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

[[ -z "$MAF_DIR" || -z "$OUT_DIR" ]] && usage
[[ ! -d "$MAF_DIR" ]] && { echo "Error: input directory not found: $MAF_DIR"; exit 1; }

mkdir -p "$OUT_DIR"

# fetch maf-convert if missing
if [[ ! -x ./maf-convert ]]; then
  echo "Downloading maf-convert..."
  wget -q https://gitlab.com/mcfrith/last/-/raw/main/bin/maf-convert -O maf-convert
  chmod 755 maf-convert
fi

# convert in parallel
if command -v parallel >/dev/null 2>&1; then
  find "$MAF_DIR" -type f \( -name "*.maf" -o -name "*.maf.gz" \) -print0 | \
  parallel -0 -j "$JOBS" --halt soon,fail=1 --no-run-if-empty '
    f={};
    base=$(basename "$f")
    base_no_ext=${base%.gz}
    base_no_ext=${base_no_ext%.maf}
    out="'$OUT_DIR'/${base_no_ext}.chain"
    if [[ "$f" == *.gz ]]; then
      zcat "$f" | python maf-convert chain - > "$out"
    else
      python maf-convert chain "$f" > "$out"
    fi
  '
else
  echo "GNU parallel not found; running sequentially"
  for f in "$MAF_DIR"/*.{maf,maf.gz}; do
    [[ -e "$f" ]] || continue
    base=$(basename "$f")
    base_no_ext=${base%.gz}
    base_no_ext=${base_no_ext%.maf}
    out="$OUT_DIR/${base_no_ext}.chain"
    if [[ "$f" == *.gz ]]; then
      zcat "$f" | python maf-convert chain - > "$out"
    else
      python maf-convert chain "$f" > "$out"
    fi
  done
fi

echo "✅ Done. Chains written to: $OUT_DIR"
