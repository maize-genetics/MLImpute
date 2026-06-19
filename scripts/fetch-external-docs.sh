#!/usr/bin/env bash
#
# Fetch documentation from the accompanying seq_sim repository and stage it
# under docs/_external/ so the MkDocs site can ingest it at build time.
#
# This directory is regenerated on every run and is git-ignored; never edit the
# downloaded files by hand.
#
# Usage:
#   bash scripts/fetch-external-docs.sh
#
# Override the source ref (branch, tag, or commit SHA):
#   SEQ_SIM_REF=v1.2.3 bash scripts/fetch-external-docs.sh

set -euo pipefail

# --- Configuration ----------------------------------------------------------
REPO="${SEQ_SIM_REPO:-maize-genetics/seq_sim}"
REF="${SEQ_SIM_REF:-main}"
DEST="${SEQ_SIM_DEST:-docs/_external/seq_sim}"
RAW_BASE="https://raw.githubusercontent.com/${REPO}/${REF}"
BLOB_BASE="https://github.com/${REPO}/blob/${REF}"

# Source path in seq_sim -> destination path (relative to $DEST), separated by
# a space. Markdown files (*.md) get an attribution banner and link rewriting;
# everything else is copied verbatim.
FILES=(
  "README.md README.md"
  "docs/commands.md commands.md"
  "docs/images/grits_v2_seq_sim_pipeline.svg images/grits_v2_seq_sim_pipeline.svg"
)

# --- Helpers ----------------------------------------------------------------
# Resolve the repo root so the script works regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() { printf '[fetch-external-docs] %s\n' "$1"; }

# Rewrite repo-relative markdown links to absolute upstream GitHub URLs so they
# resolve once the content is embedded in our site. Image links pointing at the
# locally-staged images/ folder, anchors, and absolute URLs are left untouched.
# $1 = source dir of the file within seq_sim (e.g. "" or "docs"), $2 = dest file
rewrite_links() {
  local src_dir="$1" dest_file="$2"
  local tmp
  tmp="$(mktemp)"
  awk -v base="${BLOB_BASE}" -v dir="${src_dir}" '
    function normalize(p,   parts, n, i, out, j, res) {
      n = split(p, parts, "/"); j = 0
      for (i = 1; i <= n; i++) {
        if (parts[i] == "" || parts[i] == ".") continue
        if (parts[i] == "..") { if (j > 0) j--; continue }
        out[++j] = parts[i]
      }
      res = ""
      for (i = 1; i <= j; i++) res = res (i > 1 ? "/" : "") out[i]
      return res
    }
    {
      line = $0; out = ""
      while (match(line, /\]\([^)]*\)/)) {
        # Capture outer match bounds before any nested match() clobbers them.
        ms = RSTART; ml = RLENGTH
        pre = substr(line, 1, ms - 1)
        tok = substr(line, ms, ml)
        rest = substr(line, ms + ml)
        target = substr(tok, 3, length(tok) - 3)
        if (target ~ /^(https?:|\/\/|#|mailto:)/ || target ~ /^\.?\/?images\//) {
          newtok = tok
        } else {
          anchor = ""
          h = index(target, "#")
          if (h > 0) {
            anchor = substr(target, h)
            target = substr(target, 1, h - 1)
          }
          combined = (dir == "" ? target : dir "/" target)
          newtok = "](" base "/" normalize(combined) anchor ")"
        }
        out = out pre newtok
        line = rest
      }
      print out line
    }
  ' "${dest_file}" > "${tmp}"
  mv "${tmp}" "${dest_file}"
}

# Prepend a Material admonition banner attributing the source of a markdown file.
# $1 = source path in seq_sim, $2 = absolute destination file
write_banner() {
  local src_path="$1" dest_file="$2"
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}" <<EOF
<!-- AUTO-GENERATED: do not edit. Synced from ${REPO} by scripts/fetch-external-docs.sh -->

!!! info "Imported from seq_sim"
    This page is automatically synced from
    [\`${REPO}/${src_path}\`](${BLOB_BASE}/${src_path}) at ref \`${REF}\`.
    Do not edit it here; changes are overwritten on every docs build.

EOF
  cat "${dest_file}" >> "${tmp}"
  mv "${tmp}" "${dest_file}"
}

# --- Fetch ------------------------------------------------------------------
log "Source repo : ${REPO}@${REF}"
log "Destination : ${DEST}"

rm -rf "${DEST}"
mkdir -p "${DEST}"

for entry in "${FILES[@]}"; do
  # shellcheck disable=SC2086
  set -- ${entry}
  src_path="$1"
  dest_rel="$2"
  dest_file="${DEST}/${dest_rel}"

  mkdir -p "$(dirname "${dest_file}")"
  log "GET ${src_path}"
  curl -fsSL "${RAW_BASE}/${src_path}" -o "${dest_file}"

  case "${dest_rel}" in
    *.md)
      # Rewrite image references that point at the seq_sim docs/images/ folder
      # so they resolve against the images/ folder we stage alongside the page.
      sed -i.bak -E 's#(\(|")(\./)?(docs/)?images/#\1images/#g' "${dest_file}"
      rm -f "${dest_file}.bak"
      # Point remaining repo-relative links at the upstream repo.
      rewrite_links "$(dirname "${src_path}")" "${dest_file}"
      write_banner "${src_path}" "${dest_file}"
      ;;
  esac
done

log "Done. Fetched ${#FILES[@]} file(s) into ${DEST}/"
