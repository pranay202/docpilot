#!/usr/bin/env bash
# Fetch the LibreOffice help guides (MPL-2.0) for Writer, Calc, Impress and Draw at a pinned
# commit into brand_docs/libreoffice/<product>. The eval answers were written against this revision.
set -euo pipefail

SHA="${LIBREOFFICE_HELP_SHA:-72e318cf7bc122eaf45cf9216c12b4d509b5f7e1}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/brand_docs/libreoffice"
declare -A PRODUCTS=([writer]=swriter [calc]=scalc [impress]=simpress [draw]=sdraw)

if [[ -f "$DEST/.sha" && "$(cat "$DEST/.sha")" == "$SHA" ]]; then
  echo "LibreOffice help already at $SHA"; exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" remote add origin https://github.com/LibreOffice/help.git
git -C "$tmp" sparse-checkout set $(for m in "${PRODUCTS[@]}"; do printf 'source/text/%s/guide ' "$m"; done)
git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$SHA"
git -C "$tmp" checkout -q FETCH_HEAD

rm -rf "$DEST" && mkdir -p "$DEST"
for product in "${!PRODUCTS[@]}"; do
  cp -r "$tmp/source/text/${PRODUCTS[$product]}/guide" "$DEST/$product"
done
echo "$SHA" > "$DEST/.sha"
echo "Fetched LibreOffice help ($SHA): ${!PRODUCTS[*]}"
