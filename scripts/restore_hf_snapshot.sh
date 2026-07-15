#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

REPO_ID="lhpku20010120/Synthetic-Math-RLVR"
SNAPSHOT_DIR="${1:-${TMPDIR:-/tmp}/synthetic_math_rlvr_hf_snapshot}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$SNAPSHOT_DIR"

(
  cd "$SNAPSHOT_DIR"
  shasum -a 256 -c SHA256SUMS
)

mkdir -p data/canonical data/raw data/interim data/processed/rlvr data/processed/sft

cp "$SNAPSHOT_DIR/canonical/numina_5k_seeds.jsonl" data/canonical/
cp "$SNAPSHOT_DIR/pairs/gpt_pairs.jsonl" data/raw/gpt_pairs.jsonl
cp "$SNAPSHOT_DIR/pairs/gpt_pairs_solved.jsonl" data/interim/gpt_pairs_solved.jsonl
cp "$SNAPSHOT_DIR/pairs/gpt_pairs_quality.jsonl" data/interim/gpt_pairs_quality.jsonl
cp "$SNAPSHOT_DIR/pairs/claude_heldout_pairs.partial_19.jsonl" data/raw/claude_heldout_pairs.jsonl
cp "$SNAPSHOT_DIR/processed/rlvr/"* data/processed/rlvr/
cp "$SNAPSHOT_DIR/processed/sft/"* data/processed/sft/
cp "$SNAPSHOT_DIR/manifests/numina_5k_manifest.json" data/canonical/
cp "$SNAPSHOT_DIR/manifests/gpt_pairs_quality_manifest.json" data/interim/
cp "$SNAPSHOT_DIR/manifests/replacement_views_manifest.json" data/processed/manifest.json
cp "$SNAPSHOT_DIR/manifests/release_manifest.json" data/manifest.json

echo "Snapshot restored. Claude replication can resume from 19/500 with:"
echo "  zsh scripts/run_claude_replication.sh"
