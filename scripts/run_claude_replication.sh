#!/bin/zsh
set -euo pipefail

: "${CLAUDE_API_KEY:?Set CLAUDE_API_KEY in the runtime environment}"
: "${GPT_API_KEY:?Set GPT_API_KEY in the runtime environment}"

ROOT="${0:A:h:h}"
cd "$ROOT"

mkdir -p data/logs data/raw data/interim

count_rows() {
  if [[ -f "$1" ]]; then
    wc -l < "$1" | tr -d ' '
  else
    echo 0
  fi
}

for round in {1..8}; do
  current=$(count_rows data/raw/claude_heldout_pairs.jsonl)
  [[ "$current" -eq 500 ]] && break
  echo "claude generation round=$round current=$current target=500"
  OPENAI_API_KEY="$CLAUDE_API_KEY" python scripts/generate_pool.py \
    --model claude-sonnet-4-6 \
    --reasoning-effort low \
    --workers 16 \
    --split held_out_audit \
    --output data/raw/claude_heldout_pairs.jsonl \
    --failed-output data/raw/claude_heldout_pairs_failures.jsonl
done

generated=$(count_rows data/raw/claude_heldout_pairs.jsonl)
if [[ "$generated" -ne 500 ]]; then
  echo "claude generation incomplete: $generated/500" >&2
  exit 2
fi

for round in {1..8}; do
  current=$(count_rows data/interim/claude_heldout_pairs_solved.jsonl)
  [[ "$current" -eq 500 ]] && break
  echo "claude solver round=$round current=$current target=500"
  OPENAI_API_KEY="$GPT_API_KEY" python scripts/solve_pairs.py \
    --input data/raw/claude_heldout_pairs.jsonl \
    --output data/interim/claude_heldout_pairs_solved.jsonl \
    --failed-output data/interim/claude_heldout_solver_failures.jsonl \
    --model gpt-5.4-mini \
    --reasoning-effort low \
    --rollouts 3 \
    --workers 50
done

solved=$(count_rows data/interim/claude_heldout_pairs_solved.jsonl)
if [[ "$solved" -ne 500 ]]; then
  echo "claude solver incomplete: $solved/500" >&2
  exit 3
fi

python scripts/assess_pair_quality.py \
  --input data/interim/claude_heldout_pairs_solved.jsonl \
  --output data/interim/claude_heldout_pairs_quality.jsonl \
  --manifest data/interim/claude_heldout_pairs_quality_manifest.json

echo "claude replication pipeline complete"
