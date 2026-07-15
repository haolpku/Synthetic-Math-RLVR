#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

mkdir -p data/logs data/raw data/interim data/processed

count_rows() {
  if [[ -f "$1" ]]; then
    wc -l < "$1" | tr -d ' '
  else
    echo 0
  fi
}

for round in {1..8}; do
  current=$(count_rows data/raw/gpt_pairs.jsonl)
  [[ "$current" -eq 5000 ]] && break
  echo "generation round=$round current=$current target=5000"
  python scripts/generate_pool.py \
    --model gpt-5.4 \
    --reasoning-effort low \
    --workers 50 \
    --output data/raw/gpt_pairs.jsonl \
    --failed-output data/raw/gpt_pairs_failures.jsonl
done

generated=$(count_rows data/raw/gpt_pairs.jsonl)
if [[ "$generated" -ne 5000 ]]; then
  echo "generation incomplete: $generated/5000" >&2
  exit 2
fi

for round in {1..8}; do
  current=$(count_rows data/interim/gpt_pairs_solved.jsonl)
  [[ "$current" -eq 5000 ]] && break
  echo "solver round=$round current=$current target=5000"
  python scripts/solve_pairs.py \
    --input data/raw/gpt_pairs.jsonl \
    --output data/interim/gpt_pairs_solved.jsonl \
    --failed-output data/interim/gpt_pairs_solver_failures.jsonl \
    --model gpt-5.4-mini \
    --reasoning-effort low \
    --rollouts 3 \
    --workers 50
done

solved=$(count_rows data/interim/gpt_pairs_solved.jsonl)
if [[ "$solved" -ne 5000 ]]; then
  echo "solver incomplete: $solved/5000" >&2
  exit 3
fi

python scripts/assess_pair_quality.py \
  --input data/interim/gpt_pairs_solved.jsonl \
  --output data/interim/gpt_pairs_quality.jsonl \
  --manifest data/interim/gpt_pairs_quality_manifest.json

python scripts/build_replacement_views.py \
  --input data/interim/gpt_pairs_quality.jsonl \
  --output-dir data/processed \
  --split train \
  --expected-count 4000

python scripts/make_release_manifest.py
echo "pipeline complete"
