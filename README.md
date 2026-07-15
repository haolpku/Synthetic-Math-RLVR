# The Synthetic Substitution Gap in Math RLVR: Quality, Learnability, and Training Signal

## Current idea

This project measures how replacing curated real math prompts with naturally generated question-answer pairs changes RLVR. It uses a paired design: every trusted NuminaMath question has one generated counterpart, and the training set keeps the same parent seeds while the synthetic replacement rate varies from 0% to 100%.

The analysis follows the chain `synthetic quality -> prompt learnability -> usable within-group reward signal -> reasoning outcome`. Replacement curves quantify the gap; solver consistency, later base-model K=8 rollouts, and learnability-matched controls diagnose it.

The project no longer uses question validity as a training treatment. Human annotation is a sampled correctness audit only. It estimates question validity, unique-answer existence, reference-answer correctness, and the effectiveness of automatic solver-consistency filtering.

## Experimental design

- Source pool: trusted-source subset of `nlile/NuminaMath-1.5-RL-Verifiable`.
- Data: 5,000 GPT-5.4 pairs plus a later cross-generator replication on shared held-out seeds.
- RLVR arms: 0%, 25%, 50%, 75%, and 100% synthetic replacement, plus quality-filtered synthetic with real fallback.
- Main model: `Qwen/Qwen2.5-1.5B` base.
- Replication model: `Qwen/Qwen2.5-7B` base.
- SFT control: real, raw synthetic, and filtered-fallback only.
- Human audit: 800 synthetic pairs, two independent reviewers plus selective expert adjudication.
- Evaluation: the local Qwen2.5-Math evaluation harness.

The decision-complete design is in [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md), and the paper narrative is in [docs/PAPER_STORY.md](docs/PAPER_STORY.md).

## Data pipeline status

The GPT-5.4 paired-data run is complete as of July 15, 2026:

- 5,000/5,000 real-synthetic canonical pairs;
- 5,000/5,000 samples with three independent GPT-5.4-mini solver rollouts;
- 2,838/5,000 pairs accepted by the automatic quality policy;
- six 4,000-row RLVR views (R0/R25/R50/R75/R100 and filtered real fallback);
- three 4,000-row SFT views (real, raw synthetic, and filtered real fallback).

The fallback training view retains 2,272 synthetic questions (56.8%). A cross-generator replication with Claude Sonnet 4.6 is resumable and currently contains 19/500 held-out pairs. The complete data snapshot is hosted at [lhpku20010120/Synthetic-Math-RLVR](https://huggingface.co/datasets/lhpku20010120/Synthetic-Math-RLVR).

The resumable main run is:

```bash
zsh scripts/run_full_data_pipeline.sh
```

Principal artifacts are written to:

- `data/canonical/numina_5k_seeds.jsonl`
- `data/raw/gpt_pairs.jsonl`
- `data/interim/gpt_pairs_quality.jsonl`
- `data/processed/rlvr/*.parquet`
- `data/processed/sft/*.jsonl`
- `data/manifest.json`

The generation configuration uses `gpt-5.4` with low reasoning effort. Automatic solver consistency uses three independent `gpt-5.4-mini` rollouts and requires at least two reference matches. Base-model K=8 learnability profiling remains a GPU-side step and must not be conflated with this production quality filter.

API credentials must be supplied through runtime environment variables and must not be committed to the repository.
