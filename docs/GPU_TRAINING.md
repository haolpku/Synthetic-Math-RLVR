# GPU handoff

## Data preparation

Set keys only in the runtime environment. The supplied service is OpenAI-compatible.

On the current Mac, the working local environment is `conda activate dataflow-agent`.

```bash
export OPENAI_BASE_URL=http://HOST:PORT/v1
export OPENAI_API_KEY=...

python scripts/prepare_rl_seed_pools.py
python scripts/generate_pool.py --model GENERATOR_MODEL
python scripts/auto_verify.py --model VERIFIER_MODEL
python scripts/estimate_reachability.py --model BASE_POLICY_MODEL
```

After expert audit and merge:

```bash
python scripts/build_datasets.py --input data/interim/reachability_with_human.jsonl
python scripts/validate_data.py data/processed/sft.jsonl --required sample_id group messages
python scripts/export_parquet.py --input data/processed/rlvr.jsonl --output data/processed/rlvr.parquet
```

The equivalent DataFlow-native end-to-end entry is `pipeline/run_pipeline.py`. Run it inside the DataFlow environment (or install the sibling DataFlow package), set `DATAFLOW_HOME` if that repository is elsewhere, and provide the key through `DF_API_KEY`.

```bash
export DF_API_KEY=...
export RIVQ_API_URL=http://HOST:PORT/v1/chat/completions
python pipeline/run_pipeline.py \
  --generator-model GENERATOR_MODEL \
  --verifier-model VERIFIER_MODEL \
  --solver-model BASE_POLICY_MODEL
```

## SFT

```bash
torchrun --nproc_per_node=8 training/train_sft.py \
  --model MODEL_PATH \
  --train-file data/processed/sft.jsonl \
  --output-dir outputs/sft
```

Pin the TRL/Transformers versions used on the GPU machine before the full sweep; their trainer APIs change frequently.

## RLVR with verl

Use `data/processed/rlvr.parquet` as the training file and `training/reward_fn.py` as the custom reward function. Configure the reward manager to call `compute_score`. Use `training/gated_reward_fn.py` only for the mitigation arm.

Required rollout logging per prompt: sample ID, checkpoint step, K, hit count, group reward mean/variance, response length, and KL. Without per-question logging, the dead-to-active transition cannot be tested.

Run a 100-sample smoke test first. Confirm that valid reference answers receive reward 1 and known mismatches receive 0 before allocating the full rollout budget.
