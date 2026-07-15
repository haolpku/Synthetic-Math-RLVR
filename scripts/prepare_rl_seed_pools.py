#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def dapo_problem(prompt: object) -> str:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("unexpected DAPO prompt format")
    text = str(prompt[0].get("content", "")).strip()
    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    suffix = 'Remember to put your answer on its own line after "Answer:".'
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    if text.endswith(suffix):
        text = text[:-len(suffix)].strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RLVR-standard math prompt pools and prepare synthetic-generation seeds.")
    parser.add_argument("--dapo-seeds", type=int, default=2000)
    parser.add_argument("--deepscaler-seeds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-dir", default=str(ROOT / "data/rl_seed_pools"))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("Run this script in the dataflow-agent Conda environment, which contains datasets.") from exc

    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    # The current hosted train split repeats the underlying pool roughly 100x
    # for DAPO's training schedule. Recover the unique prompt pool by source ID.
    dapo_repeated = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    dapo_by_id: dict[str, dict] = {}
    for index, row in enumerate(dapo_repeated):
        source_id = str((row.get("extra_info") or {}).get("index", index))
        dapo_by_id.setdefault(source_id, row)
    dapo = list(dapo_by_id.values())
    dapo_rows = []
    for index, row in enumerate(dapo):
        source_id = str((row.get("extra_info") or {}).get("index", index))
        dapo_rows.append({
            "seed_id": stable_id("dapo_math_17k", source_id, prefix="seed"),
            "source": "BytedTsinghua-SIA/DAPO-Math-17k",
            "source_id": source_id,
            "domain": str(row.get("ability", "MATH")),
            "difficulty": "rlvr_curated",
            "problem": dapo_problem(row["prompt"]),
            "answer": str((row.get("reward_model") or {}).get("ground_truth", "")),
            "original_rlvr_record": row,
        })
    write_jsonl(output_dir / "dapo_math_17k_full.jsonl", dapo_rows)
    rng.shuffle(dapo_rows)
    write_jsonl(output_dir / "dapo_math_2k_seeds.jsonl", dapo_rows[: args.dapo_seeds])

    deepscaler = list(load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train"))
    deepscaler_rows = [{
        "seed_id": stable_id("deepscaler_preview", index, row["problem"], prefix="seed"),
        "source": "agentica-org/DeepScaleR-Preview-Dataset",
        "source_id": str(index),
        "domain": "MATH",
        "difficulty": "rlvr_curated",
        "problem": row["problem"],
        "answer": str(row["answer"]),
        "solution": row.get("solution"),
    } for index, row in enumerate(deepscaler)]
    rng.shuffle(deepscaler_rows)
    write_jsonl(output_dir / "deepscaler_2k_seeds.jsonl", deepscaler_rows[: args.deepscaler_seeds])

    print(f"DAPO hosted rows (with schedule repeats): {len(dapo_repeated)}")
    print(f"DAPO unique source IDs: {len(dapo)}")
    print(f"DAPO generation seeds: {min(args.dapo_seeds, len(dapo))}")
    print(f"DeepScaleR full available: {len(deepscaler)}")
    print(f"DeepScaleR replication seeds: {min(args.deepscaler_seeds, len(deepscaler))}")


if __name__ == "__main__":
    main()
