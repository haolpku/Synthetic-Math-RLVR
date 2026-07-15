#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import read_jsonl, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def rank_key(parent_seed_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{parent_seed_id}".encode()).hexdigest()


def nested_synthetic_sets(rows: list[dict], seed: int) -> dict[float, set[str]]:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("source")), str(row.get("problem_type")))].append(row)
    balanced_order: list[tuple[float, str, str]] = []
    for items in strata.values():
        ordered = sorted(items, key=lambda row: rank_key(row["parent_seed_id"], seed))
        size = len(ordered)
        for index, row in enumerate(ordered):
            # Sorting by the within-stratum quantile interleaves all strata,
            # giving exact global ratios while approximately preserving every
            # source/problem-type proportion. Prefixes make ratios nested.
            quantile = (index + 0.5) / size
            balanced_order.append((quantile, rank_key(row["parent_seed_id"], seed), row["parent_seed_id"]))
    balanced_order.sort()
    return {
        ratio: {item[2] for item in balanced_order[: round(len(rows) * ratio)]}
        for ratio in RATIOS
    }


def rlvr_record(row: dict, use_synthetic: bool, arm: str, nominal_ratio: float) -> dict:
    origin = "synthetic" if use_synthetic else "real"
    problem = row["synthetic_problem"] if use_synthetic else row["real_problem"]
    answer = row["synthetic_answer"] if use_synthetic else row["real_answer"]
    return {
        "data_source": "numina_synthetic_substitution",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "math",
        "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": str(answer)},
        "extra_info": {
            "sample_id": stable_id(row["parent_seed_id"], arm, prefix="train"),
            "parent_seed_id": row["parent_seed_id"],
            "source": row.get("source"),
            "problem_type": row.get("problem_type"),
            "origin": origin,
            "arm": arm,
            "synthetic_ratio": nominal_ratio,
            "generator_model": row.get("generator_model") if use_synthetic else None,
            "prompt_version": row.get("prompt_version") if use_synthetic else None,
            "quality_accept": row.get("quality_accept") if use_synthetic else None,
            "solver_matches": row.get("solver_matches") if use_synthetic else None,
        },
    }


def sft_record(row: dict, use_synthetic: bool, arm: str) -> dict:
    problem = row["synthetic_problem"] if use_synthetic else row["real_problem"]
    solution = row["synthetic_solution"] if use_synthetic else row["real_solution"]
    answer = row["synthetic_answer"] if use_synthetic else row["real_answer"]
    return {
        "sample_id": stable_id(row["parent_seed_id"], arm, prefix="sft"),
        "parent_seed_id": row["parent_seed_id"],
        "origin": "synthetic" if use_synthetic else "real",
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"{solution}\n\n\\boxed{{{answer}}}"},
        ],
    }


def write_view(path: Path, rows: list[dict]) -> None:
    write_jsonl(path.with_suffix(".jsonl"), rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested real-to-synthetic RLVR and SFT training views.")
    parser.add_argument("--input", default=str(ROOT / "data/interim/gpt_pairs_quality.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/processed"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--expected-count", type=int, default=4000, help="Use 0 for smoke subsets.")
    parser.add_argument("--allow-missing-quality", action="store_true", help="Only for raw smoke export; fallback remains real.")
    args = parser.parse_args()

    source = read_jsonl(args.input)
    rows = source if args.split == "all" else [row for row in source if row.get("split") == args.split]
    ids = [row["parent_seed_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate parent_seed_id in selected split")
    if args.expected_count and len(rows) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} rows for {args.split}, found {len(rows)}")
    if not args.allow_missing_quality and any("quality_accept" not in row for row in rows):
        raise SystemExit("quality fields missing; run solve_pairs.py and assess_pair_quality.py first")

    output = Path(args.output_dir)
    nested = nested_synthetic_sets(rows, args.seed)
    manifest: dict[str, object] = {
        "input": args.input,
        "split": args.split,
        "rows": len(rows),
        "selection_seed": args.seed,
        "arms": {},
    }
    for ratio in RATIOS:
        arm = f"r{int(ratio * 100):03d}"
        selected = nested[ratio]
        records = [rlvr_record(row, row["parent_seed_id"] in selected, arm, ratio) for row in rows]
        write_view(output / "rlvr" / arm, records)
        manifest["arms"][arm] = {
            "rows": len(records),
            "synthetic": sum(record["extra_info"]["origin"] == "synthetic" for record in records),
            "by_source": dict(sorted(Counter(row.get("source") for row in rows).items())),
        }

    fallback_records = [rlvr_record(row, bool(row.get("quality_accept", False)), "fallback", -1.0) for row in rows]
    write_view(output / "rlvr" / "fallback", fallback_records)
    manifest["arms"]["fallback"] = {
        "rows": len(fallback_records),
        "synthetic": sum(record["extra_info"]["origin"] == "synthetic" for record in fallback_records),
    }

    sft_views = {
        "real": [sft_record(row, False, "real") for row in rows],
        "raw_synthetic": [sft_record(row, True, "raw_synthetic") for row in rows],
        "fallback": [sft_record(row, bool(row.get("quality_accept", False)), "fallback") for row in rows],
    }
    for name, records in sft_views.items():
        write_jsonl(output / "sft" / f"{name}.jsonl", records)

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
