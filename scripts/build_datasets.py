#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def human_label(row: dict) -> tuple[bool | None, bool | None]:
    audit = row.get("human_audit") or {}
    return audit.get("question_valid"), audit.get("reference_answer_correct")


def assign_group(row: dict) -> str:
    valid, reference_correct = human_label(row)
    if valid is True and reference_correct is True:
        return "valid_correct"
    if valid is False:
        return "invalid_pseudo"
    if row.get("auto_unanimous_valid") and row.get("auto_majority_reference_correct"):
        return "auto_verified"
    return "raw_unverified"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched SFT and RLVR datasets from the audited pool.")
    parser.add_argument("--input", default=str(ROOT / "data/interim/reachability.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/processed"))
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-per-group", type=int, default=0, help="0 keeps all rows.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    rng = random.Random(args.seed)
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        group = assign_group(row)
        by_group.setdefault(group, []).append({**row, "experimental_group": group})

    selected: list[dict] = []
    for group, items in sorted(by_group.items()):
        rng.shuffle(items)
        selected.extend(items if args.max_per_group <= 0 else items[: args.max_per_group])

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "canonical.jsonl", selected)
    write_jsonl(output_dir / "sft.jsonl", [
        {
            "sample_id": row["sample_id"],
            "group": row["experimental_group"],
            "messages": [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": row["reference_solution"]},
            ],
        }
        for row in selected
    ])
    write_jsonl(output_dir / "rlvr.jsonl", [
        {
            "data_source": "rivq_math",
            "prompt": [{"role": "user", "content": row["problem"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(row["reference_answer"])},
            "extra_info": {
                "sample_id": row["sample_id"],
                "group": row["experimental_group"],
                "question_valid": human_label(row)[0],
                "pseudo_answer_p_hat": row.get("pseudo_answer_p_hat"),
            },
        }
        for row in selected
    ])
    write_jsonl(output_dir / "invalid_eval.jsonl", [row for row in selected if human_label(row)[0] is False])
    for group, items in sorted(by_group.items()):
        print(f"{group}: {len(items)} available")


if __name__ == "__main__":
    main()
