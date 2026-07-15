#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from common import read_jsonl


FIELDS = [
    "sample_id", "problem", "reference_solution", "reference_answer",
    "auto_valid_votes", "auto_reference_correct_votes", "auto_invalid_type",
    "annotator_id", "question_valid", "invalid_type", "unique_answer_exists",
    "reference_answer_correct", "difficulty", "confidence", "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded CSV for expert question audit.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--hide-auto-labels", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: min(args.sample_size, len(rows))]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in FIELDS}
            if args.hide_auto_labels:
                for key in ("auto_valid_votes", "auto_reference_correct_votes", "auto_invalid_type"):
                    output[key] = ""
            writer.writerow(output)


if __name__ == "__main__":
    main()

