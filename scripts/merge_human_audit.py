#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv

from common import parse_bool, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed expert audit CSV into the generation pool.")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.audit, newline="", encoding="utf-8-sig") as handle:
        audit = {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("sample_id")}
    output = []
    for row in read_jsonl(args.pool):
        label = audit.get(row["sample_id"])
        if label:
            row = {
                **row,
                "human_audit": {
                    "annotator_id": label.get("annotator_id", ""),
                    "question_valid": parse_bool(label.get("question_valid")),
                    "invalid_type": label.get("invalid_type", ""),
                    "unique_answer_exists": parse_bool(label.get("unique_answer_exists")),
                    "reference_answer_correct": parse_bool(label.get("reference_answer_correct")),
                    "difficulty": label.get("difficulty", ""),
                    "confidence": label.get("confidence", ""),
                    "notes": label.get("notes", ""),
                },
            }
        output.append(row)
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
