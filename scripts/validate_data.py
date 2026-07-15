#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unique IDs and required fields in a JSONL artifact.")
    parser.add_argument("input")
    parser.add_argument("--required", nargs="+", default=["sample_id"])
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    ids = [row.get("sample_id") or (row.get("extra_info") or {}).get("sample_id") for row in rows]
    duplicates = [key for key, count in Counter(ids).items() if key is not None and count > 1]
    missing = [(index, key) for index, row in enumerate(rows) for key in args.required if key not in row]
    if duplicates or missing:
        raise SystemExit(f"invalid dataset: duplicate_ids={duplicates[:5]}, missing_fields={missing[:5]}")
    print(f"OK: {len(rows)} rows, unique IDs, required fields present")


if __name__ == "__main__":
    main()
