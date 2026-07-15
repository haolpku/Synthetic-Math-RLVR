#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RLVR JSONL to verl-compatible parquet.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    print(f"wrote {len(rows)} rows to {target}")


if __name__ == "__main__":
    main()
