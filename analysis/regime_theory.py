#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate dead-active-saturated probabilities over p and K.")
    parser.add_argument("--output", default="data/processed/regime_theory.csv")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()
    ps = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0]
    rows = []
    for k in args.ks:
        for p in ps:
            dead = (1 - p) ** k
            saturated = p ** k
            rows.append({"K": k, "p": p, "dead": dead, "active": 1 - dead - saturated, "saturated": saturated})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
