#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usage_sum(rows: list[dict], key: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        usage = row.get(key) or {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[name] += int(usage.get(name, 0) or 0)
    return dict(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the final data-generation manifest with hashes and usage.")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/raw/gpt_pairs.jsonl")
    parser.add_argument("--quality", type=Path, default=ROOT / "data/interim/gpt_pairs_quality.jsonl")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifest.json")
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    quality = read_jsonl(args.quality)
    solver_usage: Counter[str] = Counter()
    for row in quality:
        for rollout in row.get("solver_rollouts") or []:
            for name, value in (rollout.get("token_usage") or {}).items():
                if name in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                    solver_usage[name] += int(value or 0)

    artifact_paths = [
        ROOT / "data/canonical/numina_5k_seeds.jsonl",
        ROOT / "data/canonical/numina_5k_manifest.json",
        args.pairs,
        args.quality,
        args.processed / "manifest.json",
    ]
    artifact_paths.extend(sorted((args.processed / "rlvr").glob("*.parquet")))
    artifact_paths.extend(sorted((args.processed / "sft").glob("*.jsonl")))
    manifest = {
        "rows": len(pairs),
        "unique_parent_seeds": len({row["parent_seed_id"] for row in pairs}),
        "by_split": dict(sorted(Counter(row["split"] for row in pairs).items())),
        "by_source": dict(sorted(Counter(row["source"] for row in pairs).items())),
        "generator_model": sorted({row["generator_model"] for row in pairs}),
        "prompt_version": sorted({row["prompt_version"] for row in pairs}),
        "reasoning_effort": sorted({row.get("reasoning_effort") for row in pairs}),
        "generator_usage": usage_sum(pairs, "token_usage"),
        "solver_model": sorted({row.get("solver_model") for row in quality}),
        "solver_usage": dict(solver_usage),
        "quality_accept": sum(bool(row.get("quality_accept")) for row in quality),
        "quality_accept_rate": sum(bool(row.get("quality_accept")) for row in quality) / len(quality),
        "artifacts": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths if path.exists()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
