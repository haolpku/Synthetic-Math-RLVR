#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasketch import MinHash, MinHashLSH

from common import stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "data/raw/numinamath/data/train-00000-of-00001.parquet"
DEFAULT_EVAL_ROOT = Path("/Users/lianghao/Desktop/记忆张量/Qwen2.5-Math/evaluation/data")
SOURCE_QUOTAS = {
    "cn_contest": 2388,
    "amc_aime": 862,
    "olympiads_ref": 750,
    "inequalities": 500,
    "number_theory": 500,
}
EVAL_DATASETS = ("math", "olympiadbench", "college_math", "minerva_math", "aime24", "amc23")


def normalize_text(text: object) -> str:
    value = str(text or "").lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\\(?:left|right|displaystyle|textstyle)", " ", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return " ".join(value.split())


def is_text_only_problem(text: object) -> bool:
    value = str(text or "")
    visual_markers = (
        r"!\[", r"https?://\S+\.(?:png|jpe?g|gif|svg)", r"\bfigure\b", r"\bdiagram\b",
    )
    return not any(re.search(pattern, value, re.IGNORECASE) for pattern in visual_markers)


def shingles(text: str, size: int = 5) -> set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def minhash(values: Iterable[str], num_perm: int = 128) -> MinHash:
    result = MinHash(num_perm=num_perm)
    for value in values:
        result.update(value.encode("utf-8"))
    return result


def iter_eval_questions(root: Path) -> Iterable[tuple[str, str]]:
    for dataset in EVAL_DATASETS:
        for suffix in ("test.jsonl", "test.json"):
            path = root / dataset / suffix
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    values = [json.loads(line) for line in handle if line.strip()]
            else:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                values = loaded if isinstance(loaded, list) else loaded.get("data", [])
            for index, row in enumerate(values):
                question = row.get("problem") or row.get("question") or row.get("query")
                if question:
                    yield f"{dataset}:{index}", str(question)
            break


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical 5K NuminaMath seed pool.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/canonical/numina_5k_seeds.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/canonical/numina_5k_manifest.json")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    args = parser.parse_args()

    columns = [
        "problem", "solution", "answer", "problem_type", "question_type",
        "problem_is_valid", "solution_is_valid", "source", "synthetic",
    ]
    frame = pd.read_parquet(args.parquet, columns=columns)
    eligible = frame[
        frame["source"].isin(SOURCE_QUOTAS)
        & frame["question_type"].eq("math-word-problem")
        & frame["problem_is_valid"].eq("Yes")
        & frame["solution_is_valid"].eq("Yes")
        & frame["synthetic"].eq(False)
        & ~frame["answer"].fillna("").str.strip().str.lower().isin({"", "proof", "notfound"})
    ].copy()
    eligible["normalized_problem"] = eligible["problem"].map(normalize_text)
    eligible = eligible[eligible["normalized_problem"].str.len() >= 20]
    eligible = eligible[eligible["problem"].map(is_text_only_problem)]
    before_dedup = len(eligible)
    eligible = eligible.drop_duplicates("normalized_problem", keep="first")

    eval_rows = list(iter_eval_questions(args.eval_root))
    eval_exact = {normalize_text(question) for _, question in eval_rows}
    lsh = MinHashLSH(threshold=args.near_duplicate_threshold, num_perm=128)
    eval_minhashes: dict[str, MinHash] = {}
    for eval_id, question in eval_rows:
        signature = minhash(shingles(question))
        eval_minhashes[eval_id] = signature
        lsh.insert(eval_id, signature)

    contamination: dict[int, list[str]] = {}
    clean_indices: list[int] = []
    for index, row in eligible.iterrows():
        normalized = row["normalized_problem"]
        hits = ["exact"] if normalized in eval_exact else lsh.query(minhash(shingles(normalized)))
        if hits:
            contamination[int(index)] = sorted(hits)
        else:
            clean_indices.append(int(index))
    clean = eligible.loc[clean_indices].copy()

    rng = random.Random(args.seed)
    available = Counter(clean["source"])
    realized_quotas = dict(SOURCE_QUOTAS)
    amc_shortfall = max(0, SOURCE_QUOTAS["amc_aime"] - available["amc_aime"])
    realized_quotas["amc_aime"] -= amc_shortfall
    realized_quotas["cn_contest"] += amc_shortfall
    selected_parts = []
    for source, quota in realized_quotas.items():
        part = clean[clean["source"].eq(source)].copy()
        if len(part) < quota:
            raise SystemExit(f"source {source!r} has {len(part)} clean rows, below quota {quota}")
        indices = list(part.index)
        rng.shuffle(indices)
        selected_parts.append(part.loc[indices[:quota]])
    selected = pd.concat(selected_parts, ignore_index=True)

    order = list(range(len(selected)))
    rng.shuffle(order)
    split_for_position = {
        **{position: "train" for position in order[:4000]},
        **{position: "development" for position in order[4000:4500]},
        **{position: "held_out_audit" for position in order[4500:]},
    }
    rows = []
    for position, row in selected.iterrows():
        seed_id = stable_id(row["source"], row["problem"], row["answer"], prefix="numina")
        rows.append({
            "parent_seed_id": seed_id,
            "seed_id": seed_id,
            "split": split_for_position[position],
            "source": row["source"],
            "problem_type": row["problem_type"],
            "question_type": row["question_type"],
            "domain": row["problem_type"],
            "difficulty": "competition_math",
            "real_problem": str(row["problem"]).strip(),
            "real_solution": str(row["solution"]).strip(),
            "real_answer": str(row["answer"]).strip(),
            # Prompt compatibility with the generation script.
            "problem": str(row["problem"]).strip(),
            "solution": str(row["solution"]).strip(),
            "answer": str(row["answer"]).strip(),
        })
    rows.sort(key=lambda item: (item["split"], item["source"], item["parent_seed_id"]))
    write_jsonl(args.output, rows)

    manifest = {
        "dataset": "nlile/NuminaMath-1.5-RL-Verifiable",
        "input_parquet": str(args.parquet),
        "input_sha256": sha256_file(args.parquet),
        "selection_seed": args.seed,
        "requested_source_quotas": SOURCE_QUOTAS,
        "source_quotas": realized_quotas,
        "eligible_before_exact_dedup": before_dedup,
        "eligible_after_exact_dedup": len(eligible),
        "evaluation_datasets": list(EVAL_DATASETS),
        "evaluation_question_count": len(eval_rows),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "contaminated_candidate_count": len(contamination),
        "available_clean_by_source": dict(sorted(available.items())),
        "selected_by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
        "selected_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
