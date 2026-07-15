#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from common import answer_matches, read_jsonl, write_jsonl
from prepare_numina_seeds import EVAL_DATASETS, DEFAULT_EVAL_ROOT, iter_eval_questions, minhash, normalize_text, shingles


ROOT = Path(__file__).resolve().parents[1]


def answer_parseable(answer: object) -> bool:
    value = str(answer or "").strip()
    if not value or value.lower() in {"proof", "notfound", "invalid"}:
        return False
    try:
        from training.reward_fn import compute_score

        return compute_score("smoke", f"\\boxed{{{value}}}", value) == 1.0
    except Exception:
        return bool(re.search(r"[0-9a-zA-Z\\]", value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply automatic, label-free quality checks to paired questions.")
    parser.add_argument("--input", default=str(ROOT / "data/interim/gpt_pairs_solved.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/interim/gpt_pairs_quality.jsonl"))
    parser.add_argument("--manifest", default=str(ROOT / "data/interim/gpt_pairs_quality_manifest.json"))
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--required-solver-matches", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    eval_lsh = MinHashLSH(threshold=args.near_duplicate_threshold, num_perm=128)
    eval_exact: set[str] = set()
    eval_count = 0
    for eval_id, question in iter_eval_questions(args.eval_root):
        normalized = normalize_text(question)
        eval_exact.add(normalized)
        eval_lsh.insert(eval_id, minhash(shingles(question)))
        eval_count += 1

    synthetic_lsh = MinHashLSH(threshold=args.near_duplicate_threshold, num_perm=128)
    seen_exact: dict[str, str] = {}
    enriched = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        problem = str(row.get("synthetic_problem", ""))
        solution = str(row.get("synthetic_solution", ""))
        answer = str(row.get("synthetic_answer", ""))
        normalized = normalize_text(problem)
        signature = minhash(shingles(problem))
        schema_ok = bool(problem.strip() and solution.strip() and answer.strip())
        parseable = answer_parseable(answer)
        eval_hits = ["exact"] if normalized in eval_exact else sorted(eval_lsh.query(signature))
        duplicate_hits = []
        if normalized in seen_exact:
            duplicate_hits.append(seen_exact[normalized])
        duplicate_hits.extend(synthetic_lsh.query(signature))
        duplicate_hits = sorted(set(duplicate_hits))
        real_similarity_hit = False
        real_shingles = shingles(row.get("real_problem", ""))
        syn_shingles = shingles(problem)
        if real_shingles and syn_shingles:
            jaccard = len(real_shingles & syn_shingles) / len(real_shingles | syn_shingles)
            real_similarity_hit = jaccard >= args.near_duplicate_threshold
        else:
            jaccard = 0.0

        if row.get("solver_rollouts"):
            solver_matches = sum(
                answer_matches(item.get("prediction"), answer)
                for item in row["solver_rollouts"]
            )
        else:
            solver_matches = int(row.get("solver_matches", -1))
        solver_ok = solver_matches >= args.required_solver_matches
        quality_accept = schema_ok and parseable and solver_ok and not eval_hits and not duplicate_hits and not real_similarity_hit
        enriched.append({
            **row,
            "schema_ok": schema_ok,
            "answer_parseable": parseable,
            "eval_contaminated": bool(eval_hits),
            "eval_contamination_hits": eval_hits,
            "synthetic_duplicate": bool(duplicate_hits),
            "synthetic_duplicate_hits": duplicate_hits,
            "parent_near_duplicate": real_similarity_hit,
            "parent_problem_jaccard": jaccard,
            "required_solver_matches": args.required_solver_matches,
            "solver_matches": solver_matches,
            "solver_match_rate": solver_matches / len(row["solver_rollouts"]) if row.get("solver_rollouts") else row.get("solver_match_rate"),
            "quality_accept": quality_accept,
        })
        if normalized and normalized not in seen_exact:
            seen_exact[normalized] = row["sample_id"]
            synthetic_lsh.insert(row["sample_id"], signature)

    write_jsonl(args.output, enriched)
    manifest = {
        "input": args.input,
        "rows": len(enriched),
        "evaluation_datasets": list(EVAL_DATASETS),
        "evaluation_questions": eval_count,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "required_solver_matches": args.required_solver_matches,
        "quality_accept": sum(bool(row["quality_accept"]) for row in enriched),
        "check_failures": {
            key: sum(not bool(row[key]) for row in enriched)
            for key in ("schema_ok", "answer_parseable")
        } | {
            "eval_contaminated": sum(bool(row["eval_contaminated"]) for row in enriched),
            "synthetic_duplicate": sum(bool(row["synthetic_duplicate"]) for row in enriched),
            "parent_near_duplicate": sum(bool(row["parent_near_duplicate"]) for row in enriched),
            "solver_rejected": sum(int(row.get("solver_matches", -1)) < args.required_solver_matches for row in enriched),
        },
        "by_source": dict(sorted(Counter(row.get("source") for row in enriched).items())),
    }
    target = Path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
