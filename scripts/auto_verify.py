#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

from api_client import OpenAICompatibleClient
from common import extract_json_object, parse_bool, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
VALID_TYPES = {"valid", "underdetermined", "contradictory", "infeasible", "ambiguous_expression", "other_invalid"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated question-validity audits with an LLM judge.")
    parser.add_argument("--input", default=str(ROOT / "data/raw/generated_pool.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/interim/auto_verified.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    template = (ROOT / "prompts/verify_question.txt").read_text(encoding="utf-8")
    source = read_jsonl(args.input)
    existing = read_jsonl(args.output) if Path(args.output).exists() else []
    completed = {row["sample_id"] for row in existing}
    # Stored rows are aggregated by sample, so resume only completed samples.
    pending = [row for row in source if row["sample_id"] not in completed]
    client = OpenAICompatibleClient()

    def judge(row: dict, vote: int) -> dict:
        prompt = template.format(**row)
        response = client.chat(
            args.model,
            [{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            args.temperature,
            args.max_tokens,
            seed=vote,
        )
        verdict = extract_json_object(response["content"])
        invalid_type = str(verdict.get("invalid_type", "other_invalid"))
        if invalid_type not in VALID_TYPES:
            invalid_type = "other_invalid"
        question_valid = parse_bool(verdict.get("question_valid"))
        unique_answer_exists = parse_bool(verdict.get("unique_answer_exists"))
        reference_answer_correct = parse_bool(verdict.get("reference_answer_correct"))
        return {
            "question_valid": question_valid is True,
            "invalid_type": invalid_type,
            "unique_answer_exists": unique_answer_exists is True,
            "reference_answer_correct": reference_answer_correct is True,
            "confidence": float(verdict.get("confidence", 0.0)),
            "short_reason": str(verdict.get("short_reason", ""))[:1000],
            "vote": vote,
        }

    def run(row: dict) -> dict:
        votes = [judge(row, vote) for vote in range(args.votes)]
        valid_votes = sum(v["question_valid"] for v in votes)
        reference_votes = sum(v["reference_answer_correct"] for v in votes)
        types = Counter(v["invalid_type"] for v in votes)
        return {
            **row,
            "auto_verifier_model": args.model,
            "auto_votes": votes,
            "auto_valid_votes": valid_votes,
            "auto_reference_correct_votes": reference_votes,
            "auto_majority_valid": valid_votes > args.votes / 2,
            "auto_unanimous_valid": valid_votes == args.votes,
            "auto_majority_reference_correct": reference_votes > args.votes / 2,
            "auto_invalid_type": types.most_common(1)[0][0],
        }

    rows = list(existing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(run, pending):
            rows.append(row)
            write_jsonl(args.output, rows)
            print(json.dumps({"sample_id": row["sample_id"], "valid_votes": row["auto_valid_votes"]}))


if __name__ == "__main__":
    main()
