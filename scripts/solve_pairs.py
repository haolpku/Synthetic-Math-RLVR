#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from api_client import OpenAICompatibleClient
from common import answer_matches, append_jsonl, extract_boxed, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def rollout_seed(sample_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{sample_id}\x1fsolver\x1f{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent solver-consistency checks on synthetic pairs.")
    parser.add_argument("--input", default=str(ROOT / "data/raw/gpt_pairs.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/interim/gpt_pairs_solved.jsonl"))
    parser.add_argument("--failed-output", default=str(ROOT / "data/interim/gpt_pairs_solver_failures.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="low")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    template = (ROOT / "prompts/solve_question.txt").read_text(encoding="utf-8")
    source = read_jsonl(args.input)
    existing = read_jsonl(args.output) if Path(args.output).exists() else []
    completed = {row["sample_id"] for row in existing}
    pending = [row for row in source if row["sample_id"] not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]
    failures = read_jsonl(args.failed_output) if Path(args.failed_output).exists() else []
    client = OpenAICompatibleClient()

    def run(row: dict) -> dict:
        prompt = template.format(problem=row["synthetic_problem"])
        rollouts = []
        for index in range(args.rollouts):
            response = client.chat(
                args.model,
                [
                    {"role": "system", "content": "Solve independently. Follow the requested boxed-answer format."},
                    {"role": "user", "content": prompt},
                ],
                args.temperature,
                args.max_tokens,
                seed=rollout_seed(row["sample_id"], index),
                reasoning_effort=args.reasoning_effort,
            )
            prediction = extract_boxed(response["content"])
            rollouts.append({
                "rollout": index,
                "prediction": prediction,
                "matches_reference": answer_matches(prediction, row["synthetic_answer"]),
                "declared_invalid": str(prediction or "").strip().upper() == "INVALID",
                "raw_response": response["content"],
                "token_usage": response["usage"],
            })
        matches = sum(item["matches_reference"] for item in rollouts)
        return {
            **row,
            "solver_model": args.model,
            "solver_temperature": args.temperature,
            "solver_reasoning_effort": args.reasoning_effort,
            "solver_rollouts_k": args.rollouts,
            "solver_matches": matches,
            "solver_match_rate": matches / args.rollouts,
            "solver_declared_invalid": sum(item["declared_invalid"] for item in rollouts),
            "solver_rollouts": rollouts,
            "solved_at": datetime.now(timezone.utc).isoformat(),
        }

    rows = list(existing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, row): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            source_row = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                failures.append({
                    "sample_id": source_row["sample_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                })
                append_jsonl(args.failed_output, failures[-1])
                print(json.dumps({"status": "failed", "sample_id": source_row["sample_id"], "error": str(exc)[:300]}), flush=True)
                continue
            rows.append(row)
            append_jsonl(args.output, row)
            print(json.dumps({"status": "solved", "sample_id": row["sample_id"], "matches": row["solver_matches"], "count": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
