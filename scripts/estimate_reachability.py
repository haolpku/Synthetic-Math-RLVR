#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from api_client import OpenAICompatibleClient
from common import answer_matches, extract_boxed, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate on-policy pseudo-answer reachability with K solver rollouts.")
    parser.add_argument("--input", default=str(ROOT / "data/interim/auto_verified.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/interim/reachability.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    template = (ROOT / "prompts/solve_question.txt").read_text(encoding="utf-8")
    source = read_jsonl(args.input)
    existing = read_jsonl(args.output) if Path(args.output).exists() else []
    completed = {row["sample_id"] for row in existing}
    pending = [row for row in source if row["sample_id"] not in completed]
    client = OpenAICompatibleClient()

    def run(row: dict) -> dict:
        prompt = template.format(problem=row["problem"])

        def solve(index: int) -> dict:
            response = client.chat(
                args.model,
                [{"role": "system", "content": "Solve carefully and follow the requested answer format."},
                 {"role": "user", "content": prompt}],
                args.temperature,
                args.max_tokens,
                seed=index,
            )
            prediction = extract_boxed(response["content"])
            return {
                "rollout": index,
                "prediction": prediction,
                "matches_pseudo_reference": answer_matches(prediction, row["reference_answer"]),
                "declared_invalid": str(prediction or "").strip().upper() == "INVALID",
                "response": response["content"],
            }

        rollouts = [solve(index) for index in range(args.rollouts)]
        matches = sum(item["matches_pseudo_reference"] for item in rollouts)
        p_hat = matches / args.rollouts
        k = args.rollouts
        return {
            **row,
            "solver_model": args.model,
            "rollouts_k": k,
            "pseudo_answer_matches": matches,
            "pseudo_answer_p_hat": p_hat,
            "observed_group_regime": "dead" if matches == 0 else "pseudo_saturated" if matches == k else "deceptive_active",
            "predicted_dead_probability": (1.0 - p_hat) ** k,
            "predicted_active_probability": 1.0 - (1.0 - p_hat) ** k - p_hat ** k,
            "predicted_saturated_probability": p_hat ** k,
            "solver_rollouts": rollouts,
        }

    rows = list(existing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(run, pending):
            rows.append(row)
            write_jsonl(args.output, rows)
            print(json.dumps({"sample_id": row["sample_id"], "p_hat": row["pseudo_answer_p_hat"], "regime": row["observed_group_regime"]}))


if __name__ == "__main__":
    main()
