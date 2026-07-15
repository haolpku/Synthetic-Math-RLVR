#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from api_client import OpenAICompatibleClient
from common import append_jsonl, extract_json_object, read_jsonl, stable_id


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("problem", "reference_solution", "reference_answer", "claimed_domain", "claimed_difficulty")


def generation_seed(parent_seed_id: str, index: int) -> int:
    raw = hashlib.sha256(f"{parent_seed_id}\x1f{index}".encode()).digest()
    return int.from_bytes(raw[:4], "big") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired synthetic questions from canonical real seeds.")
    parser.add_argument("--seeds", default=str(ROOT / "data/canonical/numina_5k_seeds.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/raw/gpt_pairs.jsonl"))
    parser.add_argument("--failed-output", default=str(ROOT / "data/raw/gpt_pairs_failures.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", default="concept_de_novo_v2_single_target")
    parser.add_argument("--n-per-seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="low")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--split", choices=["train", "development", "held_out_audit", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit pending parent seeds; 0 means all.")
    parser.add_argument("--sample-seed", type=int, default=20260714, help="Deterministically shuffle jobs before --limit.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    template = (ROOT / "prompts/generate_question.txt").read_text(encoding="utf-8")
    seeds = read_jsonl(args.seeds)
    if args.split != "all":
        seeds = [row for row in seeds if row.get("split") == args.split]
    existing = read_jsonl(args.output) if Path(args.output).exists() else []
    completed = {row["sample_id"] for row in existing}
    jobs = []
    for seed in seeds:
        parent_id = seed.get("parent_seed_id") or seed["seed_id"]
        for index in range(args.n_per_seed):
            sample_id = stable_id(parent_id, args.model, args.prompt_version, args.reasoning_effort, index, prefix="pair")
            jobs.append((seed, index, sample_id))
    if args.limit > 0:
        random.Random(args.sample_seed).shuffle(jobs)
        jobs = jobs[: args.limit]
    jobs = [job for job in jobs if job[2] not in completed]

    if args.dry_run:
        if not jobs:
            print("No pending jobs.")
            return
        seed, _, sample_id = jobs[0]
        print(json.dumps({"sample_id": sample_id, "prompt": template.format(**seed)}, ensure_ascii=False, indent=2))
        return

    client = OpenAICompatibleClient()

    def run(job: tuple[dict, int, str]) -> dict:
        seed, index, sample_id = job
        parent_id = seed.get("parent_seed_id") or seed["seed_id"]
        prompt = template.format(**seed)
        errors: list[str] = []
        for schema_attempt in range(args.schema_retries + 1):
            seed_value = generation_seed(parent_id, index) + schema_attempt
            response = client.chat(
                args.model,
                [{"role": "system", "content": "Return only one valid JSON object."}, {"role": "user", "content": prompt}],
                args.temperature,
                args.max_tokens,
                seed=seed_value,
                reasoning_effort=args.reasoning_effort,
            )
            try:
                parsed = extract_json_object(response["content"])
                missing = [key for key in REQUIRED if not str(parsed.get(key, "")).strip()]
                if missing:
                    raise ValueError(f"missing fields {missing}")
                return {
                    "sample_id": sample_id,
                    "parent_seed_id": parent_id,
                    "split": seed["split"],
                    "source": seed.get("source"),
                    "problem_type": seed.get("problem_type"),
                    "question_type": seed.get("question_type"),
                    "real_problem": seed["real_problem"],
                    "real_solution": seed["real_solution"],
                    "real_answer": seed["real_answer"],
                    "synthetic_problem": str(parsed["problem"]).strip(),
                    "synthetic_solution": str(parsed["reference_solution"]).strip(),
                    "synthetic_answer": str(parsed["reference_answer"]).strip(),
                    "claimed_domain": str(parsed["claimed_domain"]).strip(),
                    "claimed_difficulty": str(parsed["claimed_difficulty"]).strip(),
                    "generator_model": args.model,
                    "prompt_version": args.prompt_version,
                    "generation_index": index,
                    "generation_seed": seed_value,
                    "generation_temperature": args.temperature,
                    "reasoning_effort": args.reasoning_effort,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "raw_response": response["content"],
                    "token_usage": response["usage"],
                    "request_attempts": response["attempts"],
                    "schema_attempts": schema_attempt + 1,
                }
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
        raise RuntimeError("; ".join(errors))

    rows = list(existing)
    failures = read_jsonl(args.failed_output) if Path(args.failed_output).exists() else []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(run, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            seed, index, sample_id = future_to_job[future]
            try:
                row = future.result()
            except Exception as exc:  # failure is persisted and remains resumable
                failure = {
                    "sample_id": sample_id,
                    "parent_seed_id": seed.get("parent_seed_id") or seed["seed_id"],
                    "split": seed.get("split"),
                    "generator_model": args.model,
                    "prompt_version": args.prompt_version,
                    "generation_index": index,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                }
                failures.append(failure)
                append_jsonl(args.failed_output, failure)
                print(json.dumps({"status": "failed", "sample_id": sample_id, "error": str(exc)[:300]}), flush=True)
                continue
            rows.append(row)
            append_jsonl(args.output, row)
            print(json.dumps({"status": "generated", "sample_id": row["sample_id"], "count": len(rows)}), flush=True)

    print(json.dumps({"complete": len(rows), "failed_attempt_records": len(failures), "pending_at_start": len(jobs)}))


if __name__ == "__main__":
    main()
