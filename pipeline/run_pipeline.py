#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATAFLOW = ROOT.parent / "DataFlow"
sys.path.insert(0, os.environ.get("DATAFLOW_HOME", str(DEFAULT_DATAFLOW)))
sys.path.insert(0, str(ROOT))

class InvalidMathDataPipeline:
    def __init__(self, args: argparse.Namespace):
        try:
            from dataflow.serving import APILLMServing_request
            from dataflow.utils.storage import FileStorage
            from pipeline.operators import InvalidQuestionAuditOperator, NaturalMathQuestionGenerator, PseudoAnswerReachabilityOperator
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "DataFlow is not fully installed. Activate its environment or install the sibling DataFlow package first. "
                f"Missing module: {exc.name}"
            ) from exc

        self.storage = FileStorage(
            first_entry_file_name=args.seeds,
            cache_path=args.cache_dir,
            file_name_prefix="invalid_math_dataflow",
            cache_type="jsonl",
        )

        def serving(model: str, temperature: float, max_tokens: int):
            return APILLMServing_request(
                api_url=args.api_url,
                key_name_of_api_key=args.key_env,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_workers=args.workers,
                read_timeout=args.read_timeout,
            )

        self.generator = NaturalMathQuestionGenerator(
            serving(args.generator_model, 0.9, 1800), n_per_seed=args.n_per_seed
        )
        self.auditor = InvalidQuestionAuditOperator(
            serving(args.verifier_model, 0.2, 900), votes=args.votes
        )
        self.reachability = PseudoAnswerReachabilityOperator(
            serving(args.solver_model, 1.0, 2048), rollouts=args.rollouts
        )

    def forward(self) -> None:
        self.generator.run(self.storage.step(), input_key="problem")
        self.auditor.run(self.storage.step(), input_key="problem")
        self.reachability.run(self.storage.step(), input_key="problem", reference_key="reference_answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataFlow-style invalid synthetic math data pipeline.")
    parser.add_argument("--seeds", default=str(ROOT / "data/rl_seed_pools/dapo_math_2k_seeds.jsonl"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data/dataflow_cache"))
    parser.add_argument("--api-url", default=os.environ.get("RIVQ_API_URL", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--key-env", default="DF_API_KEY")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--solver-model", required=True)
    parser.add_argument("--n-per-seed", type=int, default=4)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--read-timeout", type=float, default=240.0)
    return parser.parse_args()


if __name__ == "__main__":
    InvalidMathDataPipeline(parse_args()).forward()
