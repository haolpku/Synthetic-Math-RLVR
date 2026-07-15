from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.utils.storage import DataFlowStorage

from scripts.common import answer_matches, extract_boxed, extract_json_object, parse_bool, stable_id


ROOT = Path(__file__).resolve().parents[1]
VALID_TYPES = {"valid", "underdetermined", "contradictory", "infeasible", "ambiguous_expression", "other_invalid"}


class NaturalMathQuestionGenerator(OperatorABC):
    """Expand seed problems into naturally generated question-solution pairs."""

    def __init__(self, llm_serving: LLMServingABC, n_per_seed: int = 4):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.n_per_seed = n_per_seed
        self.template = (ROOT / "prompts/generate_question.txt").read_text(encoding="utf-8")

    def run(self, storage: DataFlowStorage, input_key: str = "problem") -> list[str]:
        dataframe = storage.read("dataframe")
        required = {"seed_id", "domain", "difficulty", input_key}
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(f"generation input missing columns: {missing}")
        jobs: list[tuple[dict, int]] = []
        prompts: list[str] = []
        for row in dataframe.to_dict(orient="records"):
            seed = {**row, "problem": row[input_key]}
            for index in range(self.n_per_seed):
                jobs.append((seed, index))
                prompts.append(self.template.format(**seed))
        responses = self.llm_serving.generate_from_input(prompts, "Return only valid JSON.")
        output = []
        for (seed, index), response in zip(jobs, responses):
            if not response:
                continue
            parsed = extract_json_object(response)
            required_output = ["problem", "reference_solution", "reference_answer", "claimed_domain", "claimed_difficulty"]
            if any(not str(parsed.get(key, "")).strip() for key in required_output):
                self.logger.warning("Skipping malformed generation for seed %s", seed["seed_id"])
                continue
            output.append({
                "sample_id": stable_id(seed["seed_id"], self.llm_serving.model_name, index, prefix="syn"),
                "seed_id": seed["seed_id"],
                "seed_source": seed.get("source"),
                "generator_model": self.llm_serving.model_name,
                "generation_index": index,
                **{key: parsed[key] for key in required_output},
            })
        output_file = storage.write(pd.DataFrame(output))
        self.logger.info("Natural generations saved to %s", output_file)
        return ["sample_id", "problem", "reference_solution", "reference_answer"]


class InvalidQuestionAuditOperator(OperatorABC):
    """Attach repeated question-validity votes without filtering samples away."""

    def __init__(self, llm_serving: LLMServingABC, votes: int = 3):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.votes = votes
        self.template = (ROOT / "prompts/verify_question.txt").read_text(encoding="utf-8")

    def run(self, storage: DataFlowStorage, input_key: str = "problem") -> list[str]:
        dataframe = storage.read("dataframe")
        jobs = [(row, vote) for row in dataframe.to_dict(orient="records") for vote in range(self.votes)]
        prompts = [self.template.format(**row) for row, _ in jobs]
        responses = self.llm_serving.generate_from_input(prompts, "Return only valid JSON.")
        by_id: dict[str, list[dict]] = {str(row.sample_id): [] for row in dataframe.itertuples()}
        for (row, vote), response in zip(jobs, responses):
            try:
                verdict = extract_json_object(response or "")
                invalid_type = str(verdict.get("invalid_type", "other_invalid"))
                if invalid_type not in VALID_TYPES:
                    invalid_type = "other_invalid"
                by_id[row["sample_id"]].append({
                    "vote": vote,
                    "question_valid": parse_bool(verdict.get("question_valid")) is True,
                    "unique_answer_exists": parse_bool(verdict.get("unique_answer_exists")) is True,
                    "reference_answer_correct": parse_bool(verdict.get("reference_answer_correct")) is True,
                    "invalid_type": invalid_type,
                    "confidence": float(verdict.get("confidence", 0.0)),
                    "short_reason": str(verdict.get("short_reason", ""))[:1000],
                })
            except (ValueError, TypeError):
                by_id[row["sample_id"]].append({"vote": vote, "question_valid": False, "unique_answer_exists": False,
                                                 "reference_answer_correct": False, "invalid_type": "other_invalid",
                                                 "confidence": 0.0, "short_reason": "unparseable verifier response"})
        records = []
        for row in dataframe.to_dict(orient="records"):
            votes = by_id[row["sample_id"]]
            valid_count = sum(v["question_valid"] for v in votes)
            reference_count = sum(v["reference_answer_correct"] for v in votes)
            records.append({
                **row,
                "auto_verifier_model": self.llm_serving.model_name,
                "auto_votes": votes,
                "auto_valid_votes": valid_count,
                "auto_reference_correct_votes": reference_count,
                "auto_majority_valid": valid_count > self.votes / 2,
                "auto_unanimous_valid": valid_count == self.votes,
                "auto_majority_reference_correct": reference_count > self.votes / 2,
                "auto_invalid_type": Counter(v["invalid_type"] for v in votes).most_common(1)[0][0],
            })
        output_file = storage.write(pd.DataFrame(records))
        self.logger.info("Audited generations saved to %s", output_file)
        return [input_key, "auto_valid_votes", "auto_invalid_type"]


class PseudoAnswerReachabilityOperator(OperatorABC):
    """Estimate p(answer=pseudo-reference) and attach dead/active/saturated diagnostics."""

    def __init__(self, llm_serving: LLMServingABC, rollouts: int = 8):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.rollouts = rollouts
        self.template = (ROOT / "prompts/solve_question.txt").read_text(encoding="utf-8")

    def run(self, storage: DataFlowStorage, input_key: str = "problem", reference_key: str = "reference_answer") -> list[str]:
        dataframe = storage.read("dataframe")
        jobs = [(row, index) for row in dataframe.to_dict(orient="records") for index in range(self.rollouts)]
        prompts = [self.template.format(problem=row[input_key]) for row, _ in jobs]
        responses = self.llm_serving.generate_from_input(prompts, "Solve carefully and follow the requested answer format.")
        by_id: dict[str, list[dict]] = {str(row.sample_id): [] for row in dataframe.itertuples()}
        for (row, index), response in zip(jobs, responses):
            prediction = extract_boxed(response or "")
            by_id[row["sample_id"]].append({
                "rollout": index,
                "prediction": prediction,
                "matches_pseudo_reference": answer_matches(prediction, row[reference_key]),
                "declared_invalid": str(prediction or "").strip().upper() == "INVALID",
                "response": response,
            })
        records = []
        for row in dataframe.to_dict(orient="records"):
            rollouts = by_id[row["sample_id"]]
            matches = sum(item["matches_pseudo_reference"] for item in rollouts)
            p_hat, k = matches / self.rollouts, self.rollouts
            records.append({
                **row,
                "solver_model": self.llm_serving.model_name,
                "rollouts_k": k,
                "pseudo_answer_matches": matches,
                "pseudo_answer_p_hat": p_hat,
                "observed_group_regime": "dead" if matches == 0 else "pseudo_saturated" if matches == k else "deceptive_active",
                "predicted_dead_probability": (1 - p_hat) ** k,
                "predicted_active_probability": 1 - (1 - p_hat) ** k - p_hat ** k,
                "predicted_saturated_probability": p_hat ** k,
                "solver_rollouts": rollouts,
            })
        output_file = storage.write(pd.DataFrame(records))
        self.logger.info("Reachability annotations saved to %s", output_file)
        return [input_key, reference_key, "pseudo_answer_p_hat", "observed_group_regime"]
