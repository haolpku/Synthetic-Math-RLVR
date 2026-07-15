from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "DataFlow"))

try:
    from dataflow.utils.storage import FileStorage
    from pipeline.operators import InvalidQuestionAuditOperator, NaturalMathQuestionGenerator, PseudoAnswerReachabilityOperator
except ModuleNotFoundError:
    FileStorage = None


class DummyServing:
    def __init__(self, model_name: str, response: str):
        self.model_name = model_name
        self.response = response

    def generate_from_input(self, user_inputs, system_prompt="You are helpful"):
        return [self.response for _ in user_inputs]


@unittest.skipIf(FileStorage is None, "DataFlow environment not installed")
class DataFlowPipelineTest(unittest.TestCase):
    def test_three_operator_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            seed_path = temp / "seeds.jsonl"
            seed_path.write_text(json.dumps({
                "seed_id": "s1", "source": "test", "domain": "algebra",
                "difficulty": "middle_school", "problem": "What is 1+1?", "answer": "2",
            }) + "\n", encoding="utf-8")
            storage = FileStorage(str(seed_path), str(temp), "smoke", "jsonl")

            generator = NaturalMathQuestionGenerator(DummyServing("generator", json.dumps({
                "problem": "Find x if x+y=2.",
                "reference_solution": "Assume y=0, so x=2. \\boxed{2}",
                "reference_answer": "2",
                "claimed_domain": "algebra",
                "claimed_difficulty": "middle_school",
            })), n_per_seed=1)
            auditor = InvalidQuestionAuditOperator(DummyServing("verifier", json.dumps({
                "question_valid": False,
                "invalid_type": "underdetermined",
                "unique_answer_exists": False,
                "reference_answer_correct": False,
                "confidence": 0.99,
                "short_reason": "y is unspecified",
            })), votes=3)
            reachability = PseudoAnswerReachabilityOperator(DummyServing("solver", r"work \\boxed{2}"), rollouts=4)

            generator.run(storage.step())
            auditor.run(storage.step())
            reachability.run(storage.step())
            result = storage.step().read("dataframe").iloc[0]
            self.assertEqual(result["auto_valid_votes"], 0)
            self.assertEqual(result["pseudo_answer_p_hat"], 1.0)
            self.assertEqual(result["observed_group_regime"], "pseudo_saturated")


if __name__ == "__main__":
    unittest.main()
