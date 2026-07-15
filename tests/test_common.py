from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import answer_matches, extract_boxed, extract_json_object, parse_bool, stable_id


class CommonTest(unittest.TestCase):
    def test_parse_bool_does_not_treat_false_string_as_true(self) -> None:
        self.assertIs(parse_bool("false"), False)
        self.assertIs(parse_bool("TRUE"), True)
        self.assertIsNone(parse_bool("maybe"))

    def test_extract_nested_boxed_answer(self) -> None:
        self.assertEqual(extract_boxed(r"work … \\boxed{\\frac{1}{2}}"), r"\\frac{1}{2}")

    def test_json_fence_and_answer_normalization(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"ok": true}\n```')["ok"], True)
        self.assertTrue(answer_matches(r"\\dfrac{1}{2}", r"\\frac{1}{2}"))

    def test_repairs_unescaped_latex_in_json(self) -> None:
        value = extract_json_object(r'{"problem":"Find \frac{1}{2} and \(x+1\)."}')
        self.assertIn(r"\frac", value["problem"])

    def test_stable_id(self) -> None:
        self.assertEqual(stable_id("a", 1), stable_id("a", 1))
        self.assertNotEqual(stable_id("a", 1), stable_id("a", 2))


if __name__ == "__main__":
    unittest.main()
