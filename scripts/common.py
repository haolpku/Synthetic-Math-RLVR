#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


TRUE_VALUES = {"1", "true", "yes", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "f"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_id(*parts: object, prefix: str = "item") -> str:
    raw = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    candidate = stripped if start < 0 or end <= start else stripped[start : end + 1]
    latex_command = (
        r"frac|dfrac|tfrac|sqrt|boxed|begin|end|left|right|cdot|times|div|pm|mp|"
        r"leq?|geq?|neq|approx|equiv|ldots|cdots|dots|text|mathrm|mathbf|mathbb|"
        r"operatorname|binom|sum|prod|int|lim|log|ln|sin|cos|tan|theta|alpha|beta|"
        r"gamma|delta|pi|infty|overline|underline|vec"
    )
    candidate = re.sub(
        rf'(?<!\\)\\(?=(?:{latex_command})\b|[()\[\]])',
        r'\\\\',
        candidate,
    )
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as original_error:
        if start < 0 or end <= start:
            raise ValueError("model response contains no JSON object") from original_error
        # Models frequently emit LaTeX such as \frac or \( inside otherwise
        # valid JSON without escaping the backslash. Preserve valid JSON escape
        # sequences and double only the invalid ones.
        repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', candidate)
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                value = repair_json(candidate, return_objects=True)
            except Exception:
                raise original_error
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    pos, depth = start + len(marker), 1
    for index in range(pos, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[pos:index].strip()
    return None


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("$", "").replace(" ", "")
    text = re.sub(r"[.,;:]+$", "", text)
    return text


def answer_matches(prediction: object, reference: object) -> bool:
    if prediction is None or reference is None:
        return False
    try:
        from math_verify import parse, verify

        if verify(parse(str(reference)), parse(str(prediction))):
            return True
    except Exception:
        pass
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    return pred == ref


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None
