# Human correctness audit

## Purpose

The audit estimates synthetic question-answer pair correctness and evaluates the automatic quality policy. Human labels do not determine training membership and are not presented as a new benchmark.

## Unit of annotation

Annotators first judge the question without reading the proposed solution when possible, then separately inspect the reference answer and solution.

Record:

- `question_valid`: conditions are coherent, sufficient, and feasible;
- `unique_answer_exists`: the requested target has a unique answer;
- `reference_answer_correct`: the supplied final answer is mathematically correct;
- `pair_usable`: all three fields above are true;
- `invalid_type`: `valid`, `underdetermined`, `contradictory`, `infeasible`, `ambiguous_expression`, or `other_invalid`;
- confidence, elapsed time, and short notes.

A plausible supplied answer does not make an invalid question valid.

## Review protocol

Every item receives two independent reviews. Send an item to a third expert when:

- the two reviewers disagree on any primary field;
- either reviewer marks it invalid;
- either reviewer reports low confidence or `requires_specialist`;
- it belongs to the random 10% quality-control sample of agreed items.

Use the adjudicated label only for aggregate correctness estimates and filter evaluation. Report raw agreement, Cohen's kappa, adjudication rate, time per item, and unresolved specialist cases.

## Planned sample

- 300 shared seeds with GPT and Claude outputs: 600 pairs;
- 100 GPT outputs that pass the automatic quality policy;
- 100 GPT outputs that fail or have solver disagreement.

The paired 600 estimate generator correctness. The enriched 200 estimate filter precision and recall with appropriate sampling weights. First run a 100-item timing and calibration pilot before assigning the remaining 700.
