# Experiment plan: real-to-synthetic replacement

## Research question

Can naturally generated synthetic math questions replace curated real questions for RLVR without reducing capability, training efficiency, or reliability?

The treatment variable is question origin, not a human or LLM validity label. Human annotation is used only to estimate synthetic question-answer pair correctness and to evaluate the automatic quality filter.

## Data construction

Start from `nlile/NuminaMath-1.5-RL-Verifiable`. Retain non-synthetic, valid, exact-answer `math-word-problem` samples from:

- `olympiads_ref`
- `amc_aime`
- `cn_contest`
- `inequalities`
- `number_theory`

Remove duplicate and evaluation-contaminated questions. Deterministically sample 5,000 seeds:

| Source | Count |
|---|---:|
| cn_contest | 2,000 |
| amc_aime | 1,250 |
| olympiads_ref | 750 |
| inequalities | 500 |
| number_theory | 500 |

Split by parent seed before generation: 4,000 train, 500 development, and 500 held-out audit seeds.

Generate one question-solution-answer pair with GPT for every seed. Generate a second pair with Claude for the same 500 held-out seeds. Generation prompts request a new, self-contained problem of similar domain and difficulty; they never request intentional errors.

## Automatic quality policy

Do not use LLM `valid/invalid` predictions to define training groups. A synthetic pair passes only if:

- all required fields are complete;
- the reference answer is parseable by the training reward;
- it is not an exact or near duplicate;
- it does not overlap Qwen evaluation data;
- at least two of three independent solver rollouts match its reference answer.

For the production-style filtered arm, use the synthetic question when it passes; otherwise fall back to its paired real question. This keeps question count, parent seeds, and source distribution fixed.

## RLVR matrix

Construct five 4,000-question views from the paired train split:

| Arm | Synthetic replacement rate |
|---|---:|
| R0 | 0% |
| R25 | 25% |
| R50 | 50% |
| R75 | 75% |
| R100 | 100% |

Replacement is nested and deterministic: if a parent seed is synthetic at 25%, it remains synthetic at all higher ratios. Select replacements within source and problem-type strata so all views preserve the same distribution.

Add one `quality-filtered + real fallback` arm and report its realized synthetic rate.

Main model: `Qwen/Qwen2.5-1.5B` base. Run the complete ratio curve with three seeds. Replication model: `Qwen/Qwen2.5-7B` base. Run only R0, R100, and fallback on the replication model.

## SFT control

SFT is a diagnostic control, not a second full story. On Qwen2.5-1.5B, train only:

- R0: all real;
- R100: all raw synthetic;
- quality-filtered + real fallback.

Use the same parent seeds and question count as RLVR. Do not run the full replacement curve or the 7B SFT replication unless the pilot shows a clear training-objective interaction.

## Human correctness audit

Audit 800 synthetic question-answer pairs:

- 300 shared seeds x GPT and Claude generations = 600 pairs;
- 100 additional GPT pairs that pass the automatic policy;
- 100 additional GPT pairs that fail or have solver disagreement.

Every pair receives two independent reviews. A third expert adjudicates every disagreement, every invalid label, every low-confidence case, and a random 10% of agreed cases. Expected average: 2.2-2.4 reviews per pair.

Annotators separately label:

- question validity;
- whether a unique target answer exists;
- reference-answer correctness;
- pair usability: all three conditions hold;
- invalidity category and confidence.

These labels never determine training membership. Report unbiased GPT/Claude correctness from the random paired sample; use the pass/fail-enriched sample only to estimate filter precision and recall with sampling weights.

Expected workload is approximately 210-350 person-hours. First annotate 100 questions to measure actual time and agreement before scheduling the remaining 700.

## Evaluation and analysis

Evaluate every checkpoint with the local Qwen2.5-Math harness on MATH, OlympiadBench, CollegeMath, Minerva Math, AIME24, and AMC23.

Primary outcomes:

- benchmark accuracy versus synthetic replacement rate;
- reward and solve rate during training;
- all-zero and mixed-reward group frequency;
- rollout tokens per rewarded group;
- fallback policy quality-cost tradeoff.

Data-quality outcomes:

- GPT and Claude pair-usability rates;
- invalid-question and wrong-reference rates;
- automatic filter precision, recall, retention, and cost;
- performance associations with the audited error rate.

The audit supports mechanism interpretation only. The paper must not claim that invalidity causally explains a training effect unless a later experiment explicitly constructs a validity treatment.

## Acceptance criteria

- Exactly 5,000 complete GPT pairs and 500 paired Claude generations.
- Train/development/held-out parent IDs are disjoint.
- Every RLVR arm contains exactly 4,000 unique parent seeds.
- Source and problem-type counts match across replacement arms.
- No high-similarity overlap with evaluation datasets.
- Every reference answer passes the actual reward parser.
- API generation and solving are resumable and preserve failed attempts.
- A 100-row end-to-end smoke test passes generation, filtering, verl reward, and Qwen evaluation before full API generation.

On-policy distillation, invalid-ratio training, abstention rewards, and a new correctness benchmark are out of scope.
