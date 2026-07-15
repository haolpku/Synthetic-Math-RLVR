# Paper story

## Working title

**Can Synthetic Math Questions Replace Curated Prompts for RLVR?**

## Core question

RLVR needs large numbers of questions with automatically checkable answers. Curated questions are expensive, while LLMs can generate new question-answer pairs cheaply. The paper tests whether synthetic questions are a drop-in replacement for curated RLVR prompts.

## Identification strategy

Each trusted NuminaMath question is paired with one naturally generated question of similar domain and difficulty. Training set size and parent-seed distribution stay fixed while the synthetic replacement rate changes from 0% to 100%. This paired replacement design is the main methodological contribution.

The paper does not classify training questions as valid or invalid. A separate human sample estimates synthetic pair correctness and checks whether automatic solver-consistency filtering improves quality. Human labels are explanatory measurements, not training treatments.

## Claims the experiments can support

1. The capability and training-efficiency response to increasing synthetic-question share.
2. Whether a cheap automatic quality policy plus real-question fallback makes synthetic replacement safer.
3. Whether results replicate across GPT/Claude data quality and Qwen2.5 model scales.
4. Whether observed training differences are associated with human-estimated invalid-question or wrong-reference rates.

## Claims to avoid

- Invalid questions causally harm RLVR.
- LLM judges provide ground-truth question validity.
- The human audit is a new benchmark comparable to MathDebugger.
- Qwen2.5-Math-specific behavior generalizes to ordinary base models.

## Main figures

1. Accuracy and training efficiency versus 0/25/50/75/100% synthetic replacement.
2. Raw synthetic versus quality-filtered real-fallback policy.
3. GPT versus Claude human-estimated pair correctness.
4. Qwen2.5-1.5B main curve and Qwen2.5-7B headline replication.

## Contribution

The paper is a controlled production study of synthetic question substitution in RLVR, not another question-error benchmark. Its practical output is a measured quality-cost frontier: how much curated math data can be replaced, and what automatic fallback policy is needed, before model quality or rollout efficiency degrades.
