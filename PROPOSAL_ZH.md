# 研究 Proposal：The Synthetic Substitution Gap in Math RLVR

## 1. 暂定标题

**The Synthetic Substitution Gap in Math RLVR: Quality, Learnability, and Training Signal**

备选标题：**Can Synthetic Math Questions Replace Curated Prompts for RLVR?**

项目简称：**SynMath-RLVR**；建议仓库名：`Synthetic-Math-RLVR`。

## 2. 研究问题与动机

RLVR 需要大量具有可验证答案的数学题。现实中，高质量题目通常来自竞赛、教材和人工整理，获取、清洗与验题成本很高；大模型则可以低成本生成近乎无限的题目、解答与答案。一个直接但尚未被充分回答的生产问题是：

> 在训练数据量、领域和母题分布受控时，LLM 生成的数学题能否替代经过筛选的真实题目，用于 RLVR？

本工作不把“错误题目”预先作为训练 treatment，也不依赖大模型判断题目是否正确。核心 treatment 是训练题目的来源，即真实题目与自然生成题目之间的替换比例。人工审核只用于估计 synthetic question-answer pair 的真实可用率，并解释观察到的替换差距。

## 3. 核心假设

- **H1：Synthetic substitution gap。** 随 synthetic replacement ratio 上升，RLVR 的最终能力或训练效率会系统变化；即使题量、母题和领域分布保持一致，synthetic questions 也未必是 curated questions 的等价替代品。
- **H2：质量并非唯一机制。** 替换差距不仅来自无效题或错误 reference，也来自题目对当前 policy 的可学习性，以及它能否产生有方差的 on-policy reward。
- **H3：自动过滤可以缩小但未必消除差距。** Solver-consistency filter 配合 real fallback 能提高 synthetic pair 的可靠性，但可能偏向容易题，形成 quality–learnability trade-off。
- **H4：RLVR 比 SFT 更受训练信号稀疏影响。** SFT 对每个样本都有 token-level gradient；RLVR 只有在 rollout group 内出现有区分度的 reward 时才产生有效更新。因此，两者对同一批 synthetic data 的敏感机制不同。

## 4. 分析框架

对题目 \(x\)，令当前 policy 单次 rollout 获得正确 reward 的概率为 \(p_x\)，每题采样 \(K\) 个 rollout。二元 reward 下，group 内出现混合 reward、从而提供直接相对学习信号的概率为：

\[
S_K(p_x)=1-p_x^K-(1-p_x)^K.
\]

当题目过难、无效或 reference 错误时，\(p_x\) 往往接近 0；当题目过易时，\(p_x\) 接近 1。两者都会使 \(S_K(p_x)\) 下降。由此可将 synthetic data 的影响拆为三层：

1. **Quality：** 问题是否有效、答案是否唯一、reference 是否正确；
2. **Learnability：** base policy 在该题上的初始成功概率与难度分布；
3. **Training signal：** 实际训练中 mixed-reward group、reward variance、all-zero group 和每个有效更新所需 token。

该框架不声称“错误题目必然导致性能下降”，而是给出可检验预测：replacement ratio 改变 prompt-level \(p_x\) 分布，进而改变 RLVR 获得有效训练信号的频率。SFT 对 reward sparsity 不敏感，但仍会受错误 solution/reference 的监督偏差影响。

## 5. 数据与识别设计

### 5.1 Paired seed design

从 `nlile/NuminaMath-1.5-RL-Verifiable` 中选择 5,000 道非合成、答案可验证的真实题目，来源包括 `cn_contest`、`amc_aime`、`olympiads_ref`、`inequalities` 和 `number_theory`。按 `parent_seed_id` 固定划分：4,000 train、500 development、500 held-out audit。

GPT-5.4 针对每个 seed 自然生成一道领域和难度近似的新题，并输出 solution/reference answer。生成提示不要求制造错误。主实验中的每个 parent seed 始终只贡献一道题，因此 synthetic replacement 不改变训练题量或母题构成。

### 5.2 Replacement views

从同一组 4,000 个 train parent seeds 构造：

| 训练组 | Synthetic ratio | 题目数 |
|---|---:|---:|
| R0 | 0% | 4,000 |
| R25 | 25% | 4,000 |
| R50 | 50% | 4,000 |
| R75 | 75% | 4,000 |
| R100 | 100% | 4,000 |
| Filter + fallback | 由通过率决定 | 4,000 |

替换集合按 hash 确定并在比例间嵌套，在 source/problem type strata 内采样。Filter + fallback 仅在 synthetic pair 通过自动检查时使用合成题，否则退回对应真实题。

### 5.3 自动质量策略

自动策略只使用可部署检查，不用 LLM 的 `valid/invalid` 标签决定训练分组：字段完整、reward answer 可解析、无重复、无评测污染，并且 3 次独立 solver rollout 中至少 2 次匹配生成 reference。需要同时报告 filter retention、成本，以及过滤前后的难度偏移。

## 6. 人工审核

人工标注的唯一主要目标是估计 synthetic question-answer pairs 的正确率，并评估自动 filter，而不是发布新的错误题 benchmark。

建议审核 800 个 pairs：

- 300 个共享 held-out seeds 的 GPT 与 Claude 输出，共 600 pairs，用于无偏比较生成器质量；
- 100 个随机 filter-pass GPT pairs；
- 100 个 filter-rejected 或 solver-disagreement GPT pairs。

每题由 2 人独立审核；分歧、任一 invalid、低置信度样本及 10% 随机一致样本交给第 3 位专家。分别标注 question validity、unique answer、reference correctness 和 overall pair usability。预计平均 2.2–2.4 次审核/题，总工作量约 210–350 人时；先做 100 题 pilot 测量单题时间与一致性，再决定是否缩减或扩展。

人工标签不参与主训练集选择，因此论文只能将 invalidity/wrong-reference 作为机制证据，不能声称它们是训练结果变化的已识别因果来源。

## 7. 训练与评测

### 7.1 RLVR 主实验

- 主模型：`Qwen/Qwen2.5-1.5B` base；R0/R25/R50/R75/R100/fallback，各 3 seeds；
- 规模复现：`Qwen/Qwen2.5-7B` base；优先 R0/R100/fallback；
- 训练前：对所有 prompts 用对应 base model 做 K=8 rollout profiling，但不据此修改训练集；
- 训练中记录：prompt-level reward mean/variance、all-zero/all-one/mixed group 比例、rollout tokens、KL 与 checkpoint solve rate。

### 7.2 SFT 诊断对照

只训练 R0、R100 和 filter + fallback 三组，使用相同 parent seeds。SFT 用于检验 training-objective interaction，不扩展为第二条完整故事。如果 synthetic gap 仅在 RLVR 明显，支持 training-signal 机制；若 SFT 同样下降，则更可能是内容质量或分布差异。

### 7.3 Evaluation

使用 Qwen2.5-Math evaluation harness，覆盖 MATH、OlympiadBench、CollegeMath、Minerva Math、AIME24 和 AMC23。主结果同时报告最终准确率与训练效率，避免只比较单一 benchmark endpoint。

建议分阶段执行以控制 GPU 成本：

1. 1.5B 的 R0/R50/R100 单 seed pilot；
2. pilot 确认训练稳定后补齐完整 replacement curve 和 3 seeds；
3. 再运行 fallback、SFT 对照与 7B headline replication。

## 8. 主要结果与图表

1. **Replacement curve：** benchmark accuracy 与训练效率随 synthetic ratio 的变化；
2. **Training-signal decomposition：** quality、base learnability、mixed-reward probability 与最终收益之间的关系；
3. **Mitigation：** raw R100 与 filter + real fallback 的质量—保留率—性能—API 成本对比；
4. **Objective comparison：** 同一数据在 RLVR 与 SFT 下的相对差距；
5. **Robustness：** 1.5B/7B 和 GPT/Claude 数据质量复现。

统计上以 parent-seed paired bootstrap 和 training-seed variation 给出置信区间。replacement ratio 既作为离散 treatment，也用于检验单调趋势；机制分析明确标注为关联性分析。

## 9. 预期贡献

1. **问题贡献：** 将“synthetic math data 是否有效”收敛为一个直接面向生产的替换问题，而不是泛泛比较合成数据与真实数据。
2. **设计贡献：** 提出 paired, fixed-size, nested replacement protocol，在控制题量、母题和来源分布后测量 synthetic substitution gap。
3. **机制贡献：** 用 quality → learnability → usable on-policy signal 的链条解释 RLVR 结果，并通过 SFT 对照区分 reward sparsity 与内容质量。
4. **实践贡献：** 给出 synthetic replacement ratio、自动过滤、real fallback 与训练收益之间的质量—成本前沿。

本工作更适合定位为一篇具有分析框架的 **evaluation/empirical paper**，而不是新训练算法论文。其价值取决于控制实验是否严谨、replacement curve 是否稳定、以及机制测量能否解释 endpoint performance。

## 10. Novelty 风险与边界

最接近的工作通常分为三类：synthetic math data generation、noisy/incorrect supervision in RLVR、以及数学题质量或错误检测 benchmark。本工作的区分点应始终保持为：

- treatment 是自然 synthetic question substitution，而非人工注入 label noise 或 invalid questions；
- 使用相同 parent seeds 的受控 replacement curve，而非两个独立数据集的直接对比；
- 同时测量 question quality、base learnability 与 on-policy signal，而非只报告最终 accuracy；
- 人工审核只估计正确率和 filter 性能，不包装成新 benchmark。

主要风险是：替换差距可能主要由表面难度、文本长度或风格造成；GPT-5.4 生成器可能过强/过弱；solver filter 与训练 policy 的能力不匹配；仅用 Qwen2.5 家族会限制外推性。因此必须报告长度、答案形式、领域、难度、base solve rate，并至少做一个模型规模和一个生成器层面的复现。

## 11. Go / No-Go 判据

完成 1.5B pilot 后再决定是否投入完整训练：

- **Go：** R0/R50/R100 在 endpoint、训练效率或 signal statistics 上出现稳定、可解释的差异；或 R100 与 R0 接近，但能证明高比例 synthetic replacement 在显著降低数据成本时不损害训练。
- **Revise：** 只看到差异但完全由明显难度偏移解释，则加入 learnability-matched analysis，将主张收缩为“为什么 naive synthetic replacement 失败”。
- **No-Go：** 多 seed 下所有指标都无差异，且无法形成有意义的成本结论；或者数据污染/奖励解析问题使 paired comparison 无法成立。

“没有 substitution gap”本身不等于负结果：如果控制充分，它可支持“高质量合成题能够替代昂贵 curated prompts”的强实践结论。

## 12. 最小交付物

- 5,000 个 real–synthetic canonical pairs 与完整生成 provenance；
- 五档 RLVR views、filter + fallback view、三组 SFT views；
- human audit protocol 与聚合正确率，不发布 benchmark claim；
- prompt-level K=8 learnability 与训练 signal logs；
- replacement curve、机制分解和成本前沿；
- 可复现脚本，数据跑完后再封装为 DataFlow operators/pipeline。
