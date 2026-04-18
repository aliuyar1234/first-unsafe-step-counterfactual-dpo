# Near-submission paper outline

## Title
First-Unsafe-Step Counterfactual DPO for KPI-Gaming in Autonomous LLM Agents

## Abstract
Use the target abstract from `docs/PAPER_SPEC.md` and replace generic phrases with actual result numbers only after held-out evaluation and audit are complete.

## Introduction (paragraph plan)

### P1: problem and stakes
- Define ODCV / KPI-gaming in autonomous agents.
- Motivate why benign operational goals can still induce unsafe action sequences.
- Cite ODCV-Bench.

### P2: why prompt-level refusal is not enough
- Explain the distinction between text refusal and action safety.
- Cite Mind the GAP.

### P3: gap in current methods
- Position against broader runtime guardrails and generic multi-step safety training.
- Explain the absence of a narrow ODCV-specific held-out lightweight post-training study.

### P4: paper idea
- Present first-unsafe-step localization and localized counterfactual DPO.
- Explain why next-turn supervision is the right granularity.

### P5: contributions and result summary
- list the 5 contributions
- summarize the main held-out result pattern without overclaiming

## Related work
Cover only:
- ODCV / KPI-gaming benchmarks
- text-vs-tool safety
- DPO for agentic safety
- runtime enforcement and thought correction as adjacent but different
- rigorous agentic benchmark methodology

## Method
1. Benchmark setup and notation
2. Stem-level split
3. Contract extraction
4. Trajectory judging and first-unsafe-step localization
5. Counterfactual rewrite generation
6. Pair construction
7. LoRA-DPO training
8. Inference conditions

## Experimental setup
- benchmark and split
- model and configs
- baselines and ablations
- metrics and uncertainty
- manual audit protocol
- compute budget

## Results
- main result table
- mandated vs incentivized breakdown
- ablation table
- safety-utility tradeoff
- qualitative trajectory

## Discussion
- what improved
- where the method failed
- whether the gain is merely refusal
- why the held-out discipline matters

## Limitations
- one model
- one benchmark
- AI judge dependence
- no runtime guarantees
- no broad generalization claim

## Ethics
- sandbox-only
- release caution for unsafe traces
- goal is safer autonomous systems
