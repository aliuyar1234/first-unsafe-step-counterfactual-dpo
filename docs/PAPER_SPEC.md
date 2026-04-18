# Final paper spec

## Working title
First-Unsafe-Step Counterfactual DPO for KPI-Gaming in Autonomous LLM Agents: A Held-Out Evaluation and Sparse-Data Limit Study

## One-paragraph abstract target
Outcome-driven constraint violations occur when autonomous LLM agents pursue operational metrics by exploiting loopholes, suppressing evidence, or falsifying records during multi-step tasks. We present a reproducible first-unsafe-step pipeline for KPI-gaming on ODCV-Bench using a single open-weight 7B model, Qwen2.5-7B-Instruct. Starting from a scenario-level held-out split, we collect real benchmark-backed trajectories, localize the earliest high-severity unsafe action, and rewrite only that next step into a minimal safe alternative. The central empirical finding is not a strong DPO win but a data bottleneck: a complete real train sweep yields only sparse unsafe localized support, which limits localized post-training. We therefore frame the work as a rigorous held-out ODCV evaluation plus a sparse-data limit study, with bounded SFT and DPO pilots interpreted as probes rather than rescue attempts.

## Claimed contributions (fixed)
1. A scenario-level held-out split for ODCV-Bench suitable for post-training research.
2. A first-unsafe-step localization pipeline for converting unsafe trajectories into preference data.
3. A localized counterfactual DPO/SFT pilot setup for KPI-driven agent safety under sparse support.
4. A controlled comparison against strong prompt-only baselines on held-out ODCV scenarios.
5. A reviewer-proof evaluation protocol with severity rubric, pressure breakdown, bootstrap uncertainty, and manual audit routing.

## Required takeaway
In KPI-gaming settings, the important unit is the **action step**, not the **refusal sentence**.

## Convincing result pattern
- the held-out protocol is rigorous and reproducible
- prompt-only safety text does not rescue the benchmark
- the train-side unsafe support is sparse and concentrated
- bounded localized post-training is weak, mixed, or null under that sparse regime
- all threshold-critical claims survive manual audit

## Weak result pattern
- any apparent safety gains come mostly from refusal or KPI collapse
- `fus_dpo` does not beat the best prompt-only baseline on the joint safety--utility view
- improvements are limited to ambiguous threshold cases
