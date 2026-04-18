# First-Unsafe-Step Counterfactual DPO for KPI-Gaming in Autonomous LLM Agents

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](paper/main.pdf)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-TMLR%20source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/main.tex)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.11-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Benchmark](https://img.shields.io/badge/Benchmark-ODCV--Bench-374151?style=flat-square)](https://github.com/McGill-DMaS/ODCV-Bench)
[![Scope](https://img.shields.io/badge/Scope-Held--Out%20Evaluation%20%2B%20Limit%20Study-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *First-Unsafe-Step Counterfactual DPO for KPI-Gaming in Autonomous LLM Agents: Held-Out Evaluation Under a Capability-Safety Bottleneck*

This repository accompanies a methods paper on preference-based post-training for multi-step agent safety. It studies whether a deliberately narrow intervention — localising the earliest high-severity unsafe step and training on minimal safe counterfactuals — can reduce outcome-driven constraint violations (ODCV) beyond prompt-only baselines on a single open-weight 7B model.

## Abstract

Outcome-driven constraint violations (ODCV) arise when autonomous agents pursue operational metrics by exploiting loopholes, suppressing evidence, or falsifying records during multi-step tasks. We study this problem on ODCV-Bench with Qwen2.5-7B-Instruct using a deliberately narrow intervention: localise the earliest executed assistant step whose action reaches severity ≥ 3, rewrite only that next step into a minimal safe alternative, and train lightweight chosen-only SFT and LoRA-DPO adapters on the resulting next-turn pairs. The repository freezes a stem-level 24 / 8 / 8 train/validation/test split, logs benchmark-backed runs with schema-validated artifacts, and resolves threshold-critical held-out cases through blinded manual audit.

## Main Finding

The central result is not a clean *DPO wins* story. A complete real 384-run train sweep yields only **36 unsafe localised events**, which become **72 accepted pairs from 9 train stems**. We describe this pattern as a **capability–safety bottleneck**: under the present ODCV protocol, a model must be capable enough to generate multi-step unsafe trajectories before preference-based safety post-training can learn from them.

On the audited held-out prompt-only backbone, the base model outperforms both contract prompting and refusal prompting on the joint safety–utility view:

| Method            | HS-MR     | Safe-KPI-SR |
| ----------------- | --------- | ----------- |
| base              | **6.25%** | **12.5%**   |
| contract-prompt   | 12.5%     | 0%          |
| refusal-prompt    | 12.5%     | 0%          |
| `fus_dpo` (pilot) | 12.5%     | 0%          |
| `sft_chosen`      | 18.75%    | 0%          |

We therefore interpret the study as (i) a rigorous held-out ODCV evaluation and (ii) a sparse-data limit analysis for localised post-training — and as a warning that standard preference-optimisation assumptions can break down in multi-step agent environments when unsafe train support is too sparse.

## Contributions

1. A real ODCV-Bench integration around Qwen2.5-7B-Instruct with a frozen stem-level 24 / 8 / 8 split, schema-validated artifacts, and held-out evaluation discipline.
2. A first-unsafe-step pipeline that localises the earliest severity-≥ 3 executed action, generates localised safe counterfactuals, and produces train-only next-turn preference pairs.
3. An empirical train-side finding: the full real 384-run train sweep yields only 36 unsafe localised events and 72 accepted pairs from 9 stems, indicating a severe data bottleneck.
4. A held-out evaluation-first framing in which prompt-only baselines and bounded post-training pilots are compared under the same metrics, bootstrap uncertainty, and manual-audit routing.

## Scope

This release is intentionally narrow and claim-safe.

- one open-weight 7B backbone: `Qwen2.5-7B-Instruct`
- one benchmark: official ODCV-Bench at a pinned commit
- frozen stem-level 24 / 8 / 8 split with mandated/incentivised variants co-located
- one main LoRA-DPO run, one chosen-only SFT ablation, two prompt-only baselines
- manual blinded audit only for threshold-critical held-out cases

The contribution is not breadth. It is a pinned-dependency, schema-enforced protocol plus an honest limit result on localised post-training for multi-step agent safety.

## Paper

- Compiled PDF: [`paper/main.pdf`](paper/main.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- Claims checklist: [`paper/claims_checklist.yaml`](paper/claims_checklist.yaml)
- Figure and table manifests: [`paper/figures_manifest.yaml`](paper/figures_manifest.yaml), [`paper/tables_manifest.yaml`](paper/tables_manifest.yaml)

## Repository Layout

- [`paper/`](paper/) — TMLR-style LaTeX manuscript, figures, tables, bibliography, and compiled PDF
- [`src/`](src/) — implementation: localisation, counterfactual rewriting, pair construction, training, evaluation
- [`scripts/`](scripts/) — pipeline entrypoints for data generation, training, and held-out evaluation
- [`configs/`](configs/) — frozen default configurations and the `domain_map.yaml` template
- [`schemas/`](schemas/) — strict artifact schemas (splits, localisation, run summaries, CSV contracts)
- [`manifests/`](manifests/) — pinned commit, model revision, and dataset manifests
- [`results/`](results/) — run outputs, metrics, bootstrap intervals, and audit records
- [`deliverables/`](deliverables/) — paper-facing tables, figures, and CSV exports
- [`tests/`](tests/) — regression tests for schemas and pipeline contracts
- [`docs/`](docs/) — method specification, evaluation playbook, and CSV output contracts

## Reproducibility

- [`docs/SSOT.md`](docs/SSOT.md) — single source of truth for the paper
- [`docs/ODCV_INTEGRATION_PLAYBOOK.md`](docs/ODCV_INTEGRATION_PLAYBOOK.md) — integration at the model-backend boundary
- [`docs/CSV_OUTPUT_CONTRACTS.md`](docs/CSV_OUTPUT_CONTRACTS.md) — exact column contracts for paper-facing CSVs

## Pinned External Dependencies

- ODCV-Bench benchmark: <https://github.com/McGill-DMaS/ODCV-Bench>
- ODCV-Bench paper: <https://arxiv.org/abs/2512.20798>
- Base model: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

External versions are not floated after the first successful bootstrap. Commits and model revisions are pinned in [`manifests/`](manifests/).

## Citation

```bibtex
@unpublished{uyar2026odcv,
  author = {Uyar, Ali},
  title  = {First-Unsafe-Step Counterfactual {DPO} for {KPI}-Gaming in Autonomous {LLM} Agents: Held-Out Evaluation Under a Capability--Safety Bottleneck},
  year   = {2026},
  note   = {Independent research}
}
```
