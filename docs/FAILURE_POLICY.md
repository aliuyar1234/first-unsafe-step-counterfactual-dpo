# Failure policy

This file specifies what to do when things go wrong. Follow it instead of improvising.

## 1. If ODCV integration is harder than expected
Do not redesign the benchmark.
Do:
- inspect the actual benchmark run path,
- patch the model backend boundary only,
- keep mission executor behavior conceptually unchanged,
- record actual integration points in `state/decision_log.md`.

## 2. If you cannot get a reliable judge immediately
Do not stop the project.
Do:
- keep the judge interface fixed,
- choose one frozen backend,
- validate it on two hand-made trajectories,
- manually audit threshold-critical cases,
- record the exact model/backend ID in manifests.

## 3. If training data is sparse
Do:
- increase samples from 8 to 12 only for productive train stem-variants,
- keep filters strict,
- prefer fewer high-quality pairs over many noisy pairs,
- if the repaired full train sweep still yields materially sparse support, freeze the corpus and pivot to the evaluation-first paper framing.

Do not:
- touch val/test,
- relax severity threshold below 3 for rejected examples,
- inflate the rewrite count to fake pair scale,
- change the core data unit after the sparse-data pivot is accepted.

## 4. If DPO training is unstable
Try in this order:
1. reduce sequence length to 4096
2. switch to 4-bit QLoRA pilot
3. run chosen-only SFT warm start
4. run shorter DPO with the same pairs
5. if still weak, preserve the project as an evaluation-heavy paper

## 5. If DPO gains are modest
Do not widen the paper.
Do:
- emphasize held-out discipline,
- compare against strong prompt baselines,
- report the pair-generation pipeline,
- write the negative/modest result honestly,
- show where the method failed.
- keep one bounded `sft_chosen` and one bounded `fus_dpo` pilot only; do not start hyperparameter fishing.

## 6. If the benchmark appears too easy or too hard
Do:
- verify the harness on pilot stems,
- check that base has at least some severity>=3 behavior,
- check that contract prompting changes behavior,
- verify parser correctness and judge sanity.

Do not:
- invent a benchmark extension as a rescue move.

## 7. If time runs short
Cut in this order:
1. `dpo_no_contract`
2. paraphrase robustness appendix
3. extra qualitative examples
4. extra judge sensitivity analysis

Do not cut:
- stem-level split
- prompt-only baselines
- manual audit of threshold-critical cases
- main DPO run
- main result table
