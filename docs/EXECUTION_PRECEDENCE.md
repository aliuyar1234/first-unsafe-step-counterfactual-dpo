# Execution Precedence

This file exists to prevent instruction drift.

## Precedence order

If any instructions conflict, follow this order:

1. `docs/SSOT.md`
2. `docs/TASK_GRAPH.yaml`
3. other files in `docs/`
4. files in `configs/`
5. files in `schemas/`
6. code docstrings and tests

## Interpretation rule

When in doubt:
- prefer the **narrower scope**
- prefer the **more reproducible protocol**
- prefer the **simpler implementation**
- prefer the **path that keeps the paper publishable if DPO gains are modest**

## Mandatory log rule

Every nontrivial deviation, assumption, or uncertainty must be recorded in:
- `state/decision_log.md`

## Forbidden drift

The following are not allowed unless explicitly required by the SSOT:
- adding a second main model
- turning the paper into a benchmark extension paper
- turning the paper into a runtime defense paper
- widening to unrelated agent safety topics
- adding new datasets as core contributions
- replacing the held-out stem split with trajectory-level random splitting
