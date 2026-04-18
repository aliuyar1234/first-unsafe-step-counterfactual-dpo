# Drift guards

This project is vulnerable to a few predictable drift modes. Treat them as bugs.

## Drift mode 1: Scope expansion
Symptoms:
- adding new benchmarks
- adding more model families
- adding runtime defenses
- switching the paper type

Guard:
- if a change does not directly improve the narrow v1 claim, cut it
- log cuts in `state/decision_log.md`

## Drift mode 2: Prompt-only inflation
Symptoms:
- claiming safety gains without checking refusal inflation
- comparing DPO only against weak prompts

Guard:
- always run `contract_prompt` and `refusal_prompt`
- always report `RefusalRate`, `KPI-SR`, and `Safe-KPI-SR` alongside `HS-MR`

## Drift mode 3: Data leakage
Symptoms:
- generating pairs from val/test stems
- random trajectory split instead of stem split
- using future observations in rewrite prompts

Guard:
- add assertions in code
- validate every pair against split manifest and step cutoff

## Drift mode 4: Judge overtrust
Symptoms:
- letting a single AI judge decide threshold-critical cases
- changing judge backend midstream

Guard:
- freeze judge version
- audit all severity 2/3 and threshold-crossing cases
- record disagreement resolution

## Drift mode 5: Paper overclaiming
Symptoms:
- saying the method “solves” KPI-gaming
- claiming generalization beyond tested scope
- claiming superiority over runtime defenses

Guard:
- map every claim to `paper/claims_checklist.yaml`
- delete unsupported claims aggressively

## Drift mode 6: Writing before results
Symptoms:
- large paper sections drafted before metrics and audit exist
- abstract written as if results are already known

Guard:
- method/related work can be drafted early
- results/discussion only after frozen outputs exist

## Drift mode 7: Interface drift
Symptoms:
- training prompt format differs from eval prompt format
- parser format differs between data generation and evaluation

Guard:
- define and reuse one next-turn format
- create parser round-trip tests
