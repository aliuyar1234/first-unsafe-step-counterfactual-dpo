# Data generation protocol

## Sampling policy
For each train stem and each variant:
- generate 8 trajectories initially
- temperature = 0.6
- if zero severity>=3 runs appear, stop
- if at least one severity>=3 run appears but fewer than 2 accepted pairs are produced, extend to at most 12 trajectories

## Empirical note for this paper
- On the repaired real train sweep, the pair-count-based `8 -> 12` extension trigger proved effectively non-operative because one localized unsafe run already tends to saturate the accepted-pair threshold under the top-2 keep rule.
- For the current paper, the repaired train corpus is frozen rather than re-opened with a new extension policy.
- Any future follow-up that changes the extension logic is a new protocol, not part of this submission.

## Required metadata per run
- split
- stem
- variant
- method
- seed
- model ID
- adapter ID
- benchmark commit
- repo commit
- prompt versions
- decoding settings
- wallclock
- raw outputs
- parsed actions

## First unsafe step localization
1. full-trajectory judge
2. candidate unsafe step
3. local step audit
4. previous-step check if needed
5. manual resolution for threshold-critical cases

## Rewrite policy
Generate:
- minimal_safe_progress
- safe_alternative_plan
- explicit_infeasibility_or_escalation

Keep only candidates that are:
- parseable
- safe or acceptable under step auditor
- consistent with visible history
- not future-leaking
