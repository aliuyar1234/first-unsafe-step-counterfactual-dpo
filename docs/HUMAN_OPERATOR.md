# Human operator notes

## What not to do as the human operator

- Do not give the implementation agent a new topic.
- Do not ask it to brainstorm alternatives mid-run.
- Do not override the stem split after results start coming in.
- Do not ask it to "just write the paper first".
- Do not manually cherry-pick stems into train after seeing failures.

## What you *should* inspect

- `state/current_state.md`
- `state/decision_log.md`
- `state/blockers.md`
- generated manifests in `manifests/`
- test summary CSVs in `results/metrics/`
