# State discipline for long-horizon implementation runs

Long-running implementation sessions can work for long periods, but context and drift become real risks. Use these files as durable memory:

- `state/current_state.md` — latest truthful project state
- `state/decision_log.md` — all important assumptions and deviations
- `state/blockers.md` — only real blockers

## Update cadence
Update `state/current_state.md`:
- after each completed phase
- before any long training run
- before and after evaluation
- before ending a session
- before any manual compaction or handoff

## Content rule
Keep entries short, factual, and path-based:
- what phase is active
- what was finished
- what remains next
- where the evidence files live
