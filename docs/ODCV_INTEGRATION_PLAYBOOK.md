# ODCV integration playbook

This is the highest-drift implementation surface in the repo. Follow this file before modifying the benchmark.

## Goal
Connect the official ODCV-Bench harness to a **local OpenAI-compatible model backend** while changing as little benchmark logic as possible.

## Non-goals
- Do not rewrite mission executor logic.
- Do not redesign scenario environments.
- Do not replace benchmark scoring semantics.
- Do not turn this into a custom agent framework project.

## Exact procedure
1. Clone or pin the official benchmark under `external/ODCV-Bench`.
2. Locate the benchmark's current model-calling boundary (for example: OpenRouter client, OpenAI client, provider abstraction, or request wrapper).
3. Implement a thin local client in `src/agents/local_openai_client.py` that exposes the same minimal request/response shape needed by the benchmark.
4. Patch the benchmark at the **provider boundary only** so requests go to `http://localhost:8000/v1` with API key `local`.
5. Keep raw benchmark prompts and responses intact; add logging around them rather than rewriting them.
6. Preserve the benchmark's own action loop, environment state, and evaluation logic.
7. Add one pilot runner in `src/agents/run_odcv_local.py` for a single stem/variant/seed.
8. After integration, verify that one scenario completes end-to-end and that trajectory logs validate against `schemas/trajectory_step.schema.json`.

## Acceptance test
- One scenario runs end-to-end against the local backend.
- The benchmark commit hash is pinned in a manifest.
- No benchmark behavior changes beyond provider/backend integration and additive logging.
- The repo can still reproduce the same run after restarting the local model server.

## If the repo structure differs from expectation
- Log the actual integration path in `state/decision_log.md`.
- Keep the interface contracts in this repo stable even if the benchmark internals differ.
- Prefer adapter code in this repo over invasive edits inside `external/ODCV-Bench`.
