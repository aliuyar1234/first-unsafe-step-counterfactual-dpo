# CSV output contracts

These are the exact paper-facing CSV outputs the implementation must generate. Keep names and columns stable.

## 1) `results/metrics/pair_stats.csv`
One row per `(split, scenario_stem, variant)` after pair generation.

Required columns:
- `split`
- `scenario_stem`
- `variant`
- `num_runs_sampled`
- `num_runs_severity_ge_3`
- `num_raw_rewrite_candidates`
- `num_kept_rewrites`
- `num_pairs_kept`
- `refusal_only_kept_fraction`

## 2) `results/metrics/val_checkpoint_metrics.csv`
One row per evaluated checkpoint.

Required columns:
- `checkpoint_id`
- `method`
- `global_step`
- `epoch`
- `val_pair_loss`
- `val_hs_mr`
- `val_safe_kpi_sr`
- `val_kpi_sr`
- `val_refusal_rate`
- `selected`
- `reject_reason`

## 3) `results/metrics/test_runs.csv`
One row per held-out run after final label resolution.

Required columns:
- `run_id`
- `scenario_stem`
- `domain`
- `variant`
- `split`
- `method`
- `seed`
- `max_severity`
- `primary_violation_type`
- `first_unsafe_step`
- `kpi_success`
- `refusal`
- `ambiguous`
- `audited`
- `final_label_source`

## 4) `results/metrics/test_summary.csv`
One row per method on the held-out split.

Required columns:
- `method`
- `split`
- `n_runs`
- `hs_mr`
- `avg_severity`
- `kpi_sr`
- `safe_kpi_sr`
- `refusal_rate`
- `hs_mr_mandated`
- `hs_mr_incentivized`
- `psg`
- `ci_low_hs_mr`
- `ci_high_hs_mr`
- `delta_hs_mr_vs_best_prompt`
- `delta_safe_kpi_sr_vs_best_prompt`

## 5) `results/metrics/bootstrap_ci.csv`
One row per `(method, metric)` CI.

Required columns:
- `method`
- `metric`
- `ci_low`
- `ci_high`
- `resamples`
- `bootstrap_seed`

## 6) `results/metrics/per_stem_results.csv`
One row per `(scenario_stem, method)` on held-out evaluation.

Required columns:
- `scenario_stem`
- `domain`
- `method`
- `hs_mr`
- `kpi_sr`
- `safe_kpi_sr`
- `avg_severity`
- `hs_mr_mandated`
- `hs_mr_incentivized`

## 7) `results/audits/manual_audit.csv`
One row per manually audited run after unblinding.

Required columns:
- `run_id`
- `scenario_stem`
- `variant`
- `blinded_id`
- `resolved_method`
- `judge_max_severity`
- `audit_max_severity`
- `judge_first_unsafe_step`
- `audit_first_unsafe_step`
- `kpi_success_confirmed`
- `primary_violation_type`
- `decision`
- `threshold_crossing_case`
- `auditor_note`

## Stability rule
The paper assets and claims checklist assume these file paths and columns. If a new derived CSV is useful, add it, but do not rename or silently change these contracts.
