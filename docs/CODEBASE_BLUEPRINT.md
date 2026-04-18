# Codebase blueprint

## Recommended tree

```text
src/
  agents/
    local_openai_client.py
    parse_agent_output.py
    run_odcv_local.py
  contracts/
    extract_contracts.py
  splits/
    build_split.py
  judging/
    judge_trajectory.py
    localize_first_unsafe.py
  counterfactuals/
    rewrite_first_unsafe.py
    build_pairs.py
  training/
    train_dpo.py
    train_sft.py
  eval/
    run_eval.py
    aggregate_metrics.py
    bootstrap_ci.py
    manual_audit.py
  figures/
    make_main_results_table.py
    make_ablation_table.py
    make_error_taxonomy_table.py
    make_pressure_figure.py
    make_tradeoff_figure.py
    make_qualitative_figure.py
  utils/
    manifests.py
    schemas.py
tests/
  test_schema_examples.py
  test_handoff_consistency.py
```

## Implementation order
1. `src/agents/` backend integration and one pilot benchmark run
2. `src/utils/` foundation and manifests
3. split builder
4. contract extractor
5. trajectory logging and judge
6. localization
7. rewrite generator
8. pair builder
9. DPO trainer
10. eval runner and bootstrap/manual audit
11. paper asset generation

## Design rules
- Use deterministic CLI entrypoints.
- Keep schemas separate from code logic.
- Make raw -> processed -> results flow explicit.
- Validate every artifact immediately after writing it.
- Put file hashes and config snapshots in manifests.
- Keep benchmark integration limited to the provider/backend boundary.
