# Paper assets to generate

## Main table
Title: Held-out ODCV-Bench results: safety--utility comparison  
Data: `results/metrics/test_summary.csv`  
Message: `fus_dpo` improves HS-MR relative to prompt-only baselines without winning only by refusing

## Ablation table
Title: Ablations of localized counterfactual training  
Data: `results/metrics/test_summary.csv`  
Message: DPO and contract conditioning matter

## Error taxonomy table
Title: Violation types by method  
Data: `results/metrics/test_runs.csv`  
Message: where the method helps most

## Pressure sensitivity figure
Title: Mandated vs incentivized high-severity misalignment by method  
Data: `results/metrics/test_summary.csv`  
Message: whether the method reduces proactive KPI-driven violations

## Safety-utility tradeoff figure
Title: Safety--utility frontier on held-out ODCV-Bench  
Data: `results/metrics/test_summary.csv`  
Message: whether `fus_dpo` dominates `contract_prompt` instead of simply refusing more

## Qualitative trajectory figure
Title: One held-out trajectory before and after localized post-training  
Data: selected run artifacts + audit notes  
Message: the improvement happens at the action step, not only in surface refusal text
