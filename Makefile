.PHONY: help bootstrap split contracts pilot judge pairs train-dpo train-sft eval metrics paper test

help:
	@echo "Targets:"
	@echo "  bootstrap   - environment + external benchmark dependency"
	@echo "  split       - build frozen scenario-level split"
	@echo "  contracts   - extract and validate contracts"
	@echo "  pilot       - run pilot base vs contract prompt"
	@echo "  judge       - score trajectories and localize first unsafe steps"
	@echo "  pairs       - build preference pairs"
	@echo "  train-dpo   - run main LoRA-DPO"
	@echo "  train-sft   - run chosen-only SFT ablation"
	@echo "  eval        - run held-out evaluation"
	@echo "  metrics     - aggregate metrics + CIs + tables"
	@echo "  paper       - compile paper assets"
	@echo "  test        - validate schemas/examples"

bootstrap:
	@echo "Implement in scripts or Python CLI once modules exist."

split:
	python -m src.splits.build_split --config configs/split_policy.yaml

contracts:
	python -m src.contracts.extract_contracts --config configs/defaults.yaml

pilot:
	python -m src.eval.run_eval --config configs/eval_pilot.yaml

judge:
	python -m src.judging.judge_trajectory --config configs/eval_main.yaml

pairs:
	python -m src.counterfactuals.build_pairs --config configs/defaults.yaml

train-dpo:
	python -m src.training.train_dpo --config configs/train_dpo_main.yaml

train-sft:
	python -m src.training.train_sft --config configs/train_sft_ablation.yaml

eval:
	python -m src.eval.run_eval --config configs/eval_main.yaml

metrics:
	python -m src.eval.aggregate_metrics --config configs/eval_main.yaml

paper:
	@echo "Generate tables/figures then compile LaTeX."

test:
	pytest -q
