"""Run the chosen-only SFT ablation on the same pair data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.train_dpo import _run_training


def train_sft(config_path: Path) -> Path:
    """Train chosen-only SFT and return the selected checkpoint path."""
    return _run_training(config_path, default_method="sft_chosen", trainer_kind="sft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train_sft(Path(args.config)))
