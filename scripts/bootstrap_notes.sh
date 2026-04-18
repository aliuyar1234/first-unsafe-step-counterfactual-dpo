#!/usr/bin/env bash
# Human/operator notes only. Replace with a real bootstrap script if useful.
set -euo pipefail

echo "1) Create and activate a virtual environment."
echo "2) pip install -e '.[dev,train]'"
echo "3) add ODCV-Bench as external dependency under external/ODCV-Bench"
echo "4) launch local model server for Qwen2.5-7B-Instruct"
echo "5) run one pilot scenario end-to-end"
