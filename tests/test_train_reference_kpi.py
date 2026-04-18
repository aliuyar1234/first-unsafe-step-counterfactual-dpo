from __future__ import annotations

from src.training.train_dpo import _fallback_contract_prompt_reference_kpi


def test_fallback_contract_prompt_reference_kpi_uses_configured_value() -> None:
    assert _fallback_contract_prompt_reference_kpi({"checkpoint_selection": {"fallback_contract_prompt_kpi_sr": 0.73}}) == 0.73


def test_fallback_contract_prompt_reference_kpi_defaults_without_test_leakage() -> None:
    assert _fallback_contract_prompt_reference_kpi({}) == 0.60
