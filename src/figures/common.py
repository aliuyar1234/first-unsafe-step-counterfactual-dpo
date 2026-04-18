from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fmt_pct(value: float | int) -> str:
    return f"{100 * float(value):.1f}\\%"


def fmt_num(value: float | int, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def tex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )
