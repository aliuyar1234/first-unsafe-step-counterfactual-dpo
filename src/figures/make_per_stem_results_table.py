"""Generate a compact per-stem LaTeX appendix table from held-out results."""

from __future__ import annotations

from pathlib import Path

from src.figures.common import ensure_parent, fmt_num, fmt_pct, load_csv, tex_escape


def main(csv_path: Path, out_path: Path) -> None:
    df = load_csv(csv_path).sort_values(["scenario_stem", "method"])
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Stem & Method & HS-MR & KPI-SR & Safe-KPI-SR & AvgSeverity \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    tex_escape(row["scenario_stem"]),
                    tex_escape(row["method"]),
                    fmt_pct(row["hs_mr"]),
                    fmt_pct(row["kpi_sr"]),
                    fmt_pct(row["safe_kpi_sr"]),
                    fmt_num(row["avg_severity"], 2),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    ensure_parent(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
