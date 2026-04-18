"""Generate the main LaTeX table from `results/metrics/test_summary.csv`.

Output path:
- `paper/tables/main_results.tex`
"""
from __future__ import annotations

from pathlib import Path

from src.figures.common import ensure_parent, fmt_num, fmt_pct, load_csv, tex_escape


def main(csv_path: Path, out_path: Path) -> None:
    df = load_csv(csv_path).sort_values(["method"])
    columns = [
        ("method", "Method"),
        ("hs_mr", "HS-MR"),
        ("avg_severity", "AvgSeverity"),
        ("kpi_sr", "KPI-SR"),
        ("safe_kpi_sr", "Safe-KPI-SR"),
        ("refusal_rate", "RefusalRate"),
        ("hs_mr_mandated", "HS-MR Mandated"),
        ("hs_mr_incentivized", "HS-MR Incentivized"),
    ]
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        " & ".join(label for _, label in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        rendered = [
            tex_escape(row["method"]),
            fmt_pct(row["hs_mr"]),
            fmt_num(row["avg_severity"], 2),
            fmt_pct(row["kpi_sr"]),
            fmt_pct(row["safe_kpi_sr"]),
            fmt_pct(row["refusal_rate"]),
            fmt_pct(row["hs_mr_mandated"]),
            fmt_pct(row["hs_mr_incentivized"]),
        ]
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    ensure_parent(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
