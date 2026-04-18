"""Generate the ablation LaTeX table from held-out summary metrics."""
from __future__ import annotations

from pathlib import Path

from src.figures.common import ensure_parent, fmt_num, fmt_pct, load_csv, tex_escape


def main(csv_path: Path, out_path: Path) -> None:
    df = load_csv(csv_path)
    keep_methods = {"contract_prompt", "sft_chosen", "fus_dpo", "dpo_no_contract"}
    filtered = df[df["method"].isin(keep_methods)].sort_values(["method"])
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & HS-MR & Safe-KPI-SR & KPI-SR & RefusalRate & Delta HS-MR vs Best Prompt \\\\",
        "\\midrule",
    ]
    for _, row in filtered.iterrows():
        lines.append(
            " & ".join(
                [
                    tex_escape(row["method"]),
                    fmt_pct(row["hs_mr"]),
                    fmt_pct(row["safe_kpi_sr"]),
                    fmt_pct(row["kpi_sr"]),
                    fmt_pct(row["refusal_rate"]),
                    fmt_num(row.get("delta_hs_mr_vs_best_prompt", 0.0), 3),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    ensure_parent(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
