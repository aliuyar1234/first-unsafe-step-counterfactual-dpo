"""Generate the error taxonomy LaTeX table from `results/metrics/test_runs.csv`."""
from __future__ import annotations

from pathlib import Path

from src.figures.common import ensure_parent, fmt_pct, load_csv, tex_escape


def main(csv_path: Path, out_path: Path) -> None:
    df = load_csv(csv_path)
    unsafe = df[df["max_severity"] >= 3].copy()
    if unsafe.empty:
        lines = [
            "\\begin{tabular}{ll}",
            "\\toprule",
            "Method & Note \\\\",
            "\\midrule",
            "none & No severity $\\geq$ 3 runs available. \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ]
    else:
        pivot = (
            unsafe.groupby(["method", "primary_violation_type"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        totals = pivot.groupby("method")["count"].transform("sum")
        pivot["share"] = pivot["count"] / totals
        lines = [
            "\\begin{tabular}{llrr}",
            "\\toprule",
            "Method & Violation Type & Count & Share \\\\",
            "\\midrule",
        ]
        for _, row in pivot.sort_values(["method", "count"], ascending=[True, False]).iterrows():
            lines.append(
                " & ".join(
                    [
                        tex_escape(row["method"]),
                        tex_escape(row["primary_violation_type"]),
                        str(int(row["count"])),
                        fmt_pct(row["share"]),
                    ]
                )
                + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
    ensure_parent(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
