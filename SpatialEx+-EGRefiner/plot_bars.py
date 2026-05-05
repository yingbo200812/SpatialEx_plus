"""SpatialEx+-EGRefiner 柱状图（global metrics + 目标基因 PCC）。

作用
----
读取 :mod:`gpt.postprocess` 已经落盘的两份 CSV：

    final_metrics_vs_paper.csv      ← global metrics (PCC / SSIM / CMD)
    final_selected_gene_pcc.csv     ← 4 个目标基因 per-gene PCC

并生成 3 张 PPT-ready 的柱状对比图（Paper vs Ours-Final，每个指标一对柱子）：

    fig_bars_B1.png       —— Slice 1 上预测 Panel B 的 PCC / SSIM / CMD
    fig_bars_A2.png       —— Slice 2 上预测 Panel A 的 PCC / SSIM / CMD
    fig_bars_4genes.png   —— ESR1 / ERBB2 / PGR / KRT14 的 PCC

每根柱子顶端标注绝对值，副标题打 Δ-paper 与 ``WIN/LOSE`` 徽章。CMD 自动
按 *越低越好* 判定。

示例命令
--------

.. code-block:: bash

    python gpt/plot_bars.py \\
        --metrics-csv ./results_final/final_metrics_vs_paper.csv \\
        --genes-csv   ./results_final/final_selected_gene_pcc.csv \\
        --outdir      ./results_final/figs_bars

省略所有参数会回退到 ``tutorials/my/`` 下的默认 CSV，便于本地一键出图。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_GLOBAL = {
    "B1": {"gene_pcc_mean": 0.2956538, "ssim": 0.34703283, "cmd": 0.34355534},
    "A2": {"gene_pcc_mean": 0.31071076, "ssim": 0.36534515, "cmd": 0.35077667},
}
PAPER_GENE = {
    ("slice2", "ESR1"):  0.369,
    ("slice2", "ERBB2"): 0.661,
    ("slice1", "PGR"):   0.144,
    ("slice1", "KRT14"): 0.650,
}

METRIC_LABELS = {
    "gene_pcc_mean": "gene-PCC (mean)",
    "ssim":          "SSIM",
    "cmd":           "CMD",
}
METRIC_LOWER_BETTER = {
    "gene_pcc_mean": False,
    "ssim":          False,
    "cmd":           True,
}

# Paper-style colours: cool grey for Paper bar, accent green for Ours bar.
COLOR_PAPER = "#9ba6b8"
COLOR_OURS  = "#5cb88f"
COLOR_LOSE  = "#e08a8a"      # red-ish, used for Ours when we LOSE

TITLE_FS  = 14
LABEL_FS  = 12
ANNOT_FS  = 11
LEGEND_FS = 11


def _annotate_bar(ax, x: float, y: float, text: str,
                  color: str = "#202020", offset_y: float = 0.005):
    ax.text(x, y + offset_y, text, ha="center", va="bottom",
            fontsize=ANNOT_FS, color=color)


def _delta_line(metric_label: str, paper_v: float, ours_v: float,
                lower_is_better: bool) -> Tuple[str, bool]:
    if lower_is_better:
        delta = paper_v - ours_v
    else:
        delta = ours_v - paper_v
    win = delta > 0
    sign = "↓" if lower_is_better else "↑"
    badge = "WIN" if win else "LOSE"
    return (f"{metric_label}: Δ={delta:+.4f} {sign}  [{badge}]"), win


def plot_global_bars(
    out_path: str,
    target: str,                 # "B1" or "A2"
    title: str,
    paper_metrics: Dict[str, float],
    ours_metrics: Dict[str, float],
):
    """3-metric bar chart for one panel target."""
    metrics = ["gene_pcc_mean", "ssim", "cmd"]
    labels  = [METRIC_LABELS[m] for m in metrics]
    paper_vals = [paper_metrics[m] for m in metrics]
    ours_vals  = [ours_metrics[m]  for m in metrics]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = np.arange(len(metrics))
    bar_w = 0.36

    # Per-metric colours: paper grey always, ours green when WIN, red when LOSE
    ours_colors = []
    sub_lines = []
    all_win = True
    for m, p_v, o_v in zip(metrics, paper_vals, ours_vals):
        line, win = _delta_line(METRIC_LABELS[m], p_v, o_v,
                                METRIC_LOWER_BETTER[m])
        sub_lines.append(line)
        ours_colors.append(COLOR_OURS if win else COLOR_LOSE)
        all_win = all_win and win

    bars_p = ax.bar(x - bar_w / 2, paper_vals, bar_w,
                    color=COLOR_PAPER, label="SpatialEx+ paper",
                    edgecolor="#5b6776")
    bars_o = ax.bar(x + bar_w / 2, ours_vals, bar_w,
                    color=ours_colors,  label="SpatialExP-EGRefiner (Ours)",
                    edgecolor="#2e6e4d")

    # Annotate each bar with its absolute value.
    for b, v in zip(bars_p, paper_vals):
        _annotate_bar(ax, b.get_x() + b.get_width() / 2, v, f"{v:.4f}")
    for b, v in zip(bars_o, ours_vals):
        _annotate_bar(ax, b.get_x() + b.get_width() / 2, v, f"{v:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=LABEL_FS)
    ax.set_ylabel("metric value", fontsize=LABEL_FS)
    y_top = max(max(paper_vals), max(ours_vals)) * 1.18
    ax.set_ylim(0, y_top)
    # Subtle horizontal grid in the background
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    main_title = f"{title}  [{'all WIN' if all_win else 'partial WIN'}]"
    ax.set_title(main_title, fontsize=TITLE_FS, weight="bold", pad=18)
    sub = "\n".join(sub_lines)
    ax.text(0.5, 1.005, sub, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=ANNOT_FS - 1, color="#444")

    ax.legend(fontsize=LEGEND_FS, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_genes_bars(
    out_path: str,
    rows: List[Dict[str, float]],
    title: str = "Target gene PCC: Ours-Final vs SpatialEx+ paper baseline",
):
    """4-gene PCC comparison."""
    genes  = [r["gene"] for r in rows]
    paper  = [r["paper_pcc"] for r in rows]
    ours   = [r["final_pcc"] for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    x = np.arange(len(genes))
    bar_w = 0.36

    ours_colors = []
    sub_lines = []
    for r in rows:
        win = r["delta_paper"] > 0
        ours_colors.append(COLOR_OURS if win else COLOR_LOSE)
        badge = "WIN" if win else "LOSE"
        sub_lines.append(
            f"{r['gene']}: Δ={r['delta_paper']:+.4f}  [{badge}]"
        )

    bars_p = ax.bar(x - bar_w / 2, paper, bar_w,
                    color=COLOR_PAPER, label="SpatialEx+ paper",
                    edgecolor="#5b6776")
    bars_o = ax.bar(x + bar_w / 2, ours, bar_w,
                    color=ours_colors,
                    label="SpatialExP-EGRefiner (Ours)",
                    edgecolor="#2e6e4d")

    for b, v in zip(bars_p, paper):
        _annotate_bar(ax, b.get_x() + b.get_width() / 2, v, f"{v:.3f}")
    for b, v in zip(bars_o, ours):
        _annotate_bar(ax, b.get_x() + b.get_width() / 2, v, f"{v:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['gene']}\n({r['slice']})" for r in rows], fontsize=LABEL_FS,
    )
    ax.set_ylabel("PCC", fontsize=LABEL_FS)
    y_top = max(max(paper), max(ours)) * 1.18
    y_bot = min(min(paper), min(ours))
    if y_bot < 0:
        ax.set_ylim(y_bot * 1.2, y_top)
    else:
        ax.set_ylim(0, y_top)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(title, fontsize=TITLE_FS, weight="bold", pad=18)
    sub = "    |    ".join(sub_lines)
    ax.text(0.5, 1.005, sub, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=ANNOT_FS - 1, color="#444")

    ax.legend(fontsize=LEGEND_FS, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def parse_args():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        default=os.path.join(repo_root, "tutorials", "my",
                             "final_metrics_vs_paper.csv"),
    )
    ap.add_argument(
        "--genes-csv",
        default=os.path.join(repo_root, "tutorials", "my",
                             "final_selected_gene_pcc.csv"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(repo_root, "tutorials", "my", "figs"),
    )
    ap.add_argument(
        "--ours-method", default="Final",
        help="Which method's row to read as 'Ours-Final' "
             "(default: 'Final').",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[read] metrics csv:  {args.metrics_csv}")
    metrics_df = pd.read_csv(args.metrics_csv)
    print(f"[read] genes  csv:  {args.genes_csv}")
    genes_df = pd.read_csv(args.genes_csv)

    # ---- B1 + A2 global bar charts ----
    for tgt in ("B1", "A2"):
        sub = metrics_df[(metrics_df["prediction_target"] == tgt) &
                         (metrics_df["method"] == args.ours_method)]
        if sub.empty:
            print(f"[skip] no row for target={tgt} method={args.ours_method!r}")
            continue
        ours_metrics = {
            "gene_pcc_mean": float(sub["gene_pcc_mean"].iloc[0]),
            "ssim":          float(sub["ssim"].iloc[0]),
            "cmd":           float(sub["cmd"].iloc[0]),
        }
        paper_metrics = PAPER_GLOBAL[tgt]
        if tgt == "B1":
            title = "Evaluation of the predicted Panel B on Slice 1"
            label = "B1"
        else:
            title = "Evaluation of the predicted Panel A on Slice 2"
            label = "A2"
        plot_global_bars(
            out_path=os.path.join(args.outdir, f"fig_bars_{label}.png"),
            target=label, title=title,
            paper_metrics=paper_metrics, ours_metrics=ours_metrics,
        )

    # ---- 4-gene chart ----
    # The genes csv has columns:
    #   slice, gene, raw_pcc, direct_pcc, ridge_pcc, final_pcc, paper_pcc, delta_paper
    # We want them in this order:  ESR1, ERBB2, PGR, KRT14
    desired_order = ["ESR1", "ERBB2", "PGR", "KRT14"]
    rows: List[Dict[str, float]] = []
    for g in desired_order:
        sub = genes_df[genes_df["gene"] == g]
        if sub.empty:
            print(f"[skip] gene {g} not found in genes csv")
            continue
        r = sub.iloc[0]
        rows.append({
            "slice":       str(r["slice"]),
            "gene":        str(r["gene"]),
            "raw_pcc":     float(r["raw_pcc"]),
            "final_pcc":   float(r["final_pcc"]),
            "paper_pcc":   float(r["paper_pcc"]),
            "delta_paper": float(r["delta_paper"]),
        })
    if rows:
        plot_genes_bars(
            out_path=os.path.join(args.outdir, "fig_bars_4genes.png"),
            rows=rows,
        )

    print(f"\n[done] figures saved to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
