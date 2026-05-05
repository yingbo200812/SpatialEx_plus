"""
SpatialEx-pro 对比可视化脚本。

从 ``--results-dir`` 读取 ``metrics_spatialex_pro.csv``（以及可选的
``metrics_baseline.csv`` —— 在同一份数据切分上重新跑过的 SpatialEx
baseline 结果），输出三张 PNG 对比图：

1. ``bar_train_slice1_test_slice2.png``：Training Slice 1 → Test Slice 2
   方向上的 PCC / SSIM / CMD，柱簇为 [论文 baseline | SpatialEx 本次复
   现 | SpatialEx-pro]。
2. ``bar_train_slice2_test_slice1.png``：另一方向同款柱状图。
3. ``bar_marker_pcc.png``：左右两子图分别对应两个方向，分别画 EPCAM /
   ESR1 / PGR 三个 marker 基因的 PCC 对比。

约定：
- CSV 中 ``slice1_*`` = head-B (slice2 训练) 在 slice1 上的预测对真值
  的指标，对应 "Train Slice 2 / Test Slice 1"；``slice2_*`` 同理对应
  "Train Slice 1 / Test Slice 2"。
- CMD 是 *越低越好*，其余指标越高越好；在轴标签里直接用 ↑/↓ 标注。

用法::

    python SpatialEx_pro/make_plots.py --results-dir <OUTDIR>
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless on the remote server
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paper-reported numbers (Fig. 2c/d of the SpatialEx paper).
# These are the *fixed* reference values; the user wants every plot to
# put SpatialEx-pro head-to-head with these.
# ---------------------------------------------------------------------------

PAPER = {
    # Train Slice 1 -> Test Slice 2  (i.e. ``slice2_*`` columns)
    "slice2_PCC":  0.2576,
    "slice2_SSIM": 0.3654,
    "slice2_CMD":  0.2149,
    # Train Slice 2 -> Test Slice 1  (i.e. ``slice1_*`` columns)
    "slice1_PCC":  0.2733,
    "slice1_SSIM": 0.3809,
    "slice1_CMD":  0.2033,
    # Marker genes (paper Fig. 2d)
    "slice2_PCC[EPCAM]": 0.7563,
    "slice2_PCC[ESR1]":  0.3170,
    "slice2_PCC[PGR]":   0.1130,
    "slice1_PCC[EPCAM]": 0.7610,
    "slice1_PCC[ESR1]":  0.2402,
    "slice1_PCC[PGR]":   0.1168,
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_metrics(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    row = df.iloc[0].to_dict()
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in row.items() if k != "label"}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# A small palette borrowed from a colorblind-friendly set.
_COLOR_PAPER     = "#9aa0a6"   # neutral grey -> paper baseline
_COLOR_BASELINE  = "#7baaf7"   # light blue   -> in-house SpatialEx run
_COLOR_PRO       = "#f4794d"   # warm orange  -> SpatialEx-pro


def _bar_three_metrics(
    paper: Dict[str, float],
    baseline: Optional[Dict[str, float]],
    pro: Dict[str, float],
    metric_keys,
    title: str,
    out_path: str,
) -> None:
    """Plot PCC / SSIM / CMD as a grouped bar chart.

    ``metric_keys`` is the list of CSV column names in plot order
    (e.g. ``("slice2_PCC", "slice2_SSIM", "slice2_CMD")``).
    """
    metric_labels = []
    for k in metric_keys:
        if k.endswith("_PCC"):
            metric_labels.append("PCC ↑")
        elif k.endswith("_SSIM"):
            metric_labels.append("SSIM ↑")
        elif k.endswith("_CMD"):
            metric_labels.append("CMD ↓")
        else:
            metric_labels.append(k)

    has_baseline = baseline is not None
    n_groups = len(metric_keys)
    n_bars = 3 if has_baseline else 2
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    paper_vals    = [paper.get(k, np.nan) for k in metric_keys]
    pro_vals      = [pro.get(k, np.nan)   for k in metric_keys]

    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width
    bars = []
    bars.append(("Paper baseline", paper_vals,    _COLOR_PAPER))
    if has_baseline:
        bl_vals = [baseline.get(k, np.nan) for k in metric_keys]
        bars.append(("SpatialEx (this run)", bl_vals, _COLOR_BASELINE))
    bars.append(("SpatialEx-pro", pro_vals, _COLOR_PRO))

    for i, (label, vals, color) in enumerate(bars):
        rects = ax.bar(x + offsets[i], vals, width, label=label,
                       color=color, edgecolor="black", linewidth=0.6)
        for r, v in zip(rects, vals):
            if not np.isfinite(v):
                continue
            ax.text(r.get_x() + r.get_width() / 2.0, r.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("metric value", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # tighten y-limits to keep label headroom but stay informative
    ymax = max([v for v in (paper_vals + pro_vals
                            + ([] if not has_baseline else
                               [baseline.get(k, np.nan) for k in metric_keys]))
                if np.isfinite(v)] + [0.0])
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _bar_marker_genes(
    paper: Dict[str, float],
    baseline: Optional[Dict[str, float]],
    pro: Dict[str, float],
    out_path: str,
    marker_genes=("EPCAM", "ESR1", "PGR"),
) -> None:
    """Per-gene PCC for the three Fig. 2d marker genes.

    Two side-by-side subplots, one per cross-slice direction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    direction_specs = [
        ("Train Slice 1 -> Test Slice 2",
         [f"slice2_PCC[{g}]" for g in marker_genes]),
        ("Train Slice 2 -> Test Slice 1",
         [f"slice1_PCC[{g}]" for g in marker_genes]),
    ]

    has_baseline = baseline is not None
    n_bars = 3 if has_baseline else 2
    width = 0.8 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

    all_vals = []
    for ax, (title, keys) in zip(axes, direction_specs):
        x = np.arange(len(marker_genes))
        paper_vals = [paper.get(k, np.nan) for k in keys]
        pro_vals   = [pro.get(k, np.nan)   for k in keys]
        all_vals.extend([v for v in paper_vals + pro_vals if np.isfinite(v)])

        bars = [("Paper baseline", paper_vals, _COLOR_PAPER)]
        if has_baseline:
            bl_vals = [baseline.get(k, np.nan) for k in keys]
            bars.append(("SpatialEx (this run)", bl_vals, _COLOR_BASELINE))
            all_vals.extend([v for v in bl_vals if np.isfinite(v)])
        bars.append(("SpatialEx-pro", pro_vals, _COLOR_PRO))

        for i, (label, vals, color) in enumerate(bars):
            rects = ax.bar(x + offsets[i], vals, width, label=label,
                           color=color, edgecolor="black", linewidth=0.6)
            for r, v in zip(rects, vals):
                if not np.isfinite(v):
                    continue
                ax.text(r.get_x() + r.get_width() / 2.0,
                        r.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(list(marker_genes), fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("PCC (higher is better)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    ymax = max(all_vals) if all_vals else 1.0
    for ax in axes:
        ax.set_ylim(0, ymax * 1.18)

    fig.suptitle("Marker-gene PCC: SpatialEx-pro vs paper baseline",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="/data1/linxin/1/SpatialEx_pro/results/run_default",
        help="Folder containing metrics_spatialex_pro.csv "
             "(and, optionally, metrics_baseline.csv).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Where to save the PNG figures.  Default: <results-dir>/figures.",
    )
    args = ap.parse_args()

    pro_csv = os.path.join(args.results_dir, "metrics_spatialex_pro.csv")
    bl_csv  = os.path.join(args.results_dir, "metrics_baseline.csv")

    pro = _read_metrics(pro_csv)
    if pro is None:
        raise SystemExit(
            f"[make_plots] Required file not found: {pro_csv}\n"
            f"           Run SpatialEx-pro first (see SpatialEx-pro.sh)."
        )
    baseline = _read_metrics(bl_csv)
    if baseline is None:
        print(f"[make_plots] {bl_csv} not found -- plots will only show "
              f"paper baseline vs SpatialEx-pro.")

    out_dir = args.outdir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Train Slice 1, Test Slice 2  (slice2_* columns)
    _bar_three_metrics(
        paper=PAPER, baseline=baseline, pro=pro,
        metric_keys=("slice2_PCC", "slice2_SSIM", "slice2_CMD"),
        title="Training on Slice 1, test on Slice 2",
        out_path=os.path.join(out_dir, "bar_train_slice1_test_slice2.png"),
    )

    # Plot 2: Train Slice 2, Test Slice 1  (slice1_* columns)
    _bar_three_metrics(
        paper=PAPER, baseline=baseline, pro=pro,
        metric_keys=("slice1_PCC", "slice1_SSIM", "slice1_CMD"),
        title="Training on Slice 2, test on Slice 1",
        out_path=os.path.join(out_dir, "bar_train_slice2_test_slice1.png"),
    )

    # Plot 3: Marker-gene PCC
    _bar_marker_genes(
        paper=PAPER, baseline=baseline, pro=pro,
        out_path=os.path.join(out_dir, "bar_marker_pcc.png"),
    )

    print(f"\n[make_plots] All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
