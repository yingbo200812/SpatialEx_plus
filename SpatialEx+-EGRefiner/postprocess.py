"""SpatialEx+-EGRefiner 后处理与最终决策（per-gene final selection）。

作用
----
本脚本是「**只跑后处理、不重训 SpatialEx+**」的快速路径：吃 :mod:`gpt.seed_search`
（或任何前序流水线）落盘的 ``B1_raw / A2_raw / *_direct / *_indirect``
四组预测，把它们组合成 *逐基因* 最优的 final 预测，并和 paper 基线做
对照。

流程
----
1. 从 ``--load-preds`` 目录加载缓存的 SpatialEx+ 预测。
2. 在 UNI HE 特征上做 panel-wise Ridge 回归，用 5-fold OOF 在
   training-side 得到无偏估计。
3. 在 training slice 上对每个基因拟合线性 stacking，把 ``[raw, ridge]``
   组合成 stack。
4. 按以下顺序为每个基因选定最终预测：

       Rule 1   train-side ``raw`` PCC < ``raw_broken_threshold``  →  ``direct``
               （挽救 cycle 已断的基因，例如 *PGR*）
       Rule 2   stack 在 OOF 训练 PCC 上比 raw 提升 ≥ ``--stack-margin`` →  ``stack``
               （捕捉与 raw 互补的 HE 信号，例如 *ESR1, KRT14*）
       Rule 3   兜底用 ``raw``

5. 计算 global + per-target-gene metric 并与 paper 数字逐项对比。
6. 渲染 Figure-3 风格的 4×3 空间图。

输出文件
--------
* ``final_metrics_vs_paper.csv``        —— 所有 method × 所有 target 的指标
* ``final_selected_gene_pcc.csv``       —— 4 个目标基因 per-gene PCC
* ``final_B1.npy`` / ``final_A2.npy``   —— 每个细胞的最终预测
* ``figure3_spatial_maps.png``          —— Measured | Raw | Ours 4×3 图
* ``figure3_compare_paper_vs_ours.png`` —— 上下对比 raw vs ours

示例命令
--------

.. code-block:: bash

    python gpt/postprocess.py \\
        --adata1 .../Rep1_uni_resolution64_panelA.h5ad \\
        --adata2 .../Rep2_uni_resolution64_panelB.h5ad \\
        --raw-data-root .../raw_data/HBC \\
        --selection     .../raw_data/HBC/Selection_by_name.csv \\
        --load-preds    ./results_seed_search \\
        --device cuda:0 \\
        --outdir ./results_final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for p in (_REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import SpatialEx as se  # noqa: E402

from gpt.pipeline import (  # noqa: E402
    TARGET_GENES,
    _to_dense,
    _ensure_he_key,
    _gene_pcc_array,
    _spot_pcc_array,
    compute_full_metrics,
    load_full_ground_truth,
    _he_ridge_oof_and_test,
    _per_gene_linear_stack_torch,
    _maybe_image_coor,
)

# Paper-reported metrics (Tutorial 2 evaluation, same evaluation set as ours).
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


# ---------------------------------------------------------------------------
# Stage 1: load cached predictions
# ---------------------------------------------------------------------------

REQUIRED_PREDS = (
    "B1_raw", "A2_raw",
    "B1_direct", "A2_direct",
    "B1_indirect", "A2_indirect",
    "A1_direct", "A1_indirect",
    "B2_direct", "B2_indirect",
)


def load_cached_preds(load_dir: str) -> Dict[str, np.ndarray]:
    print(f"\n[load] reading cached predictions from {load_dir}")
    preds: Dict[str, np.ndarray] = {}
    for k in REQUIRED_PREDS:
        path = os.path.join(load_dir, f"{k}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"missing cached prediction: {path}\n"
                f"please run gpt/pipeline.py first to "
                f"populate this directory.")
        preds[k] = np.load(path).astype(np.float32, copy=False)
        print(f"  {k}: {preds[k].shape}")

    # Optional: pick up EGRefiner outputs if they exist.
    for k in ("B1_refined", "A2_refined"):
        path = os.path.join(load_dir, f"{k}.npy")
        if os.path.isfile(path):
            preds[k] = np.load(path).astype(np.float32, copy=False)
            print(f"  {k}: {preds[k].shape}  (optional)")
    return preds


# ---------------------------------------------------------------------------
# Stage 2: per-gene final prediction (broken-raw heuristic + stacking)
# ---------------------------------------------------------------------------

def per_gene_final(
    raw_train: np.ndarray,
    direct_train: np.ndarray,
    ridge_oof_train: np.ndarray,
    gt_train: np.ndarray,
    raw_test: np.ndarray,
    direct_test: np.ndarray,
    ridge_test: np.ndarray,
    raw_broken_threshold: float = 0.4,
    direct_low_margin: float = 0.05,
    stack_margin: float = 0.005,
    stack_reg: float = 1e-2,
    device: str = "cuda:0",
    label: str = "panel",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Choose the per-gene final prediction on the test slice.

    Returns
    -------
    final_test : (n_test, n_genes)
        Per-gene final test prediction.
    decisions : DataFrame
        Per-gene decision log with columns
        ``[gene_idx, raw_pcc_train, direct_pcc_train, ridge_pcc_train,
        stack_pcc_train, decision]`` -- useful for downstream analysis /
        debugging.
    """
    n_genes = gt_train.shape[1]

    # Per-gene OOF training PCCs.
    raw_pcc_t = _gene_pcc_array(raw_train, gt_train)
    direct_pcc_t = _gene_pcc_array(direct_train, gt_train)
    ridge_pcc_t = _gene_pcc_array(ridge_oof_train, gt_train)

    # Per-gene linear stacking on training slice.
    stack_train = _per_gene_linear_stack_torch(
        features_train=[raw_train, ridge_oof_train],
        gt_train=gt_train,
        features_test=[raw_train, ridge_oof_train],
        reg=stack_reg, device=device, nonneg=False,
    )
    stack_test = _per_gene_linear_stack_torch(
        features_train=[raw_train, ridge_oof_train],
        gt_train=gt_train,
        features_test=[raw_test, ridge_test],
        reg=stack_reg, device=device, nonneg=False,
    )
    stack_pcc_t = _gene_pcc_array(stack_train, gt_train)

    final_test = raw_test.copy()
    decisions = []
    n_broken = n_stack = n_raw = 0
    for gi in range(n_genes):
        r = raw_pcc_t[gi]
        d = direct_pcc_t[gi]
        rd = ridge_pcc_t[gi]
        st = stack_pcc_t[gi]
        kind = "raw"

        if np.isfinite(r) and r < raw_broken_threshold and np.isfinite(d) \
                and d >= r + direct_low_margin:
            final_test[:, gi] = direct_test[:, gi]
            kind = "direct(broken_raw)"
            n_broken += 1
        elif np.isfinite(st) and np.isfinite(r) and st >= r + stack_margin:
            final_test[:, gi] = stack_test[:, gi]
            kind = "stack"
            n_stack += 1
        else:
            n_raw += 1

        decisions.append(dict(
            gene_idx=gi,
            raw_train_pcc=float(r) if np.isfinite(r) else np.nan,
            direct_train_pcc=float(d) if np.isfinite(d) else np.nan,
            ridge_train_pcc=float(rd) if np.isfinite(rd) else np.nan,
            stack_train_pcc=float(st) if np.isfinite(st) else np.nan,
            decision=kind,
        ))

    print(f"  [{label}] decisions: raw={n_raw}/{n_genes}  "
          f"stack={n_stack}/{n_genes}  direct(broken)={n_broken}/{n_genes}")
    return final_test, pd.DataFrame(decisions)


# ---------------------------------------------------------------------------
# Stage 3: visualisation (Figure-3 style)
# ---------------------------------------------------------------------------

def _safe_quantile(v: np.ndarray, q: float) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.quantile(v, q))


def _scatter_panel(ax, coor: np.ndarray, value: np.ndarray, title: str,
                   v_lo: float, v_hi: float, cmap: str = "viridis"):
    sc_obj = ax.scatter(
        coor[:, 0], -coor[:, 1],
        c=value, cmap=cmap, s=2.0, vmin=v_lo, vmax=v_hi,
        rasterized=True, linewidths=0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc_obj


def plot_figure3_style(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    gtA2: np.ndarray,
    gtB1: np.ndarray,
    A2_raw: np.ndarray, A2_final: np.ndarray,
    B1_raw: np.ndarray, B1_final: np.ndarray,
    panelA_genes: List[str],
    panelB_genes: List[str],
    out_path: str,
):
    """Three-column figure: Measured | Raw SpatialEx+ | Ours-Final."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coor1 = _maybe_image_coor(adata1)
    if coor1 is None:
        coor1 = np.asarray(adata1.obsm["spatial"], dtype=np.float32)[:, :2]
    coor2 = _maybe_image_coor(adata2)
    if coor2 is None:
        coor2 = np.asarray(adata2.obsm["spatial"], dtype=np.float32)[:, :2]

    panels = []
    # Slice 2 (Panel A target genes)
    for g in TARGET_GENES["slice2"]:
        if g not in panelA_genes:
            continue
        gi = panelA_genes.index(g)
        panels.append(dict(
            gene=g, slice="slice2", coor=coor2,
            true=gtA2[:, gi], raw=A2_raw[:, gi], final=A2_final[:, gi],
        ))
    # Slice 1 (Panel B target genes)
    for g in TARGET_GENES["slice1"]:
        if g not in panelB_genes:
            continue
        gi = panelB_genes.index(g)
        panels.append(dict(
            gene=g, slice="slice1", coor=coor1,
            true=gtB1[:, gi], raw=B1_raw[:, gi], final=B1_final[:, gi],
        ))

    n = len(panels)
    if n == 0:
        print("[plot] no target genes available")
        return
    fig, axes = plt.subplots(n, 3, figsize=(12.5, 4.0 * n))
    if n == 1:
        axes = np.asarray(axes).reshape(1, 3)
    cmap = "viridis"

    for r, p in enumerate(panels):
        gt = p["true"]; raw = p["raw"]; fin = p["final"]
        v_lo = _safe_quantile(gt, 0.02)
        v_hi = _safe_quantile(gt, 0.98)
        if v_hi <= v_lo:
            v_hi = v_lo + 1e-3

        raw_pcc = _gene_pcc_array(raw[:, None], gt[:, None])[0]
        fin_pcc = _gene_pcc_array(fin[:, None], gt[:, None])[0]

        sc_obj1 = _scatter_panel(
            axes[r, 0], p["coor"], gt,
            f"Measured\n{p['slice']} ({p['gene']})", v_lo, v_hi, cmap)
        sc_obj2 = _scatter_panel(
            axes[r, 1], p["coor"], raw,
            f"Raw SpatialEx+\n{p['slice']} ({p['gene']}) | PCC={raw_pcc:+.3f}",
            v_lo, v_hi, cmap)
        sc_obj3 = _scatter_panel(
            axes[r, 2], p["coor"], fin,
            f"SpatialExP-EGRefiner (Ours)\n{p['slice']} ({p['gene']}) | PCC={fin_pcc:+.3f}",
            v_lo, v_hi, cmap)
        for ax_, sc_ in zip(axes[r], (sc_obj1, sc_obj2, sc_obj3)):
            plt.colorbar(sc_, ax=ax_, fraction=0.04, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_compare_paper_vs_ours(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    gtA2: np.ndarray, gtB1: np.ndarray,
    A2_raw: np.ndarray, A2_final: np.ndarray,
    B1_raw: np.ndarray, B1_final: np.ndarray,
    panelA_genes: List[str], panelB_genes: List[str],
    out_path: str,
):
    """Two-row figure: top = Raw SpatialEx+ (paper baseline), bottom = ours.

    Layout: rows = {Raw SpatialEx+, Ours-Final}, columns = 4 target genes
    (ESR1, ERBB2, PGR, KRT14).  Same colour scale per gene.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coor1 = _maybe_image_coor(adata1)
    if coor1 is None:
        coor1 = np.asarray(adata1.obsm["spatial"], dtype=np.float32)[:, :2]
    coor2 = _maybe_image_coor(adata2)
    if coor2 is None:
        coor2 = np.asarray(adata2.obsm["spatial"], dtype=np.float32)[:, :2]

    columns = []
    # Order to mirror Figure 3: ESR1, ERBB2, PGR, KRT14
    for g in ("ESR1", "ERBB2"):
        if g not in panelA_genes:
            continue
        gi = panelA_genes.index(g)
        columns.append(dict(
            gene=g, coor=coor2, gt=gtA2[:, gi],
            raw=A2_raw[:, gi], fin=A2_final[:, gi]))
    for g in ("PGR", "KRT14"):
        if g not in panelB_genes:
            continue
        gi = panelB_genes.index(g)
        columns.append(dict(
            gene=g, coor=coor1, gt=gtB1[:, gi],
            raw=B1_raw[:, gi], fin=B1_final[:, gi]))

    n_cols = len(columns)
    if n_cols == 0:
        print("[plot] no target genes available")
        return
    fig, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 8.5))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    cmap = "viridis"

    for c, col in enumerate(columns):
        gt = col["gt"]
        v_lo = _safe_quantile(gt, 0.02)
        v_hi = _safe_quantile(gt, 0.98)
        if v_hi <= v_lo:
            v_hi = v_lo + 1e-3
        paper_v = PAPER_GENE.get(("slice2", col["gene"]),
                                 PAPER_GENE.get(("slice1", col["gene"])))
        raw_pcc = _gene_pcc_array(col["raw"][:, None], gt[:, None])[0]
        fin_pcc = _gene_pcc_array(col["fin"][:, None], gt[:, None])[0]
        title_paper = (f"SpatialEx+ (paper baseline)\n{col['gene']}\n"
                       f"ours={raw_pcc:+.3f} | paper={paper_v:+.3f}")
        title_ours = (f"SpatialExP-EGRefiner (Ours)\n{col['gene']}\n"
                      f"PCC={fin_pcc:+.3f}  Δ-paper={fin_pcc - paper_v:+.3f}")
        sc1 = _scatter_panel(axes[0, c], col["coor"], col["raw"],
                             title_paper, v_lo, v_hi, cmap)
        sc2 = _scatter_panel(axes[1, c], col["coor"], col["fin"],
                             title_ours, v_lo, v_hi, cmap)
        plt.colorbar(sc1, ax=axes[0, c], fraction=0.04, pad=0.02)
        plt.colorbar(sc2, ax=axes[1, c], fraction=0.04, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata1", required=True)
    ap.add_argument("--adata2", required=True)
    ap.add_argument("--raw-data-root", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--sample1", default="Human_Breast_Cancer_Rep1")
    ap.add_argument("--sample2", default="Human_Breast_Cancer_Rep2")
    ap.add_argument("--load-preds", required=True,
                    help="Directory with cached *.npy predictions, e.g. results_v1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--ridge-folds", type=int, default=5)
    ap.add_argument("--raw-broken-threshold", type=float, default=0.4,
                    help="If raw_train_pcc on a gene is below this, the cycle "
                         "is considered broken and we fall back to direct.")
    ap.add_argument("--direct-low-margin", type=float, default=0.05)
    ap.add_argument("--stack-margin", type=float, default=0.005,
                    help="Min OOF improvement required to switch raw->stack.")
    ap.add_argument("--stack-reg", type=float, default=1e-2)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[run] device={args.device} outdir={args.outdir}")

    # ---- 1. data ----
    adata1 = sc.read_h5ad(args.adata1)
    adata2 = sc.read_h5ad(args.adata2)
    adata1.X = _to_dense(adata1.X)
    adata2.X = _to_dense(adata2.X)
    _ensure_he_key(adata1, "adata1")
    _ensure_he_key(adata2, "adata2")
    print(f"[data] slice1: n_cells={adata1.n_obs} | n_genes_panelA={adata1.n_vars}")
    print(f"[data] slice2: n_cells={adata2.n_obs} | n_genes_panelB={adata2.n_vars}")

    # ---- 2. cached predictions ----
    preds = load_cached_preds(args.load_preds)

    # ---- 3. ground truth on missing panel ----
    panelA_genes = list(map(str, adata1.var_names))
    panelB_genes = list(map(str, adata2.var_names))
    obs1 = list(map(str, adata1.obs_names))
    obs2 = list(map(str, adata2.obs_names))
    gtB1 = load_full_ground_truth(
        args.raw_data_root, args.sample1, "Rep1",
        gene_names=panelB_genes, obs_names_target=obs1)
    gtA2 = load_full_ground_truth(
        args.raw_data_root, args.sample2, "Rep2",
        gene_names=panelA_genes, obs_names_target=obs2)
    print(f"[gt] B1: {gtB1.shape}  A2: {gtA2.shape}")

    # ---- 4. Ridge OOF + test ----
    he1 = np.asarray(adata1.obsm["he"], dtype=np.float32)
    he2 = np.asarray(adata2.obsm["he"], dtype=np.float32)
    measured_A1 = _to_dense(adata1.X)
    measured_B2 = _to_dense(adata2.X)

    print("\n[ridge] panel A: HE1 -> measured Panel A on Slice 1")
    A1_ridge_oof, A2_ridge_test = _he_ridge_oof_and_test(
        he_train=he1, gt_train=measured_A1, he_test=he2,
        alpha=args.ridge_alpha, n_folds=args.ridge_folds, seed=0,
        device=args.device,
    )
    print("[ridge] panel B: HE2 -> measured Panel B on Slice 2")
    B2_ridge_oof, B1_ridge_test = _he_ridge_oof_and_test(
        he_train=he2, gt_train=measured_B2, he_test=he1,
        alpha=args.ridge_alpha, n_folds=args.ridge_folds, seed=0,
        device=args.device,
    )
    np.save(os.path.join(args.outdir, "A2_ridge.npy"), A2_ridge_test)
    np.save(os.path.join(args.outdir, "B1_ridge.npy"), B1_ridge_test)

    # ---- 5. per-gene final ----
    print("\n[final] computing per-gene final on Panel A (test = Slice 2)")
    A2_final, dec_A = per_gene_final(
        raw_train=preds["A1_indirect"],
        direct_train=preds["A1_direct"],
        ridge_oof_train=A1_ridge_oof,
        gt_train=measured_A1,
        raw_test=preds["A2_raw"],
        direct_test=preds["A2_direct"],
        ridge_test=A2_ridge_test,
        raw_broken_threshold=args.raw_broken_threshold,
        direct_low_margin=args.direct_low_margin,
        stack_margin=args.stack_margin,
        stack_reg=args.stack_reg,
        device=args.device,
        label="A2",
    )
    print("[final] computing per-gene final on Panel B (test = Slice 1)")
    B1_final, dec_B = per_gene_final(
        raw_train=preds["B2_indirect"],
        direct_train=preds["B2_direct"],
        ridge_oof_train=B2_ridge_oof,
        gt_train=measured_B2,
        raw_test=preds["B1_raw"],
        direct_test=preds["B1_direct"],
        ridge_test=B1_ridge_test,
        raw_broken_threshold=args.raw_broken_threshold,
        direct_low_margin=args.direct_low_margin,
        stack_margin=args.stack_margin,
        stack_reg=args.stack_reg,
        device=args.device,
        label="B1",
    )
    np.save(os.path.join(args.outdir, "final_A2.npy"), A2_final.astype(np.float32))
    np.save(os.path.join(args.outdir, "final_B1.npy"), B1_final.astype(np.float32))
    dec_A.to_csv(os.path.join(args.outdir, "decisions_panelA.csv"), index=False)
    dec_B.to_csv(os.path.join(args.outdir, "decisions_panelB.csv"), index=False)

    # ---- 6. global metrics for every method ----
    g1 = se.pp.Build_graph(
        adata1.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo")
    g2 = se.pp.Build_graph(
        adata2.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo")

    method_arrays_B1 = {
        "raw":      preds["B1_raw"],
        "direct":   preds["B1_direct"],
        "indirect": preds["B1_indirect"],
        "Ridge":    B1_ridge_test,
        "Final":    B1_final,
    }
    if "B1_refined" in preds:
        method_arrays_B1["EGRefiner"] = preds["B1_refined"]

    method_arrays_A2 = {
        "raw":      preds["A2_raw"],
        "direct":   preds["A2_direct"],
        "indirect": preds["A2_indirect"],
        "Ridge":    A2_ridge_test,
        "Final":    A2_final,
    }
    if "A2_refined" in preds:
        method_arrays_A2["EGRefiner"] = preds["A2_refined"]

    rows = []
    print("\n--- Slice 1 prediction of Panel B ---")
    for name, arr in method_arrays_B1.items():
        m = compute_full_metrics(arr, gtB1, spatial_graph=g1)
        rows.append({"prediction_target": "B1", "method": name, **m})
        print(f"  [{name:>10}] gPCC={m['gene_pcc_mean']:+.4f}  "
              f"sPCC={m['spot_pcc_mean']:+.4f}  RMSE={m['rmse']:.4f}  "
              f"SSIM={m['ssim']:+.4f}  CMD={m['cmd']:.4f}")
    print("\n--- Slice 2 prediction of Panel A ---")
    for name, arr in method_arrays_A2.items():
        m = compute_full_metrics(arr, gtA2, spatial_graph=g2)
        rows.append({"prediction_target": "A2", "method": name, **m})
        print(f"  [{name:>10}] gPCC={m['gene_pcc_mean']:+.4f}  "
              f"sPCC={m['spot_pcc_mean']:+.4f}  RMSE={m['rmse']:.4f}  "
              f"SSIM={m['ssim']:+.4f}  CMD={m['cmd']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.outdir, "final_metrics_vs_paper.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] {metrics_path}")

    # ---- 7. selected gene PCC table ----
    sel_rows = []
    for g in TARGET_GENES["slice2"]:
        if g not in panelA_genes:
            continue
        gi = panelA_genes.index(g)
        true_col = gtA2[:, gi]
        sel_rows.append({
            "slice": "slice2", "gene": g,
            "raw_pcc":     _gene_pcc_array(preds["A2_raw"][:, [gi]],   true_col[:, None])[0],
            "direct_pcc":  _gene_pcc_array(preds["A2_direct"][:, [gi]], true_col[:, None])[0],
            "ridge_pcc":   _gene_pcc_array(A2_ridge_test[:, [gi]],     true_col[:, None])[0],
            "final_pcc":   _gene_pcc_array(A2_final[:, [gi]],          true_col[:, None])[0],
            "paper_pcc":   PAPER_GENE.get(("slice2", g), float("nan")),
        })
    for g in TARGET_GENES["slice1"]:
        if g not in panelB_genes:
            continue
        gi = panelB_genes.index(g)
        true_col = gtB1[:, gi]
        sel_rows.append({
            "slice": "slice1", "gene": g,
            "raw_pcc":     _gene_pcc_array(preds["B1_raw"][:, [gi]],    true_col[:, None])[0],
            "direct_pcc":  _gene_pcc_array(preds["B1_direct"][:, [gi]], true_col[:, None])[0],
            "ridge_pcc":   _gene_pcc_array(B1_ridge_test[:, [gi]],      true_col[:, None])[0],
            "final_pcc":   _gene_pcc_array(B1_final[:, [gi]],           true_col[:, None])[0],
            "paper_pcc":   PAPER_GENE.get(("slice1", g), float("nan")),
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_df["delta_paper"] = sel_df["final_pcc"] - sel_df["paper_pcc"]
    sel_path = os.path.join(args.outdir, "final_selected_gene_pcc.csv")
    sel_df.to_csv(sel_path, index=False)
    print(f"[save] {sel_path}")

    # ---- 8. vs-paper summary ----
    print("\n" + "=" * 80)
    print("[vs paper] global metrics  (positive Δ = we win)")
    print("=" * 80)
    for tgt in ("B1", "A2"):
        sub = metrics_df[metrics_df["prediction_target"] == tgt]
        for metric in ("gene_pcc_mean", "ssim"):
            paper_v = PAPER_GLOBAL[tgt][metric]
            for _, r in sub.iterrows():
                ours = r[metric]
                d = ours - paper_v
                tag = "WIN" if d > 0 else "LOSE"
                print(f"  {tgt}.{r['method']:>10} {metric:<14}: "
                      f"ours={ours:+.4f}  paper={paper_v:+.4f}  Δ={d:+.4f}  [{tag}]")
        # CMD lower is better
        paper_v = PAPER_GLOBAL[tgt]["cmd"]
        for _, r in sub.iterrows():
            ours = r["cmd"]
            d = paper_v - ours
            tag = "WIN" if d > 0 else "LOSE"
            print(f"  {tgt}.{r['method']:>10} {'cmd':<14}: "
                  f"ours={ours:+.4f}  paper={paper_v:+.4f}  Δ={d:+.4f}  [{tag}]")
        print()

    print("[vs paper] selected gene PCC")
    print("-" * 80)
    for _, r in sel_df.iterrows():
        d = r["delta_paper"]
        tag = "WIN" if d > 0 else "LOSE"
        print(f"  {r['slice']}.{r['gene']:>5}  "
              f"raw={r['raw_pcc']:+.4f}  direct={r['direct_pcc']:+.4f}  "
              f"ridge={r['ridge_pcc']:+.4f}  final={r['final_pcc']:+.4f}  "
              f"paper={r['paper_pcc']:+.4f}  Δ={d:+.4f}  [{tag}]")

    # ---- 9. visualisations ----
    plot_figure3_style(
        adata1=adata1, adata2=adata2,
        gtA2=gtA2, gtB1=gtB1,
        A2_raw=preds["A2_raw"], A2_final=A2_final,
        B1_raw=preds["B1_raw"], B1_final=B1_final,
        panelA_genes=panelA_genes, panelB_genes=panelB_genes,
        out_path=os.path.join(args.outdir, "figure3_spatial_maps.png"),
    )
    plot_compare_paper_vs_ours(
        adata1=adata1, adata2=adata2,
        gtA2=gtA2, gtB1=gtB1,
        A2_raw=preds["A2_raw"], A2_final=A2_final,
        B1_raw=preds["B1_raw"], B1_final=B1_final,
        panelA_genes=panelA_genes, panelB_genes=panelB_genes,
        out_path=os.path.join(args.outdir, "figure3_compare_paper_vs_ours.png"),
    )

    with open(os.path.join(args.outdir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"\n[done] all outputs in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
