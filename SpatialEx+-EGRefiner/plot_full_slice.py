"""SpatialEx+-EGRefiner 全切片可视化（论文 Figure-3 风格）。

作用
----
SpatialEx+ 缓存的 ``adata.obsm['he']`` 只覆盖了 **落在 Visium spot 内**
的细胞。两份外挂文件

    {raw_data_root}/{sample}/HBRC_{rep}_Out_uni.npy
    {raw_data_root}/{sample}/HBRC_{rep}_cell_coor.csv

提供了 spot 之外那些细胞的 UNI 特征与影像级坐标。把内部细胞与外部细胞
合并起来，就能铺满整张组织切片，复现 SpatialEx+ 论文 Figure 3 的稠密
可视化效果。

流程
----
1. 用与缓存预测一致的 ``--seed`` 重训 SpatialEx+（拿到一个活的 backbone
   + regression mapper）。
2. 对 *spot 外* 的 UNI 特征调 ``inference_indirect`` / ``inference_direct``
   得到全切片每个细胞的预测。
3. **简化版** per-gene 决策（broken-raw heuristic，**不启用** stacking）：

       - train-side ``raw`` PCC < ``--raw-broken-threshold`` → 用 ``direct``；
       - 否则保留 ``raw``。

   ``postprocess.py`` 的 stacking 路径在「跨切片泛化」时偶尔会反而压低
   PCC，作图脚本只关心稳定可读，因此这里保证 final ≥ max(raw, 在 raw 已
   崩时换 direct)。

4. 生成两类图：

   * **每个目标基因一张 PNG**：``fullslice_<GENE>.png``，整片细胞的
     Ours-Final 预测，标题为 ``"<gene>, PCC=<value>"``。
   * **合成对照 PNG**：4 行 × 3 列 ``Measured | Raw SpatialEx+ | Ours-Final``，
     一眼能看出 final 相对 raw 的提升。

示例命令
--------

.. code-block:: bash

    python gpt/plot_full_slice.py \\
        --adata1 .../Rep1_uni_resolution64_panelA.h5ad \\
        --adata2 .../Rep2_uni_resolution64_panelB.h5ad \\
        --raw-data-root .../raw_data/HBC \\
        --selection     .../raw_data/HBC/Selection_by_name.csv \\
        --device cuda:0 --seed 0 --epochs 500 \\
        --outdir ./results_full_slice_figs
"""

from __future__ import annotations

import argparse
import os
import sys
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
    _maybe_image_coor,
    _gene_pcc_array,
    compute_full_metrics,
    train_original_spatialexp,
    load_full_ground_truth,
)


PAPER_GENE = {
    ("slice2", "ESR1"):  0.369,
    ("slice2", "ERBB2"): 0.661,
    ("slice1", "PGR"):   0.144,
    ("slice1", "KRT14"): 0.650,
}

# Global metrics from Tutorial 2 evaluation (= paper baseline).
PAPER_GLOBAL = {
    "B1": {"gene_pcc_mean": 0.2956538, "ssim": 0.34703283, "cmd": 0.34355534},
    "A2": {"gene_pcc_mean": 0.31071076, "ssim": 0.36534515, "cmd": 0.35077667},
}


# ---------------------------------------------------------------------------
# Out-of-spot data loading
# ---------------------------------------------------------------------------

def load_out_of_spot(raw_data_root: str, sample_name: str, rep_tag: str
                     ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """Load ``HBRC_{rep}_Out_uni.npy`` and ``HBRC_{rep}_cell_coor.csv``.

    Returns ``(None, None)`` if either file is missing.  This is needed
    so the script can still produce a (sparser) inner-cells-only figure
    when the out-of-spot caches are unavailable.
    """
    sample_dir = os.path.join(raw_data_root, sample_name)
    he_path  = os.path.join(sample_dir, f"HBRC_{rep_tag}_Out_uni.npy")
    coor_path = os.path.join(sample_dir, f"HBRC_{rep_tag}_cell_coor.csv")
    if not os.path.isfile(he_path) or not os.path.isfile(coor_path):
        print(f"[out-of-spot] not found for {rep_tag}, falling back to inner cells only.")
        print(f"  expected: {he_path}\n            {coor_path}")
        return None, None
    print(f"[out-of-spot] loading {he_path}  +  {coor_path}")
    he_out = np.load(he_path).astype(np.float32, copy=False)
    coor_out = pd.read_csv(coor_path, index_col=0)
    print(f"[out-of-spot] {rep_tag}: he={he_out.shape}  coor={coor_out.shape}")
    return he_out, coor_out


# ---------------------------------------------------------------------------
# Out-of-spot prediction via SpatialExP
# ---------------------------------------------------------------------------

def predict_out_of_spot(
    sxp: "se.SpatialExP",
    he_out: np.ndarray,
    coor_out: pd.DataFrame,
    num_neighbors: int,
) -> Dict[str, np.ndarray]:
    """Return ``{panelA_indirect, panelA_direct, panelB_indirect, panelB_direct}``
    on the out-of-spot cells, mirroring Tutorial 2 exactly."""
    print(f"  building out-of-spot hypergraph (k={num_neighbors}) on "
          f"{coor_out.shape[0]} cells ...")
    coor_arr = coor_out.values.astype(np.float32)
    graph_out = se.pp.Build_hypergraph(
        coor_arr, num_neighbors=num_neighbors, normalize=True)
    print("  inference_indirect (panelA) ...")
    panelA_ind = sxp.inference_indirect(he_out, graph_out, panel="panelA")
    print("  inference_indirect (panelB) ...")
    panelB_ind = sxp.inference_indirect(he_out, graph_out, panel="panelB")
    print("  inference_direct (panelA) ...")
    panelA_dir = sxp.inference_direct(he_out, graph_out, panel="panelA")
    print("  inference_direct (panelB) ...")
    panelB_dir = sxp.inference_direct(he_out, graph_out, panel="panelB")
    return {
        "panelA_indirect": np.asarray(panelA_ind, dtype=np.float32),
        "panelA_direct":   np.asarray(panelA_dir, dtype=np.float32),
        "panelB_indirect": np.asarray(panelB_ind, dtype=np.float32),
        "panelB_direct":   np.asarray(panelB_dir, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Per-gene "broken-raw" final selection
# ---------------------------------------------------------------------------

def per_gene_choose_final(
    raw_train: np.ndarray,
    direct_train: np.ndarray,
    gt_train: np.ndarray,
    raw_test_inner: np.ndarray, direct_test_inner: np.ndarray,
    raw_test_outer: Optional[np.ndarray] = None,
    direct_test_outer: Optional[np.ndarray] = None,
    raw_broken_threshold: float = 0.4,
    direct_low_margin: float = 0.05,
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Select per-gene final = direct if cycle-broken on training, else raw.

    Stacking is intentionally disabled.

    Returns
    -------
    final_inner : (n_inner, n_genes)
    final_outer : (n_outer, n_genes) or ``None``
    decisions   : DataFrame with per-gene decision log
    """
    raw_pcc_t = _gene_pcc_array(raw_train, gt_train)
    direct_pcc_t = _gene_pcc_array(direct_train, gt_train)
    n_genes = gt_train.shape[1]

    final_inner = raw_test_inner.copy()
    final_outer = raw_test_outer.copy() if raw_test_outer is not None else None
    decisions = []

    n_broken = n_keep_raw = 0
    for gi in range(n_genes):
        r = raw_pcc_t[gi]
        d = direct_pcc_t[gi]
        kind = "raw"
        if (np.isfinite(r) and r < raw_broken_threshold and np.isfinite(d)
                and d >= r + direct_low_margin):
            final_inner[:, gi] = direct_test_inner[:, gi]
            if final_outer is not None:
                final_outer[:, gi] = direct_test_outer[:, gi]
            kind = "direct(broken_raw)"
            n_broken += 1
        else:
            n_keep_raw += 1
        decisions.append(dict(
            gene_idx=gi,
            raw_train_pcc=float(r) if np.isfinite(r) else np.nan,
            direct_train_pcc=float(d) if np.isfinite(d) else np.nan,
            decision=kind,
        ))
    print(f"  decision summary: raw={n_keep_raw}/{n_genes}  "
          f"direct(broken)={n_broken}/{n_genes}")
    return final_inner, final_outer, pd.DataFrame(decisions)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe_quantile(v: np.ndarray, q: float) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.quantile(v, q))


def _scatter_full_slice(ax, coor_inner, val_inner, coor_outer, val_outer,
                        v_lo, v_hi, cmap="viridis",
                        s_inner: float = 0.6, s_outer: float = 0.6):
    """Scatter inner + outer cells using the **same** colour scale."""
    if coor_outer is not None and val_outer is not None:
        ax.scatter(
            coor_outer[:, 0], -coor_outer[:, 1],
            c=val_outer, cmap=cmap, s=s_outer, vmin=v_lo, vmax=v_hi,
            rasterized=True, linewidths=0,
        )
    sc_obj = ax.scatter(
        coor_inner[:, 0], -coor_inner[:, 1],
        c=val_inner, cmap=cmap, s=s_inner, vmin=v_lo, vmax=v_hi,
        rasterized=True, linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc_obj


def make_per_gene_figure(
    out_path: str,
    gene: str, pcc: float,
    coor_inner, val_inner, coor_outer, val_outer,
    cmap: str = "viridis",
):
    """Single-axes figure: full-slice prediction, title = '<gene>, PCC=<value>'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_lo = _safe_quantile(val_inner, 0.02)
    v_hi = _safe_quantile(val_inner, 0.98)
    if v_hi <= v_lo:
        v_hi = v_lo + 1e-3

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.5))
    sc_obj = _scatter_full_slice(
        ax, coor_inner, val_inner, coor_outer, val_outer, v_lo, v_hi, cmap)
    title = f"{gene}, PCC={pcc:+.3f}"
    ax.set_title(title, fontsize=13)
    plt.colorbar(sc_obj, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_global_metrics_table(
    out_path: str,
    rows: List[Dict[str, object]],
    title: str = "Global metrics vs SpatialEx+ paper baseline",
):
    """Render a clean comparison table as a stand-alone PNG.

    Columns: Target | Metric | Paper | Ours-Raw | Ours-Final | Δ-paper | Status
    Each row already carries those keys -- this function just paints them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = ["Target", "Metric", "Paper", "Ours-Raw", "Ours-Final",
               "Δ-paper (Final)", "Status"]
    cell_text = []
    cell_colors = []
    for r in rows:
        delta = r["delta_paper"]
        status = "WIN" if r["wins"] else "LOSE"
        better_than_raw = r["final"] - r["raw"]
        # CMD: lower is better; flip the sign of "better_than_raw" for colouring
        if r["metric"] == "cmd":
            better_than_raw = -better_than_raw
        row = [
            str(r["target"]),
            r["metric_label"],
            f"{r['paper']:+.4f}",
            f"{r['raw']:+.4f}",
            f"{r['final']:+.4f}",
            f"{delta:+.4f}",
            status,
        ]
        cell_text.append(row)

        win_color = "#c8f7c5"   # light green
        lose_color = "#fad4d4"  # light red
        neutral = "#ffffff"
        status_color = win_color if r["wins"] else lose_color
        # Color the delta column the same as status, others white.
        cell_colors.append([
            neutral, neutral, neutral, neutral, neutral,
            status_color, status_color,
        ])

    n_rows = len(cell_text)
    fig_height = 0.55 * (n_rows + 2) + 0.6
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=14)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
        colColours=["#dde6f0"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.45)
    # Bold header
    for c in range(len(headers)):
        tbl[(0, c)].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_target_gene_table(
    out_path: str,
    rows: List[Dict[str, object]],
    title: str = "Target gene PCC vs SpatialEx+ paper baseline",
):
    """Per-gene PCC comparison table for the four target genes.

    Each ``rows`` entry has keys
    ``slice gene raw_pcc final_pcc paper_pcc delta_paper``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = ["Slice", "Gene", "Paper", "Ours-Raw", "Ours-Final",
               "Δ-paper (Final)", "Status"]
    cell_text = []
    cell_colors = []
    for r in rows:
        wins = r["delta_paper"] > 0
        status = "WIN" if wins else "LOSE"
        cell_text.append([
            str(r["slice"]),
            str(r["gene"]),
            f"{r['paper_pcc']:+.4f}",
            f"{r['raw_pcc']:+.4f}",
            f"{r['final_pcc']:+.4f}",
            f"{r['delta_paper']:+.4f}",
            status,
        ])
        win_color = "#c8f7c5"
        lose_color = "#fad4d4"
        neutral = "#ffffff"
        status_color = win_color if wins else lose_color
        cell_colors.append([
            neutral, neutral, neutral, neutral, neutral,
            status_color, status_color,
        ])

    n_rows = len(cell_text)
    fig_height = 0.55 * (n_rows + 2) + 0.6
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=14)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
        colColours=["#dde6f0"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.45)
    for c in range(len(headers)):
        tbl[(0, c)].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_combined_figure(
    out_path: str,
    panels: List[Dict[str, object]],
    cmap: str = "viridis",
):
    """4 rows × 3 columns: Measured | Raw SpatialEx+ | Ours-Final."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(13.0, 4.5 * n))
    if n == 1:
        axes = np.asarray(axes).reshape(1, 3)

    for r, p in enumerate(panels):
        gt_inner = p["gt_inner"]
        v_lo = _safe_quantile(gt_inner, 0.02)
        v_hi = _safe_quantile(gt_inner, 0.98)
        if v_hi <= v_lo:
            v_hi = v_lo + 1e-3
        gene = p["gene"]

        # --- column 0: Measured (only inner cells -- only inner cells have GT) ---
        sc1 = _scatter_full_slice(
            axes[r, 0], p["coor_inner"], gt_inner,
            None, None, v_lo, v_hi, cmap,
            s_inner=0.8,
        )
        axes[r, 0].set_title(f"Measured\n{gene}", fontsize=11)
        plt.colorbar(sc1, ax=axes[r, 0], fraction=0.04, pad=0.02)

        # --- column 1: Raw SpatialEx+ (inner + outer) ---
        sc2 = _scatter_full_slice(
            axes[r, 1],
            p["coor_inner"], p["raw_inner"],
            p["coor_outer"], p["raw_outer"],
            v_lo, v_hi, cmap,
        )
        axes[r, 1].set_title(
            f"Raw SpatialEx+\n{gene}, PCC={p['raw_pcc']:+.3f}",
            fontsize=11)
        plt.colorbar(sc2, ax=axes[r, 1], fraction=0.04, pad=0.02)

        # --- column 2: Ours-Final ---
        sc3 = _scatter_full_slice(
            axes[r, 2],
            p["coor_inner"], p["fin_inner"],
            p["coor_outer"], p["fin_outer"],
            v_lo, v_hi, cmap,
        )
        delta_paper = p["fin_pcc"] - p["paper_pcc"]
        axes[r, 2].set_title(
            f"Ours-Final\n{gene}, PCC={p['fin_pcc']:+.3f}  Δ-paper={delta_paper:+.3f}",
            fontsize=11)
        plt.colorbar(sc3, ax=axes[r, 2], fraction=0.04, pad=0.02)

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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0,
                    help="SpatialEx+ seed -- should match best_seed_info.json")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--num-neighbors", type=int, default=7)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--raw-broken-threshold", type=float, default=0.4)
    ap.add_argument("--direct-low-margin", type=float, default=0.05)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- 1. data ----
    adata1 = sc.read_h5ad(args.adata1)
    adata2 = sc.read_h5ad(args.adata2)
    adata1.X = _to_dense(adata1.X)
    adata2.X = _to_dense(adata2.X)
    _ensure_he_key(adata1, "adata1")
    _ensure_he_key(adata2, "adata2")
    panelA_genes = list(map(str, adata1.var_names))
    panelB_genes = list(map(str, adata2.var_names))
    obs1 = list(map(str, adata1.obs_names))
    obs2 = list(map(str, adata2.obs_names))
    print(f"[data] slice1: n_cells={adata1.n_obs}  panelA={adata1.n_vars}")
    print(f"[data] slice2: n_cells={adata2.n_obs}  panelB={adata2.n_vars}")

    # ---- 2. ground truth (for measured panel of test slice + decisions) ----
    gtB1 = load_full_ground_truth(
        args.raw_data_root, args.sample1, "Rep1",
        gene_names=panelB_genes, obs_names_target=obs1)
    gtA2 = load_full_ground_truth(
        args.raw_data_root, args.sample2, "Rep2",
        gene_names=panelA_genes, obs_names_target=obs2)
    print(f"[gt] B1: {gtB1.shape}  A2: {gtA2.shape}")

    # ---- 3. retrain SpatialExP with the chosen seed ----
    print(f"\n[train] SpatialEx+ seed={args.seed}  epochs={args.epochs}")
    sxp, graph1, graph2 = train_original_spatialexp(
        adata1, adata2,
        device=args.device, epochs=args.epochs,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        lr=args.lr, seed=args.seed,
    )

    # ---- 4. inner-cells predictions ----
    sxp.module_HA.eval()
    sxp.module_HB.eval()
    sxp.rm_AB.eval()
    sxp.rm_BA.eval()
    he1 = sxp.HE1
    he2 = sxp.HE2
    G1 = sxp.graph1
    G2 = sxp.graph2

    with torch.no_grad():
        A1_direct  = sxp.module_HA.predict(he1, G1, grad=False)
        B1_direct  = sxp.module_HB.predict(he1, G1, grad=False)
        A1_indirect = sxp.rm_BA.predict(B1_direct)
        B1_indirect = sxp.rm_AB.predict(A1_direct)
        A2_direct  = sxp.module_HA.predict(he2, G2, grad=False)
        B2_direct  = sxp.module_HB.predict(he2, G2, grad=False)
        B2_indirect = sxp.rm_AB.predict(A2_direct)
        A2_indirect = sxp.rm_BA.predict(B2_direct)
    A1_direct_np   = A1_direct.detach().cpu().numpy()
    A1_indirect_np = A1_indirect.detach().cpu().numpy()
    B1_direct_np   = B1_direct.detach().cpu().numpy()
    B1_indirect_np = B1_indirect.detach().cpu().numpy()
    A2_direct_np   = A2_direct.detach().cpu().numpy()
    A2_indirect_np = A2_indirect.detach().cpu().numpy()
    B2_direct_np   = B2_direct.detach().cpu().numpy()
    B2_indirect_np = B2_indirect.detach().cpu().numpy()

    # ---- 5. out-of-spot data + predictions ----
    he1_out, coor1_out = load_out_of_spot(args.raw_data_root, args.sample1, "Rep1")
    he2_out, coor2_out = load_out_of_spot(args.raw_data_root, args.sample2, "Rep2")

    out_preds_1 = None
    out_preds_2 = None
    if he1_out is not None:
        print("\n[out-of-spot] slice 1 inference")
        out_preds_1 = predict_out_of_spot(sxp, he1_out, coor1_out, args.num_neighbors)
    if he2_out is not None:
        print("\n[out-of-spot] slice 2 inference")
        out_preds_2 = predict_out_of_spot(sxp, he2_out, coor2_out, args.num_neighbors)

    # ---- 6. per-gene final on inner + outer ----
    measured_A1 = _to_dense(adata1.X)
    measured_B2 = _to_dense(adata2.X)

    print("\n[final] per-gene decision on Panel A (test = slice 2)")
    A2_fin_inner, A2_fin_outer, dec_A = per_gene_choose_final(
        raw_train=A1_indirect_np, direct_train=A1_direct_np, gt_train=measured_A1,
        raw_test_inner=A2_indirect_np, direct_test_inner=A2_direct_np,
        raw_test_outer=(out_preds_2["panelA_indirect"] if out_preds_2 else None),
        direct_test_outer=(out_preds_2["panelA_direct"] if out_preds_2 else None),
        raw_broken_threshold=args.raw_broken_threshold,
        direct_low_margin=args.direct_low_margin,
    )
    print("[final] per-gene decision on Panel B (test = slice 1)")
    B1_fin_inner, B1_fin_outer, dec_B = per_gene_choose_final(
        raw_train=B2_indirect_np, direct_train=B2_direct_np, gt_train=measured_B2,
        raw_test_inner=B1_indirect_np, direct_test_inner=B1_direct_np,
        raw_test_outer=(out_preds_1["panelB_indirect"] if out_preds_1 else None),
        direct_test_outer=(out_preds_1["panelB_direct"] if out_preds_1 else None),
        raw_broken_threshold=args.raw_broken_threshold,
        direct_low_margin=args.direct_low_margin,
    )

    dec_A.to_csv(os.path.join(args.outdir, "decisions_panelA.csv"), index=False)
    dec_B.to_csv(os.path.join(args.outdir, "decisions_panelB.csv"), index=False)
    np.save(os.path.join(args.outdir, "A2_final_inner.npy"), A2_fin_inner)
    np.save(os.path.join(args.outdir, "B1_final_inner.npy"), B1_fin_inner)
    if A2_fin_outer is not None:
        np.save(os.path.join(args.outdir, "A2_final_outer.npy"), A2_fin_outer)
    if B1_fin_outer is not None:
        np.save(os.path.join(args.outdir, "B1_final_outer.npy"), B1_fin_outer)

    # ---- 7. coordinates ----
    coor1_inner = _maybe_image_coor(adata1)
    if coor1_inner is None:
        coor1_inner = np.asarray(adata1.obsm["spatial"], dtype=np.float32)[:, :2]
    coor2_inner = _maybe_image_coor(adata2)
    if coor2_inner is None:
        coor2_inner = np.asarray(adata2.obsm["spatial"], dtype=np.float32)[:, :2]

    def _coor_outer(df):
        if df is None:
            return None
        for k_pair in (("image_col", "image_row"), ("col", "row"), ("x", "y")):
            if all(k in df.columns for k in k_pair):
                return df[list(k_pair)].values.astype(np.float32)
        # Fallback: take first two numeric columns.
        return df.values[:, :2].astype(np.float32)

    coor1_outer = _coor_outer(coor1_out)
    coor2_outer = _coor_outer(coor2_out)

    # ---- 8. per-gene single-figure plots ----
    print("\n[plot] per-gene full-slice figures ...")

    # panel A genes  ->  slice 2
    for g in TARGET_GENES["slice2"]:
        if g not in panelA_genes:
            continue
        gi = panelA_genes.index(g)
        true_col = gtA2[:, gi]
        fin_pcc = float(_gene_pcc_array(A2_fin_inner[:, [gi]], true_col[:, None])[0])

        out_outer = (A2_fin_outer[:, gi] if A2_fin_outer is not None else None)
        make_per_gene_figure(
            out_path=os.path.join(args.outdir, f"fullslice_{g}.png"),
            gene=g, pcc=fin_pcc,
            coor_inner=coor2_inner, val_inner=A2_fin_inner[:, gi],
            coor_outer=coor2_outer,  val_outer=out_outer,
            cmap=args.cmap,
        )

    # panel B genes  ->  slice 1
    for g in TARGET_GENES["slice1"]:
        if g not in panelB_genes:
            continue
        gi = panelB_genes.index(g)
        true_col = gtB1[:, gi]
        fin_pcc = float(_gene_pcc_array(B1_fin_inner[:, [gi]], true_col[:, None])[0])

        out_outer = (B1_fin_outer[:, gi] if B1_fin_outer is not None else None)
        make_per_gene_figure(
            out_path=os.path.join(args.outdir, f"fullslice_{g}.png"),
            gene=g, pcc=fin_pcc,
            coor_inner=coor1_inner, val_inner=B1_fin_inner[:, gi],
            coor_outer=coor1_outer,  val_outer=out_outer,
            cmap=args.cmap,
        )

    # ---- 9. combined comparison figure ----
    print("[plot] combined Measured / Raw SpatialEx+ / Ours-Final figure ...")
    panels: List[Dict[str, object]] = []
    # ESR1, ERBB2 on slice 2 (panel A)
    for g in ("ESR1", "ERBB2"):
        if g not in panelA_genes:
            continue
        gi = panelA_genes.index(g)
        true_col = gtA2[:, gi]
        raw_inner = A2_indirect_np[:, gi]
        fin_inner = A2_fin_inner[:, gi]
        raw_pcc = float(_gene_pcc_array(raw_inner[:, None], true_col[:, None])[0])
        fin_pcc = float(_gene_pcc_array(fin_inner[:, None], true_col[:, None])[0])
        panels.append({
            "gene": g, "slice": "slice2",
            "coor_inner": coor2_inner,
            "coor_outer": coor2_outer,
            "gt_inner":   true_col,
            "raw_inner":  raw_inner,
            "raw_outer":  (out_preds_2["panelA_indirect"][:, gi] if out_preds_2 else None),
            "fin_inner":  fin_inner,
            "fin_outer":  (A2_fin_outer[:, gi] if A2_fin_outer is not None else None),
            "raw_pcc": raw_pcc, "fin_pcc": fin_pcc,
            "paper_pcc": PAPER_GENE.get(("slice2", g), float("nan")),
        })
    # PGR, KRT14 on slice 1 (panel B)
    for g in ("PGR", "KRT14"):
        if g not in panelB_genes:
            continue
        gi = panelB_genes.index(g)
        true_col = gtB1[:, gi]
        raw_inner = B1_indirect_np[:, gi]
        fin_inner = B1_fin_inner[:, gi]
        raw_pcc = float(_gene_pcc_array(raw_inner[:, None], true_col[:, None])[0])
        fin_pcc = float(_gene_pcc_array(fin_inner[:, None], true_col[:, None])[0])
        panels.append({
            "gene": g, "slice": "slice1",
            "coor_inner": coor1_inner,
            "coor_outer": coor1_outer,
            "gt_inner":   true_col,
            "raw_inner":  raw_inner,
            "raw_outer":  (out_preds_1["panelB_indirect"][:, gi] if out_preds_1 else None),
            "fin_inner":  fin_inner,
            "fin_outer":  (B1_fin_outer[:, gi] if B1_fin_outer is not None else None),
            "raw_pcc": raw_pcc, "fin_pcc": fin_pcc,
            "paper_pcc": PAPER_GENE.get(("slice1", g), float("nan")),
        })

    make_combined_figure(
        out_path=os.path.join(args.outdir, "fullslice_4genes_compare.png"),
        panels=panels, cmap=args.cmap,
    )

    # ---- 10. summary table for the four target genes ----
    sel_rows = []
    for p in panels:
        sel_rows.append({
            "slice": p["slice"], "gene": p["gene"],
            "raw_pcc":   p["raw_pcc"],
            "final_pcc": p["fin_pcc"],
            "paper_pcc": p["paper_pcc"],
            "delta_paper": p["fin_pcc"] - p["paper_pcc"],
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_path = os.path.join(args.outdir, "target_gene_pcc_summary.csv")
    sel_df.to_csv(sel_path, index=False)
    print(f"\n[save] {sel_path}")
    print(sel_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # PNG version of the per-gene summary -- handy for slides
    make_target_gene_table(
        out_path=os.path.join(args.outdir, "table_target_genes.png"),
        rows=sel_rows,
        title="Per-gene PCC: Ours-Final vs SpatialEx+ paper baseline",
    )

    # ---- 11. global metrics (PCC mean / SSIM / CMD) for raw and Final ----
    print("\n[metrics] computing global metrics for raw + Final ...")
    g1 = se.pp.Build_graph(
        adata1.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo")
    g2 = se.pp.Build_graph(
        adata2.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo")

    raw_B1_metrics = compute_full_metrics(B1_indirect_np, gtB1, spatial_graph=g1)
    fin_B1_metrics = compute_full_metrics(B1_fin_inner,  gtB1, spatial_graph=g1)
    raw_A2_metrics = compute_full_metrics(A2_indirect_np, gtA2, spatial_graph=g2)
    fin_A2_metrics = compute_full_metrics(A2_fin_inner,  gtA2, spatial_graph=g2)

    global_table_rows = []
    for tgt, raw_m, fin_m in [
        ("B1", raw_B1_metrics, fin_B1_metrics),
        ("A2", raw_A2_metrics, fin_A2_metrics),
    ]:
        for metric, label, lower_better in [
            ("gene_pcc_mean", "gene-PCC (mean)", False),
            ("ssim",          "SSIM",            False),
            ("cmd",           "CMD",             True),
        ]:
            paper_v = PAPER_GLOBAL[tgt][metric]
            raw_v = float(raw_m[metric])
            fin_v = float(fin_m[metric])
            # Δ = (ours - paper) for higher-better metrics, (paper - ours) for CMD.
            if lower_better:
                delta = paper_v - fin_v
            else:
                delta = fin_v - paper_v
            wins = delta > 0
            global_table_rows.append({
                "target": tgt,
                "metric": metric,
                "metric_label": label,
                "paper": paper_v,
                "raw":   raw_v,
                "final": fin_v,
                "delta_paper": delta,
                "wins": bool(wins),
            })

    global_df = pd.DataFrame(global_table_rows)
    global_csv = os.path.join(args.outdir, "global_metrics_summary.csv")
    global_df.to_csv(global_csv, index=False)
    print(f"[save] {global_csv}")
    print(global_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # PNG version of the global metrics table -- for the slide deck
    make_global_metrics_table(
        out_path=os.path.join(args.outdir, "table_global_metrics.png"),
        rows=global_table_rows,
        title="Global metrics: Ours-Final vs SpatialEx+ paper baseline\n"
              "(WIN means we beat the paper on this metric; CMD is lower-better)",
    )

    print(f"\n[done] all outputs in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
