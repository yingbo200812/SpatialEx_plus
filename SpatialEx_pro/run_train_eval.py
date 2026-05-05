"""
SpatialEx-pro 端到端训练与评估 CLI 入口。

复现原论文 Fig. 2c/d 的评估设置（HBC Rep1/Rep2，Choice 1：两片共享
同一 313 基因 panel，leave-one-slice-out 跨片预测），并可在同一份数据
分割上同时跑 SpatialEx baseline 做严格的并排对比。

默认数据路径（与远程服务器目录对齐）::

    /data1/linxin/1/SpatialEx/data/
    ├── Human_breast_cancer_small/Human_breast_cancer_small/
    │   ├── Human_Breast_Cancer_Rep1/Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad
    │   └── Human_Breast_Cancer_Rep2/Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad
    └── raw_data/HBC/
        └── Human_Breast_Cancer_Rep{1,2}/{cell_feature_matrix.h5, cells.csv, ...}

典型远程用法（推荐用一键脚本 ``SpatialEx-pro.sh``）::

    python SpatialEx_pro/run_train_eval.py \\
        --device cuda:0 --epochs 500 --n-seeds 3 \\
        --outdir /data1/linxin/1/SpatialEx_pro/results/run_default \\
        --run-baseline

如果 ``*_uni_resolution64_full.h5ad`` 缓存不存在，会自动回退到从原始
Xenium 文件 (``--raw-data-root``) 跑完整预处理流水线（首次较慢，结果
会缓存供后续直接复用）。

注：脚本所在目录被加入 ``sys.path``，因此即便用户把外层文件夹改名为
``SpatialEx_pro1`` / ``SpatialEx-pro`` 也能正确 import。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

# 把本脚本所在目录与仓库根都放进 sys.path：
#   * 仓库根：让 ``import SpatialEx`` (sibling baseline package) 可用；
#   * 本目录：让本包内的模块可以用 ``from utils import ...`` 直接 import，
#     从而对外层文件夹名 (SpatialEx_pro / SpatialEx_pro1 / SpatialEx-pro) 不敏感。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import SpatialEx as se  # noqa: E402

# 这里走"目录内裸 import"，找的是 ``_THIS_DIR/utils.py`` 与
# ``_THIS_DIR/SpatialEx_pro.py``——文件级 import，不依赖外层包名。
from utils import (  # noqa: E402
    SpatialExProConfig,
    evaluate_two_directions,
    per_gene_pcc_table,
    side_by_side,
)
from SpatialEx_pro import SpatialExPro  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Default paths (match the remote server layout)
# ---------------------------------------------------------------------------

DEFAULT_H5AD_ROOT = (
    "/data1/linxin/1/SpatialEx/data/Human_breast_cancer_small/"
    "Human_breast_cancer_small"
)
DEFAULT_RAW_ROOT = "/data1/linxin/1/SpatialEx/data/raw_data/HBC"

_FULL_H5AD_CANDIDATES = {
    "Rep1": [
        "Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad",
        "adata_preprocessed_uni_res64.h5ad",
        "adata_panelA_uni_res64.h5ad",
    ],
    "Rep2": [
        "Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad",
        "adata_preprocessed_uni_res64.h5ad",
        "adata_panelB_uni_res64.h5ad",
    ],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _to_dense_X(adata) -> None:
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = np.ascontiguousarray(np.asarray(adata.X, dtype=np.float32))


def _ensure_he_key(adata, name: str) -> None:
    """Make sure adata.obsm['he'] exists; try common alternate key names."""
    if "he" in adata.obsm:
        return
    for alt in ("X_uni", "uni", "UNI", "X_he", "image_features"):
        if alt in adata.obsm:
            print(f"[{name}] renaming obsm['{alt}'] -> obsm['he']")
            adata.obsm["he"] = np.asarray(adata.obsm[alt])
            return
    raise KeyError(
        f"[{name}] no UNI/H&E embedding found in obsm. "
        f"Available keys: {list(adata.obsm.keys())}"
    )


def _ensure_spatial_key(adata, name: str) -> None:
    if "spatial" in adata.obsm:
        return
    if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
        return
    raise KeyError(
        f"[{name}] no 'spatial' obsm and no x_centroid/y_centroid columns; "
        "cannot build spatial graph."
    )


def _find_h5ad(rep_dir: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        p = os.path.join(rep_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _preprocess_from_raw(
    raw_data_root: str,
    rep_tag: str,
    cache_path: str,
    resolution: int = 64,
    device: str = "cuda:0",
) -> "sc.AnnData":
    """Run the SpatialEx preprocessing pipeline from raw Xenium files.

    Mirrors Tutorial 1's preprocessing exactly: ``Read_Xenium ->
    Preprocess_adata -> Read_HE_image -> Register_physical_to_pixel ->
    Tiling_HE_patches -> Extract_HE_patches_representaion(uni)``.
    """
    rep_dir = os.path.join(raw_data_root, f"Human_Breast_Cancer_{rep_tag}")
    h5_path = os.path.join(rep_dir, "cell_feature_matrix.h5")
    obs_path = os.path.join(rep_dir, "cells.csv")
    img_path = os.path.join(rep_dir, f"Xenium_FFPE_Human_Breast_Cancer_{rep_tag}_he_image.ome.tif")
    align_path = os.path.join(rep_dir, f"Xenium_FFPE_Human_Breast_Cancer_{rep_tag}_he_imagealignment.csv")

    for p in (h5_path, obs_path, img_path, align_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Raw preprocessing requested but file not found: {p}\n"
                f"Either run Tutorial-1 to pre-build the cache, or pass --adata{rep_tag[-1]} "
                f"explicitly."
            )

    print(f"[preprocess] {rep_tag} -- running from raw (this may take ~30 min) ...")
    adata = se.pp.Read_Xenium(h5_path, obs_path)
    adata = se.pp.Preprocess_adata(adata)
    img, scale = se.pp.Read_HE_image(img_path)
    transform_mtx = pd.read_csv(align_path, header=None).values
    adata = se.pp.Register_physical_to_pixel(adata, transform_mtx, scale=scale)
    he_patches, adata = se.pp.Tiling_HE_patches(resolution, adata, img)
    adata = se.pp.Extract_HE_patches_representaion(
        he_patches, store_key="he", adata=adata, image_encoder="uni", device=device,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    adata.write_h5ad(cache_path)
    print(f"[preprocess] cache saved: {cache_path}")
    return adata


def load_two_slices(
    h5ad_root: str,
    raw_data_root: str,
    adata1_path: Optional[str] = None,
    adata2_path: Optional[str] = None,
    device: str = "cuda:0",
    resolution: int = 64,
) -> Tuple["sc.AnnData", "sc.AnnData"]:
    """Locate (or generate) the two cached h5ad files and load them.

    Search order, per slice:

    1. Explicit ``--adata1`` / ``--adata2`` argument.
    2. ``<h5ad_root>/Human_Breast_Cancer_Rep{i}/<full_or_preprocessed>.h5ad``.
    3. ``<raw_data_root>/Human_Breast_Cancer_Rep{i}/<full_or_preprocessed>.h5ad``.
    4. Fall back to raw preprocessing (only if Tutorial-1 raw files exist
       at ``<raw_data_root>/Human_Breast_Cancer_Rep{i}/``).
    """

    def _locate(rep_tag: str, override: Optional[str]) -> str:
        if override is not None:
            if not os.path.isfile(override):
                raise FileNotFoundError(f"--adata{rep_tag[-1]} not found: {override}")
            return override
        for root in (h5ad_root, raw_data_root):
            rep_dir = os.path.join(root, f"Human_Breast_Cancer_{rep_tag}")
            if os.path.isdir(rep_dir):
                hit = _find_h5ad(rep_dir, _FULL_H5AD_CANDIDATES[rep_tag])
                if hit is not None:
                    return hit
        cache_path = os.path.join(
            raw_data_root, f"Human_Breast_Cancer_{rep_tag}",
            f"adata_preprocessed_uni_res{resolution}.h5ad",
        )
        adata = _preprocess_from_raw(raw_data_root, rep_tag, cache_path,
                                     resolution=resolution, device=device)
        return cache_path

    p1 = _locate("Rep1", adata1_path)
    p2 = _locate("Rep2", adata2_path)
    print(f"[data] Slice 1 h5ad: {p1}")
    print(f"[data] Slice 2 h5ad: {p2}")
    adata1 = sc.read_h5ad(p1)
    adata2 = sc.read_h5ad(p2)
    _to_dense_X(adata1)
    _to_dense_X(adata2)
    _ensure_he_key(adata1, "Rep1")
    _ensure_he_key(adata2, "Rep2")
    _ensure_spatial_key(adata1, "Rep1")
    _ensure_spatial_key(adata2, "Rep2")

    # Align gene panels (intersect var_names if they differ).
    v1 = list(map(str, adata1.var_names))
    v2 = list(map(str, adata2.var_names))
    if v1 != v2:
        common = [g for g in v1 if g in set(v2)]
        if len(common) < 50:
            raise ValueError(
                f"Slices share only {len(common)} genes -- something is wrong with the "
                "preprocessing.  SpatialEx-pro's Fig. 2 setup expects the same panel."
            )
        print(f"[data] var_names differ; intersecting to {len(common)} common genes")
        adata1 = adata1[:, common].copy()
        adata2 = adata2[:, common].copy()
        _to_dense_X(adata1)
        _to_dense_X(adata2)

    print(f"[data] Slice 1: n_cells={adata1.n_obs}, n_genes={adata1.n_vars}, "
          f"he={adata1.obsm['he'].shape}")
    print(f"[data] Slice 2: n_cells={adata2.n_obs}, n_genes={adata2.n_vars}, "
          f"he={adata2.obsm['he'].shape}")
    return adata1, adata2


# ---------------------------------------------------------------------------
# Baseline (upstream SpatialEx)
# ---------------------------------------------------------------------------

def run_baseline_spatialex(
    adata1, adata2, graph1, graph2, device: str, epochs: int, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-train the upstream SpatialEx baseline for a faithful side-by-side."""
    print(f"\n=========================== Baseline SpatialEx (seed={seed}) ===========================")
    sx = se.SpatialEx(
        adata1, adata2, graph1, graph2,
        device=device, epochs=epochs, seed=seed,
    )
    sx.train()
    panelB1, panelA2 = sx.auto_inference()
    return panelB1, panelA2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_csv_list(s: str) -> List[str]:
    return [g.strip() for g in s.split(",") if g.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="SpatialEx-pro: improved cross-slice H&E -> single-cell expression "
                    "translator (Fig. 2c/d task).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ----- data -----
    ap.add_argument("--h5ad-root", default=DEFAULT_H5AD_ROOT,
                    help="Root containing Human_Breast_Cancer_Rep{1,2}/*_full.h5ad")
    ap.add_argument("--raw-data-root", default=DEFAULT_RAW_ROOT,
                    help="Fallback root with raw cell_feature_matrix.h5 + cells.csv")
    ap.add_argument("--adata1", default=None, help="Direct path to Slice 1 h5ad")
    ap.add_argument("--adata2", default=None, help="Direct path to Slice 2 h5ad")
    ap.add_argument("--resolution", type=int, default=64)

    # ----- training -----
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="If >1, train this many models with different seeds and "
                         "average the cross-slice predictions (ensembling).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-cosine-lr", action="store_true",
                    help="Disable cosine LR schedule (uses constant LR).")
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-share-projection", action="store_true",
                    help="Disable cross-slice MLP sharing (ablation).")
    ap.add_argument("--no-dgi", action="store_true")

    # ----- supervised loss weights -----
    ap.add_argument("--lambda-mse-cell", type=float, default=1.0)
    ap.add_argument("--lambda-mse-spot", type=float, default=0.5)
    ap.add_argument("--lambda-pearson", type=float, default=1.0)

    # ----- cross-slice anchor -----
    ap.add_argument("--lambda-anchor-mse", type=float, default=0.5)
    ap.add_argument("--lambda-anchor-pearson", type=float, default=0.3)
    ap.add_argument("--lambda-anchor-warmup", type=int, default=80)
    ap.add_argument("--anchor-k", type=int, default=8)
    ap.add_argument("--anchor-sim-floor", type=float, default=0.3)
    ap.add_argument("--use-mnn-anchors", action="store_true")

    # ----- TV / DGI -----
    ap.add_argument("--lambda-spatial-tv", type=float, default=0.02)
    ap.add_argument("--tv-max-edges", type=int, default=200_000)
    ap.add_argument("--lambda-dgi", type=float, default=0.3)

    # ----- gene-gene CMD alignment -----
    ap.add_argument("--lambda-cmd-align", type=float, default=0.5,
                    help="Weight of the gene-gene correlation matrix "
                         "alignment loss (directly minimises CMD).")
    ap.add_argument("--cmd-align-subsample", type=int, default=30000,
                    help="Sub-sample this many cells per slice when "
                         "computing the (genes, genes) corr matrix; 0 to "
                         "use all cells.")

    # ----- gene weighting -----
    ap.add_argument("--no-invstd-weighting", action="store_true")
    ap.add_argument("--marker-genes", type=str,
                    default="EPCAM,ESR1,PGR,ERBB2,KRT14")
    ap.add_argument("--marker-weight", type=float, default=2.0)
    ap.add_argument("--marker-pearson-weight", type=float, default=3.0)

    # ----- inference -----
    ap.add_argument("--alpha-spatial", type=float, default=0.1)
    ap.add_argument("--beta-anchor", type=float, default=0.1)
    ap.add_argument("--refine-anchor-k", type=int, default=15)

    # ----- bookkeeping -----
    ap.add_argument("--num-neighbors", type=int, default=7,
                    help="K for the spatial KNN hypergraph (matches baseline).")
    ap.add_argument("--outdir", default="/data1/linxin/1/SpatialEx_pro/results/run_default")
    ap.add_argument("--run-baseline", action="store_true",
                    help="Also run the upstream SpatialEx baseline for a side-by-side.")
    ap.add_argument("--baseline-epochs", type=int, default=None,
                    help="Override epochs for the baseline run only.")
    ap.add_argument("--config-json", default=None,
                    help="If given, load all hyper-params from a previous "
                         "config.json (overrides everything else).")

    args = ap.parse_args(argv)

    # If a previous config.json is given, replay it on top of CLI args
    # (CLI flags appear before --config-json in the env when both are set;
    # the json is the ground truth for reproducibility).
    if args.config_json is not None and os.path.isfile(args.config_json):
        with open(args.config_json) as fh:
            saved = json.load(fh)
        print(f"[config] loaded {args.config_json}")
        for k, v in saved.items():
            if hasattr(args, k):
                setattr(args, k, v)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[device] using {device}")
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    adata1, adata2 = load_two_slices(
        h5ad_root=args.h5ad_root,
        raw_data_root=args.raw_data_root,
        adata1_path=args.adata1,
        adata2_path=args.adata2,
        device=device,
        resolution=args.resolution,
    )

    # ------------------------------------------------------------------
    # Hypergraphs
    # ------------------------------------------------------------------
    print("[graph] building spatial hypergraphs (knn, k=", args.num_neighbors, ") ...")
    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        adata1, args.num_neighbors, graph_kind="spatial", return_type="crs",
    )
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        adata2, args.num_neighbors, graph_kind="spatial", return_type="crs",
    )

    # ------------------------------------------------------------------
    # SpatialEx-pro (multi-seed ensemble when --n-seeds > 1)
    # ------------------------------------------------------------------
    seeds_to_run = [args.seed + i for i in range(max(1, args.n_seeds))]
    panelB1_runs: List[np.ndarray] = []
    panelA2_runs: List[np.ndarray] = []
    for run_idx, seed in enumerate(seeds_to_run):
        print(f"\n##################  RUN {run_idx + 1}/{len(seeds_to_run)} (seed={seed})  ##################")
        cfg = SpatialExProConfig(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            share_projection=not args.no_share_projection,
            use_dgi=not args.no_dgi,
            lr=args.lr,
            use_cosine_lr=not args.no_cosine_lr,
            lr_min_ratio=args.lr_min_ratio,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seed=seed,
            lambda_mse_cell=args.lambda_mse_cell,
            lambda_mse_spot=args.lambda_mse_spot,
            lambda_pearson=args.lambda_pearson,
            lambda_anchor_mse=args.lambda_anchor_mse,
            lambda_anchor_pearson=args.lambda_anchor_pearson,
            lambda_anchor_warmup=args.lambda_anchor_warmup,
            anchor_k=args.anchor_k,
            anchor_sim_floor=args.anchor_sim_floor,
            use_mnn_anchors=args.use_mnn_anchors,
            lambda_spatial_tv=args.lambda_spatial_tv,
            tv_max_edges=args.tv_max_edges,
            lambda_dgi=args.lambda_dgi,
            lambda_cmd_align=args.lambda_cmd_align,
            cmd_align_subsample=args.cmd_align_subsample,
            use_invstd_weighting=not args.no_invstd_weighting,
            marker_genes=_parse_csv_list(args.marker_genes),
            marker_weight=args.marker_weight,
            marker_pearson_weight=args.marker_pearson_weight,
            alpha_spatial=args.alpha_spatial,
            beta_anchor=args.beta_anchor,
            refine_anchor_k=args.refine_anchor_k,
            device=device,
        )
        if run_idx == 0:
            print("[cfg]", cfg.to_dict())
        run_save = os.path.join(args.outdir, "predictions", f"seed{seed}")
        trainer = SpatialExPro(
            adata1, adata2, graph1, graph2, cfg=cfg,
            device=device, save_path=run_save,
        )
        trainer.train()
        b1, a2 = trainer.auto_inference()
        panelB1_runs.append(b1)
        panelA2_runs.append(a2)
        del trainer
        torch.cuda.empty_cache()

    panelB1 = np.mean(np.stack(panelB1_runs, axis=0), axis=0).astype(np.float32)
    panelA2 = np.mean(np.stack(panelA2_runs, axis=0), axis=0).astype(np.float32)
    pred_dir = os.path.join(args.outdir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    if len(panelB1_runs) > 1:
        np.save(os.path.join(pred_dir, "panelB1_ensemble.npy"), panelB1)
        np.save(os.path.join(pred_dir, "panelA2_ensemble.npy"), panelA2)
        print(f"[ensemble] averaged {len(panelB1_runs)} seeds")

    metrics_pro = evaluate_two_directions(
        panelB1, panelA2, adata1, adata2,
        label="SpatialEx-pro",
        out_csv=os.path.join(args.outdir, "metrics_spatialex_pro.csv"),
    )

    # Also: per-gene PCC table.
    per_gene_pcc_table(
        panelB1, panelA2, adata1, adata2,
        out_csv=os.path.join(args.outdir, "per_gene_pcc.csv"),
    )

    # Save config snapshot.
    with open(os.path.join(args.outdir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    # ------------------------------------------------------------------
    # Optional: baseline SpatialEx for side-by-side.
    # ------------------------------------------------------------------
    if args.run_baseline:
        bl_epochs = args.baseline_epochs or args.epochs
        # Build hypergraphs once more in the format the baseline expects (same).
        panelB1_bl, panelA2_bl = run_baseline_spatialex(
            adata1, adata2, graph1, graph2,
            device=device, epochs=bl_epochs, seed=args.seed,
        )
        # Save the baseline predictions too.
        np.save(os.path.join(pred_dir, "panelB1_baseline.npy"), panelB1_bl)
        np.save(os.path.join(pred_dir, "panelA2_baseline.npy"), panelA2_bl)

        metrics_bl = evaluate_two_directions(
            panelB1_bl, panelA2_bl, adata1, adata2,
            label="SpatialEx (baseline)",
            out_csv=os.path.join(args.outdir, "metrics_baseline.csv"),
        )

        df_cmp = side_by_side(
            metrics_pro, metrics_bl,
            out_csv=os.path.join(args.outdir, "compare.csv"),
        )

        print("\n=========================== Side-by-side ===========================")
        print(df_cmp.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print(f"[saved] {os.path.join(args.outdir, 'compare.csv')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
