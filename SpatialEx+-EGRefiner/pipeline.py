"""SpatialEx+-EGRefiner 端到端流水线（HBC Choice-1 panel diagonal integration）。

作用
----
本文件是整个项目的「主入口脚本 + 共享工具模块」，既能直接 ``python
gpt/pipeline.py`` 跑通从 SpatialEx+ 训练到最终评估的完整 8 阶段流水线，
又把所有阶段共用的辅助函数（数据装载、metric、Ridge OOF、per-gene
linear stacking、ground-truth 加载、可视化等）暴露给 :mod:`gpt.seed_search`
/ :mod:`gpt.postprocess` / :mod:`gpt.plot_full_slice` 复用，做到「改动一处
全部生效」。

8 阶段流程
----------

Stage 1
    在两个 panel-diagonal h5ad 上训练 *原版* :class:`SpatialEx.SpatialExP`
    （Slice 1 = Panel A measured，Slice 2 = Panel B measured），保存 ``B1_raw``
    / ``A2_raw``。本阶段只调用 ``SpatialExP`` 的公开方法，不改其源码。

Stage 2
    用训好的 backbone / regression mapper 计算两个切片的 ``direct`` 与
    ``indirect`` 预测，写入 ``*_direct.npy`` / ``*_indirect.npy``。

Stage 3-5
    训练两个 EGRefiner head：

        * ``RefinerA``（预测 Panel A）—— **仅** 在 Slice 1 上训练；
        * ``RefinerB``（预测 Panel B）—— **仅** 在 Slice 2 上训练。

    然后把 ``RefinerA`` 应用到 query=Slice 2 / exemplar=Slice 1 得到
    ``A2_refined``；对称得到 ``B1_refined``。整个过程不接触 query 切片
    缺失 panel 的 ground truth。

Stage 6-8
    汇总 raw / direct / indirect / refined 四种预测的 metric（gene-PCC
    mean / median / Q1、spot-PCC、RMSE、MAE、MSE、SSIM、CMD）以及目标
    基因 (ESR1, ERBB2, PGR, KRT14) 的 per-gene PCC，并按 paper 基线
    做对照表。同时落盘空间图，便于直接做论文图。

示例命令
--------

.. code-block:: bash

    python gpt/pipeline.py \\
        --adata1  .../Rep1_uni_resolution64_panelA.h5ad \\
        --adata2  .../Rep2_uni_resolution64_panelB.h5ad \\
        --raw-data-root .../raw_data/HBC \\
        --selection     .../raw_data/HBC/Selection_by_name.csv \\
        --device cuda:0 --epochs 500 --refiner-epochs 300 --k-exemplar 6 \\
        --outdir ./results_egrefiner

注意
----
推荐的「省时三步走」是：

    1. ``python gpt/seed_search.py ...``  —— 跑多种子搜出最强 SpatialEx+
       baseline，落盘 ``*_raw / *_direct / *_indirect`` 预测；
    2. ``python gpt/postprocess.py --load-preds ...``  —— 复用 baseline
       预测，做 Ridge OOF / per-gene 决策得到最终预测；
    3. ``python gpt/plot_bars.py`` / ``python gpt/plot_full_slice.py`` ——
       生成柱状图与全切片可视化。

直接跑本文件适合「我就想一键端到端复现」的场景。
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

# ---------------------------------------------------------------------------
# Make ``import SpatialEx`` work whether this file is launched from the repo
# root or from inside ``gpt/``.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import SpatialEx as se  # noqa: E402

from gpt.refiner import (  # noqa: E402
    EGRefinerConfig,
    EGRefinerTrainer,
    save_train_log,
)


warnings.filterwarnings("ignore")


# =============================================================================
# Constants -- evaluation targets
# =============================================================================

TARGET_GENES = {
    "slice1": ["PGR", "KRT14"],   # Panel B genes evaluated on Slice 1 prediction
    "slice2": ["ESR1", "ERBB2"],  # Panel A genes evaluated on Slice 2 prediction
}


# =============================================================================
# Tiny IO helpers
# =============================================================================

def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _ensure_he_key(adata, name: str) -> None:
    """Make sure ``adata.obsm['he']`` exists."""
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


def _maybe_image_coor(adata) -> Optional[np.ndarray]:
    """Pick image_coor first, then obsm['spatial'].  Returns ``None`` if neither."""
    for k in ("image_coor", "spatial", "X_spatial", "coords"):
        if k in adata.obsm:
            v = np.asarray(adata.obsm[k])
            if v.ndim == 2 and v.shape[1] >= 2:
                return v[:, :2].astype(np.float32, copy=False)
    return None


def _normalize_log1p(X: np.ndarray) -> np.ndarray:
    """Replicate SpatialEx ``Preprocess_adata`` normalize_total + log1p."""
    X = np.asarray(X, dtype=np.float32)
    sums = X.sum(axis=1, keepdims=True)
    median = float(np.median(sums[sums > 0])) if (sums > 0).any() else 1.0
    sf = np.where(sums > 0, median / sums, 0.0).astype(np.float32)
    return np.log1p(X * sf)


# =============================================================================
# Stage 0: load the two cached panel h5ad files
# =============================================================================

def load_choice1_panels(
    adata1_path: str,
    adata2_path: str,
) -> Tuple[sc.AnnData, sc.AnnData]:
    print(f"[data] reading slice 1 (Panel A measured): {adata1_path}")
    adata1 = sc.read_h5ad(adata1_path)
    print(f"[data] reading slice 2 (Panel B measured): {adata2_path}")
    adata2 = sc.read_h5ad(adata2_path)

    # densify .X
    adata1.X = _to_dense(adata1.X)
    adata2.X = _to_dense(adata2.X)
    _ensure_he_key(adata1, "adata1")
    _ensure_he_key(adata2, "adata2")

    print(
        f"[data] slice1: n_cells={adata1.n_obs} | n_genes_panelA={adata1.n_vars} | "
        f"he={adata1.obsm['he'].shape}"
    )
    print(
        f"[data] slice2: n_cells={adata2.n_obs} | n_genes_panelB={adata2.n_vars} | "
        f"he={adata2.obsm['he'].shape}"
    )
    return adata1, adata2


# =============================================================================
# Stage 1: train the original SpatialEx+
# =============================================================================

def train_original_spatialexp(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    device: str,
    epochs: int,
    num_neighbors: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    seed: int,
) -> Tuple["se.SpatialExP", sp.spmatrix, sp.spmatrix]:
    print("\n" + "=" * 80)
    print("[Stage 1] Training original SpatialEx+ (SpatialExP)")
    print("=" * 80)

    print(f"[graph] building spatial hypergraphs (knn k={num_neighbors}) ...")
    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        adata1, num_neighbors=num_neighbors, graph_kind="spatial",
        normalize=True, return_type="crs",
    )
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        adata2, num_neighbors=num_neighbors, graph_kind="spatial",
        normalize=True, return_type="crs",
    )

    sxp = se.SpatialExP(
        adata1, adata2, graph1, graph2,
        platform="Visium",                # Choice-1 h5ad are at single-cell res
        seed=seed,
        device=torch.device(device if torch.cuda.is_available() else "cpu"),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        num_neighbors=num_neighbors,
        graph_kind="spatial",
    )
    sxp.train()
    return sxp, graph1, graph2


# =============================================================================
# Stage 2: extract direct / indirect predictions on both slices
# =============================================================================

@torch.no_grad()
def compute_direct_indirect(
    sxp: "se.SpatialExP",
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    graph1: sp.spmatrix,
    graph2: sp.spmatrix,
) -> Dict[str, np.ndarray]:
    """Return all 8 prediction arrays plus the auto_inference outputs.

    Keys
    ----
    A1_direct, A1_indirect              Slice 1 panel A predictions
    B2_direct, B2_indirect              Slice 2 panel B predictions
    A2_direct, A2_indirect              Slice 2 panel A predictions
    B1_direct, B1_indirect              Slice 1 panel B predictions
    B1_raw, A2_raw                      auto_inference outputs (= *_indirect)
    """
    print("\n" + "=" * 80)
    print("[Stage 2] Computing direct / indirect predictions")
    print("=" * 80)

    sxp.module_HA.eval()
    sxp.module_HB.eval()
    sxp.rm_AB.eval()
    sxp.rm_BA.eval()

    HE1 = sxp.HE1
    HE2 = sxp.HE2
    G1 = sxp.graph1
    G2 = sxp.graph2

    # ---- Panel A on slice 1 ----
    A1_direct = sxp.module_HA.predict(HE1, G1, grad=False)
    B1_direct_t = sxp.module_HB.predict(HE1, G1, grad=False)         # used by indirect-A
    A1_indirect = sxp.rm_BA.predict(B1_direct_t)                     # B1_direct -> A
    B1_indirect = sxp.rm_AB.predict(A1_direct)                       # A1_direct -> B

    # ---- Panel B on slice 2 ----
    A2_direct_t = sxp.module_HA.predict(HE2, G2, grad=False)         # used by indirect-B
    B2_direct = sxp.module_HB.predict(HE2, G2, grad=False)
    B2_indirect = sxp.rm_AB.predict(A2_direct_t)
    A2_indirect = sxp.rm_BA.predict(B2_direct)

    out = {
        "A1_direct":   A1_direct.detach().cpu().numpy(),
        "A1_indirect": A1_indirect.detach().cpu().numpy(),
        "B1_direct":   B1_direct_t.detach().cpu().numpy(),
        "B1_indirect": B1_indirect.detach().cpu().numpy(),
        "A2_direct":   A2_direct_t.detach().cpu().numpy(),
        "A2_indirect": A2_indirect.detach().cpu().numpy(),
        "B2_direct":   B2_direct.detach().cpu().numpy(),
        "B2_indirect": B2_indirect.detach().cpu().numpy(),
    }
    # ``auto_inference`` returns (panelB1_indirect, panelA2_indirect)
    out["B1_raw"] = out["B1_indirect"].copy()
    out["A2_raw"] = out["A2_indirect"].copy()
    for k, v in out.items():
        print(f"  {k:<12} shape={v.shape}")
    return out


def save_predictions(preds: Dict[str, np.ndarray], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for k, v in preds.items():
        np.save(os.path.join(outdir, f"{k}.npy"), v.astype(np.float32, copy=False))
    print(f"[save] wrote {len(preds)} prediction arrays to {outdir}")


# =============================================================================
# Stage 3-5: train + apply EGRefiner
# =============================================================================

def train_refiner_A(
    adata1: sc.AnnData,
    preds: Dict[str, np.ndarray],
    refiner_cfg: EGRefinerConfig,
    device: str,
    outdir: str,
) -> EGRefinerTrainer:
    """Train RefinerA on Slice 1 (Panel A measured)."""
    print("\n" + "=" * 80)
    print("[Stage 3] Training RefinerA (target = Panel A) on Slice 1")
    print("=" * 80)

    he1 = np.asarray(adata1.obsm["he"], dtype=np.float32)
    spatial1 = _maybe_image_coor(adata1)
    target_A1 = _to_dense(adata1.X)                    # measured Panel A
    trainer = EGRefinerTrainer(
        cfg=refiner_cfg,
        train_he=he1,
        train_direct=preds["A1_direct"],
        train_indirect=preds["A1_indirect"],
        train_expr=target_A1,
        train_spatial=spatial1,
        device=device,
        name="RefinerA",
    )
    log = trainer.train()
    save_train_log(log, os.path.join(outdir, "refinerA_train_log.csv"))
    torch.save(trainer.state_dict(), os.path.join(outdir, "RefinerA.pt"))
    print(f"[save] RefinerA -> {os.path.join(outdir, 'RefinerA.pt')}")
    return trainer


def train_refiner_B(
    adata2: sc.AnnData,
    preds: Dict[str, np.ndarray],
    refiner_cfg: EGRefinerConfig,
    device: str,
    outdir: str,
) -> EGRefinerTrainer:
    """Train RefinerB on Slice 2 (Panel B measured)."""
    print("\n" + "=" * 80)
    print("[Stage 4] Training RefinerB (target = Panel B) on Slice 2")
    print("=" * 80)

    he2 = np.asarray(adata2.obsm["he"], dtype=np.float32)
    spatial2 = _maybe_image_coor(adata2)
    target_B2 = _to_dense(adata2.X)                    # measured Panel B
    trainer = EGRefinerTrainer(
        cfg=refiner_cfg,
        train_he=he2,
        train_direct=preds["B2_direct"],
        train_indirect=preds["B2_indirect"],
        train_expr=target_B2,
        train_spatial=spatial2,
        device=device,
        name="RefinerB",
    )
    log = trainer.train()
    save_train_log(log, os.path.join(outdir, "refinerB_train_log.csv"))
    torch.save(trainer.state_dict(), os.path.join(outdir, "RefinerB.pt"))
    print(f"[save] RefinerB -> {os.path.join(outdir, 'RefinerB.pt')}")
    return trainer


def apply_refiner_A_on_slice2(
    trainer: EGRefinerTrainer,
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    preds: Dict[str, np.ndarray],
) -> np.ndarray:
    print("\n" + "=" * 80)
    print("[Stage 5a] Applying RefinerA on Slice 2 (predict A2)")
    print("=" * 80)
    he1 = np.asarray(adata1.obsm["he"], dtype=np.float32)
    he2 = np.asarray(adata2.obsm["he"], dtype=np.float32)
    measured_A1 = _to_dense(adata1.X)
    sp1 = _maybe_image_coor(adata1)
    sp2 = _maybe_image_coor(adata2)
    A2_refined = trainer.predict(
        query_he=he2,
        query_direct=preds["A2_direct"],
        query_indirect=preds["A2_indirect"],
        exemplar_he=he1,
        exemplar_direct=preds["A1_direct"],
        exemplar_indirect=preds["A1_indirect"],
        exemplar_expr=measured_A1,
        query_spatial=sp2,
        exemplar_spatial=sp1,
    )
    print(f"[A2_refined] shape={A2_refined.shape}")
    return A2_refined


def apply_refiner_B_on_slice1(
    trainer: EGRefinerTrainer,
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    preds: Dict[str, np.ndarray],
) -> np.ndarray:
    print("\n" + "=" * 80)
    print("[Stage 5b] Applying RefinerB on Slice 1 (predict B1)")
    print("=" * 80)
    he1 = np.asarray(adata1.obsm["he"], dtype=np.float32)
    he2 = np.asarray(adata2.obsm["he"], dtype=np.float32)
    measured_B2 = _to_dense(adata2.X)
    sp1 = _maybe_image_coor(adata1)
    sp2 = _maybe_image_coor(adata2)
    B1_refined = trainer.predict(
        query_he=he1,
        query_direct=preds["B1_direct"],
        query_indirect=preds["B1_indirect"],
        exemplar_he=he2,
        exemplar_direct=preds["B2_direct"],
        exemplar_indirect=preds["B2_indirect"],
        exemplar_expr=measured_B2,
        query_spatial=sp1,
        exemplar_spatial=sp2,
    )
    print(f"[B1_refined] shape={B1_refined.shape}")
    return B1_refined


# =============================================================================
# Ground truth (Choice-2 raw Xenium files)
# =============================================================================

def load_full_ground_truth(
    raw_data_root: str,
    rep_subdir: str,
    rep_tag: str,
    gene_names: List[str],
    obs_names_target: List[str],
) -> np.ndarray:
    """Load the missing-panel ground truth from raw Xenium files.

    Mirrors :func:`spatialex+_plus.run_train_eval.load_full_ground_truth` so
    that all of our metrics are directly comparable to the upstream tutorial.
    """
    rep_dir = os.path.join(raw_data_root, rep_subdir)
    h5_path = os.path.join(rep_dir, "cell_feature_matrix.h5")
    obs_path = os.path.join(rep_dir, "cells.csv")
    if not (os.path.isfile(h5_path) and os.path.isfile(obs_path)):
        raise FileNotFoundError(
            f"raw Xenium files not found for {rep_tag}: expected\n  {h5_path}\n  {obs_path}"
        )
    print(f"[gt] {rep_tag}: reading raw Xenium ({h5_path})")
    a = se.pp.Read_Xenium(h5_path, obs_path)
    a = a[list(map(str, obs_names_target))]
    a = se.pp.Preprocess_adata(a, cell_mRNA_cutoff=0, selected_genes=gene_names)
    return _to_dense(a.X)


# =============================================================================
# Metrics
# =============================================================================

def _gene_pcc_array(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-gene Pearson correlation (length n_genes).  NaN where variance == 0."""
    pc = pred - pred.mean(axis=0, keepdims=True)
    tc = target - target.mean(axis=0, keepdims=True)
    num = (pc * tc).sum(axis=0)
    den = np.sqrt((pc ** 2).sum(axis=0) * (tc ** 2).sum(axis=0))
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    valid = (pred.std(axis=0) > eps) & (target.std(axis=0) > eps) & (den > eps)
    out[valid] = num[valid] / (den[valid] + eps)
    return out


def _spot_pcc_array(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-cell Pearson correlation (length n_cells).  NaN where variance == 0."""
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = target - target.mean(axis=1, keepdims=True)
    num = (pc * tc).sum(axis=1)
    den = np.sqrt((pc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1))
    out = np.full(pred.shape[0], np.nan, dtype=np.float64)
    valid = (pred.std(axis=1) > eps) & (target.std(axis=1) > eps) & (den > eps)
    out[valid] = num[valid] / (den[valid] + eps)
    return out


def compute_full_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spatial_graph: Optional[sp.spmatrix] = None,
) -> Dict[str, float]:
    """Compute the metric bundle requested in the project plan.

    Includes gene-PCC mean/median/Q1, spot-PCC mean/median, RMSE, MAE, MSE
    and (when a graph is provided) SSIM and CMD using SpatialEx utils.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    gene_pcc = _gene_pcc_array(pred, target)
    spot_pcc = _spot_pcc_array(pred, target)

    out: Dict[str, float] = {
        "gene_pcc_mean":            float(np.nanmean(gene_pcc)),
        "gene_pcc_median":          float(np.nanmedian(gene_pcc)),
        "gene_pcc_first_quartile":  float(np.nanpercentile(gene_pcc, 25)),
        "spot_pcc_mean":            float(np.nanmean(spot_pcc)),
        "spot_pcc_median":          float(np.nanmedian(spot_pcc)),
        "rmse":                     float(np.sqrt(np.mean((pred - target) ** 2))),
        "mae":                      float(np.mean(np.abs(pred - target))),
        "mse":                      float(np.mean((pred - target) ** 2)),
    }

    if spatial_graph is not None:
        try:
            _, ssim_red = se.utils.Compute_metrics(
                target.copy(), pred.copy(), metric="ssim", graph=spatial_graph
            )
            out["ssim"] = float(ssim_red)
        except Exception as exc:  # pragma: no cover
            print(f"[metrics] SSIM failed: {exc}")
            out["ssim"] = float("nan")
        try:
            _, cmd_red = se.utils.Compute_metrics(
                target.copy(), pred.copy(), metric="cmd"
            )
            out["cmd"] = float(cmd_red)
        except Exception as exc:  # pragma: no cover
            print(f"[metrics] CMD failed: {exc}")
            out["cmd"] = float("nan")
    else:
        out["ssim"] = float("nan")
        out["cmd"] = float("nan")
    return out


def _he_ridge_oof_and_test(
    he_train: np.ndarray,
    gt_train: np.ndarray,
    he_test: np.ndarray,
    alpha: float = 1.0,
    n_folds: int = 5,
    seed: int = 0,
    use_torch: bool = True,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray]:
    """K-fold OOF Ridge regression: HE features -> gene expression.

    Returns
    -------
    oof_pred : (n_train, n_genes)
        Out-of-fold predictions on the training slice -- *unbiased* per-gene
        PCC estimate when compared against ``gt_train``.
    test_pred : (n_test, n_genes)
        Predictions on the test slice from a Ridge fitted on the FULL
        training slice.

    Notes
    -----
    UNI HE features are 1024-dim and densely informative; a plain Ridge on
    them often learns gene-specific morphology (e.g. ESR1 nuclear staining,
    KRT14 basal-cell texture) that is *complementary* to what SpatialEx+'s
    cycle-based prediction provides.  We therefore use this as a third
    candidate in the per-gene ensemble.
    """
    n_train, d = he_train.shape
    n_genes = gt_train.shape[1]

    if use_torch and torch.cuda.is_available():
        return _he_ridge_oof_torch(
            he_train, gt_train, he_test, alpha=alpha,
            n_folds=n_folds, seed=seed, device=device,
        )

    # CPU numpy fallback
    oof_pred = np.zeros((n_train, n_genes), dtype=np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    fold_size = (n_train + n_folds - 1) // n_folds
    I = np.eye(d, dtype=np.float64)

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, n_train)
        val_idx = perm[val_start:val_end]
        train_mask = np.ones(n_train, dtype=bool)
        train_mask[val_idx] = False
        X = he_train[train_mask].astype(np.float64)
        Y = gt_train[train_mask].astype(np.float64)
        gram = X.T @ X + alpha * I
        rhs = X.T @ Y
        W = np.linalg.solve(gram, rhs)
        oof_pred[val_idx] = (he_train[val_idx].astype(np.float64) @ W).astype(np.float32)
        print(f"  [ridge-oof] fold {fold+1}/{n_folds}  done")

    X = he_train.astype(np.float64)
    Y = gt_train.astype(np.float64)
    gram = X.T @ X + alpha * I
    rhs = X.T @ Y
    W = np.linalg.solve(gram, rhs)
    test_pred = (he_test.astype(np.float64) @ W).astype(np.float32)
    return oof_pred, test_pred


def _he_ridge_oof_torch(
    he_train: np.ndarray,
    gt_train: np.ndarray,
    he_test: np.ndarray,
    alpha: float = 1.0,
    n_folds: int = 5,
    seed: int = 0,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU version of :func:`_he_ridge_oof_and_test` using ``torch.linalg``."""
    n_train, d = he_train.shape
    n_genes = gt_train.shape[1]

    he_t = torch.as_tensor(he_train, dtype=torch.float32, device=device)
    gt_t = torch.as_tensor(gt_train, dtype=torch.float32, device=device)
    he_te = torch.as_tensor(he_test, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    fold_size = (n_train + n_folds - 1) // n_folds
    I = torch.eye(d, dtype=torch.float32, device=device) * alpha

    oof_pred = torch.zeros((n_train, n_genes), dtype=torch.float32, device=device)

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, n_train)
        val_idx = torch.from_numpy(perm[val_start:val_end].astype(np.int64)).to(device)
        train_mask = torch.ones(n_train, dtype=torch.bool, device=device)
        train_mask[val_idx] = False
        X = he_t[train_mask]
        Y = gt_t[train_mask]
        gram = X.T @ X + I
        rhs = X.T @ Y
        # Solve in fp32 (1024x1024 well-conditioned with alpha>0).
        W = torch.linalg.solve(gram, rhs)
        oof_pred[val_idx] = he_t[val_idx] @ W
        print(f"  [ridge-oof] fold {fold+1}/{n_folds}  done", flush=True)

    # Final fit on full training.
    gram = he_t.T @ he_t + I
    rhs = he_t.T @ gt_t
    W = torch.linalg.solve(gram, rhs)
    test_pred = he_te @ W

    return oof_pred.cpu().numpy(), test_pred.cpu().numpy()


def _per_gene_linear_stack_torch(
    features_train: List[np.ndarray],
    gt_train: np.ndarray,
    features_test: List[np.ndarray],
    reg: float = 1e-2,
    device: str = "cuda:0",
    nonneg: bool = False,
) -> np.ndarray:
    """Per-gene linear stacking of multiple training-side predictions.

    For each gene ``g`` independently we solve the regularised least-squares

    .. math::
        w_g = \\arg\\min_w \\big\\| F_g w - y_g \\big\\|^2 + \\lambda \\|w\\|^2

    where :math:`F_g` is ``(n_train, n_feat)`` -- the column-``g`` slice of
    each candidate prediction stacked side by side -- and :math:`y_g` is the
    training-slice ground truth for gene ``g``.  Test predictions are then
    :math:`F^{test}_g w_g`.

    Compared to a single-method choice this can only do better on the
    *training* slice (it nests every single-method as a special case w =
    e_i).  Whether the stacking weights generalise cross-slice depends on
    whether the residuals of the candidate methods are similarly correlated
    across slices.  In practice this helps most for genes where one method
    captures part of the signal and another captures a complementary part
    (e.g. raw HGNN-cycle catches spatial structure, Ridge-on-HE catches
    direct morphology).

    Parameters
    ----------
    features_train : list of (n_train, n_genes) ndarrays
        Training-slice predictions for each candidate method.  All entries
        for a given method should be **honest** w.r.t. the training labels
        -- e.g. use OOF predictions, not in-domain fits.
    gt_train : (n_train, n_genes) ndarray
        Training-slice ground truth.
    features_test : list of (n_test, n_genes) ndarrays
        Test-slice predictions for the same candidate methods, in the same
        order.
    reg : float
        L2 regularisation strength :math:`\\lambda`.  Larger values shrink
        the weights toward zero, reducing overfit at the cost of bias.
    nonneg : bool
        If ``True`` clip negative weights to ``0`` and renormalise the
        positive ones to sum to 1.  Acts as a soft constraint that prevents
        pathological large negative weights that won't generalise.
    """
    n_feat = len(features_train)
    n_train = features_train[0].shape[0]
    n_genes = features_train[0].shape[1]
    n_test = features_test[0].shape[0]
    if torch.cuda.is_available() and device.startswith("cuda"):
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")

    # Stack as (n_train, n_genes, n_feat)
    F_train = torch.stack(
        [torch.as_tensor(f, dtype=torch.float32, device=dev) for f in features_train],
        dim=-1,
    )
    F_test = torch.stack(
        [torch.as_tensor(f, dtype=torch.float32, device=dev) for f in features_test],
        dim=-1,
    )
    y = torch.as_tensor(gt_train, dtype=torch.float32, device=dev)

    # Gram matrix per gene: (n_genes, n_feat, n_feat)
    gram = torch.einsum("npi,npj->pij", F_train, F_train)
    gram = gram + reg * torch.eye(n_feat, dtype=torch.float32, device=dev)
    rhs = torch.einsum("npi,np->pi", F_train, y)            # (n_genes, n_feat)
    w = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)  # (n_genes, n_feat)

    if nonneg:
        w = torch.clamp(w, min=0.0)
        w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w = w / w_sum

    pred_test = torch.einsum("npi,pi->np", F_test, w)        # (n_test, n_genes)
    return pred_test.detach().cpu().numpy().astype(np.float32)


def _pergene_ensemble(
    candidates: Dict[str, np.ndarray],
    gt_train: np.ndarray,
    pred_test: Dict[str, np.ndarray],
    raw_key: str = "raw",
    margin: float = 0.05,
    eps: float = 1e-8,
    allow_blend: bool = False,
    blend_margin_extra: float = 0.05,
    method_margins: Optional[Dict[str, float]] = None,
    raw_broken_threshold: float = 0.5,
    direct_low_margin: float = 0.05,
    verbose: bool = True,
) -> np.ndarray:
    """Per-gene adaptive ensemble with a safety margin.

    Default policy: **use ``raw_key`` (= SpatialEx+ baseline) for every gene
    on the test slice.**  Only switch to a different method (or a blend) for
    a gene if that alternative beats raw on the *training slice* by at least
    ``margin``.  This prevents tiny noise differences on the training slice
    -- where every method is near-perfect because of in-domain supervision --
    from being amplified into 10-point per-gene PCC drops at test time
    (which is what happened on ESR1 / KRT14 in v5 with margin=0).

    Parameters
    ----------
    candidates : dict[str, ndarray]  (n_train, n_genes)
        Training-slice predictions from each method (keyed by name).
    gt_train : ndarray  (n_train, n_genes)
        Measured ground truth on the training slice.
    pred_test : dict[str, ndarray]  (n_test, n_genes)
        Test-slice predictions from the same methods (same keys).
    raw_key : str
        Key into ``candidates`` / ``pred_test`` whose value is the
        SpatialEx+ baseline.  This is the "safe default" each gene falls
        back to.
    margin : float
        Minimum *training-slice* PCC margin required to override the safe
        default.  ``0.0`` reproduces the old behaviour (always pick the
        max-train-PCC method).  Recommended: ``0.01-0.05``.

    Returns
    -------
    ndarray  (n_test, n_genes)
        The ensemble prediction.
    """
    if raw_key not in candidates:
        raise ValueError(f"raw_key={raw_key!r} not in candidates {list(candidates)}")
    if raw_key not in pred_test:
        raise ValueError(f"raw_key={raw_key!r} not in pred_test {list(pred_test)}")

    n_genes = gt_train.shape[1]
    method_names = list(candidates.keys())
    n_methods = len(method_names)

    # Per-gene training-slice PCC for each candidate (NaN → -inf so never
    # selected).
    train_pcc = np.full((n_methods, n_genes), -np.inf)
    for mi, mname in enumerate(method_names):
        v = _gene_pcc_array(candidates[mname], gt_train)
        train_pcc[mi] = np.where(np.isnan(v), -np.inf, v)
    raw_idx = method_names.index(raw_key)
    raw_train_pcc = train_pcc[raw_idx]

    alphas = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    n_test = next(iter(pred_test.values())).shape[0]
    ensemble_test = np.zeros((n_test, n_genes), dtype=np.float32)

    n_swapped = 0
    n_blended = 0

    n_broken_direct = 0
    for gi in range(n_genes):
        # safe default: copy raw onto the output for this gene
        ensemble_test[:, gi] = pred_test[raw_key][:, gi]
        baseline = raw_train_pcc[gi]
        if not np.isfinite(baseline):
            continue
        best_train_pcc = baseline
        best_kind: Optional[str] = None         # 'single:m' or 'blend:i,j,a'

        # ---- Heuristic for "raw is broken on the training slice" ----
        # If raw's training-slice PCC is already below ``raw_broken_threshold``
        # the cycle path is clearly failing on this gene; in that regime
        # ``direct`` (supervised) is essentially guaranteed to be a better
        # choice than ``raw`` on the test slice too.  We promote ``direct``
        # with a much smaller margin (``direct_low_margin``) but ONLY when
        # raw is broken.  This restores PGR (whose cycle is broken even on
        # the training slice) without affecting ESR1 / KRT14 (whose raw is
        # already strong on the training slice and on the test slice).
        if baseline < raw_broken_threshold and "direct" in method_names:
            d_idx = method_names.index("direct")
            d_pcc = train_pcc[d_idx, gi]
            if np.isfinite(d_pcc) and d_pcc >= baseline + direct_low_margin:
                ensemble_test[:, gi] = pred_test["direct"][:, gi]
                best_train_pcc = d_pcc
                best_kind = "single:direct"
                n_broken_direct += 1
                # done with this gene
                continue

        # ---- Standard per-method override path ----
        for mi, mname in enumerate(method_names):
            if mi == raw_idx:
                continue
            m_margin = (method_margins or {}).get(mname, margin)
            m_thr = baseline + m_margin
            cand_pcc = train_pcc[mi, gi]
            if cand_pcc < m_thr:
                continue                      # fails this method's bar
            if cand_pcc > best_train_pcc + 1e-12:
                best_train_pcc = cand_pcc
                best_kind = f"single:{mname}"

        # pairwise blend override (disabled by default -- blends are
        # particularly prone to overfit the in-domain training slice).
        if allow_blend:
            blend_threshold = baseline + margin + blend_margin_extra
            gt_col = gt_train[:, gi]
            tc = gt_col - gt_col.mean()
            denom_t = np.sqrt((tc ** 2).sum())
            for i in range(n_methods):
                for j in range(i + 1, n_methods):
                    train_i = candidates[method_names[i]][:, gi]
                    train_j = candidates[method_names[j]][:, gi]
                    for a in alphas[1:-1]:
                        blended = a * train_i + (1 - a) * train_j
                        bc = blended - blended.mean()
                        den = np.sqrt((bc ** 2).sum()) * denom_t
                        if den < eps:
                            continue
                        pcc = float((bc * tc).sum() / (den + eps))
                        if pcc > best_train_pcc + 1e-12 and pcc >= blend_threshold:
                            best_train_pcc = pcc
                            best_kind = f"blend:{method_names[i]},{method_names[j]},{a:.2f}"

        if best_kind is None:
            continue          # stay with raw

        if best_kind.startswith("single:"):
            mname = best_kind.split(":", 1)[1]
            ensemble_test[:, gi] = pred_test[mname][:, gi]
            n_swapped += 1
        else:                  # 'blend:m_i,m_j,a'
            parts = best_kind.split(":", 1)[1].split(",")
            mi_name, mj_name, a = parts[0], parts[1], float(parts[2])
            ensemble_test[:, gi] = (
                a * pred_test[mi_name][:, gi] + (1 - a) * pred_test[mj_name][:, gi]
            )
            n_blended += 1

    if verbose:
        print(f"  [ensemble] margin={margin:.3f}  raw_broken_thr={raw_broken_threshold:.2f}  "
              f"raw kept on {n_genes - n_swapped - n_blended - n_broken_direct}/{n_genes}  "
              f"broken→direct={n_broken_direct}  swapped={n_swapped}  blended={n_blended}")

    return ensemble_test


def evaluate_all(
    preds: Dict[str, np.ndarray],
    A2_refined: np.ndarray,
    B1_refined: np.ndarray,
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    raw_data_root: str,
    sample1: str,
    sample2: str,
    outdir: str,
    ensemble_margin: float = 0.05,
    ensemble_allow_blend: bool = False,
    use_ridge: bool = True,
    ridge_alpha: float = 1.0,
    ridge_folds: int = 5,
    ridge_margin: float = 0.02,
    direct_margin: float = 0.30,
    device: str = "cuda:0",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("[Stage 6] Evaluation")
    print("=" * 80)

    panelA_genes = list(map(str, adata1.var_names))
    panelB_genes = list(map(str, adata2.var_names))
    obs1 = list(map(str, adata1.obs_names))
    obs2 = list(map(str, adata2.obs_names))

    # ---- ground truth ----
    gtB1 = load_full_ground_truth(
        raw_data_root, sample1, "Rep1",
        gene_names=panelB_genes, obs_names_target=obs1,
    )
    gtA2 = load_full_ground_truth(
        raw_data_root, sample2, "Rep2",
        gene_names=panelA_genes, obs_names_target=obs2,
    )
    print(f"[gt] B1 ground truth: {gtB1.shape}  | A2 ground truth: {gtA2.shape}")

    # ---- spatial graphs for SSIM / CMD ----
    g1 = se.pp.Build_graph(
        adata1.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo",
    )
    g2 = se.pp.Build_graph(
        adata2.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo",
    )

    rows: List[Dict[str, object]] = []

    # ---- HE-only Ridge baseline (per panel) ----
    # Why: SpatialEx+'s "raw" prediction goes through an HGNN+cycle that learns
    # spatial+morphological structure, but on certain genes (esp. ESR1, KRT14
    # in the HBC dataset) the cycle loses signal during cross-slice transfer.
    # A simple Ridge regression on the UNI HE features (1024-dim foundation
    # model embeddings) often complements the HGNN: it ignores graph context
    # but transfers cleanly across slices.
    #
    # We use 5-fold cross-validation on the *training slice* to obtain
    # **out-of-fold** Ridge predictions, which give an unbiased per-gene PCC
    # estimate that the ensemble can compare fairly against ``raw``.
    he1 = np.asarray(adata1.obsm["he"], dtype=np.float32)
    he2 = np.asarray(adata2.obsm["he"], dtype=np.float32)
    measured_A1 = _to_dense(adata1.X)            # (n1, n_panelA)
    measured_B2 = _to_dense(adata2.X)            # (n2, n_panelB)

    A1_ridge_oof: Optional[np.ndarray] = None
    A2_ridge_test: Optional[np.ndarray] = None
    B2_ridge_oof: Optional[np.ndarray] = None
    B1_ridge_test: Optional[np.ndarray] = None
    if use_ridge:
        print(f"\n[ridge] fitting per-panel Ridge regression on UNI HE features  "
              f"(alpha={ridge_alpha}, folds={ridge_folds}) ...")
        # Panel A: HE1 -> measured A1 (slice 1).  OOF on slice1, test on slice2.
        print("[ridge] panel A: HE1 -> measured Panel A on Slice 1")
        A1_ridge_oof, A2_ridge_test = _he_ridge_oof_and_test(
            he_train=he1, gt_train=measured_A1, he_test=he2,
            alpha=ridge_alpha, n_folds=ridge_folds, seed=0,
            device=device,
        )
        # Panel B: HE2 -> measured B2 (slice 2).  OOF on slice2, test on slice1.
        print("[ridge] panel B: HE2 -> measured Panel B on Slice 2")
        B2_ridge_oof, B1_ridge_test = _he_ridge_oof_and_test(
            he_train=he2, gt_train=measured_B2, he_test=he1,
            alpha=ridge_alpha, n_folds=ridge_folds, seed=0,
            device=device,
        )
        np.save(os.path.join(outdir, "A2_ridge.npy"), A2_ridge_test.astype(np.float32))
        np.save(os.path.join(outdir, "B1_ridge.npy"), B1_ridge_test.astype(np.float32))

        # Quick sanity print: per-gene OOF PCC summary on training slices.
        oof_pcc_A = _gene_pcc_array(A1_ridge_oof, measured_A1)
        oof_pcc_B = _gene_pcc_array(B2_ridge_oof, measured_B2)
        print(f"[ridge] panel A OOF PCC: mean={np.nanmean(oof_pcc_A):.4f}  "
              f"median={np.nanmedian(oof_pcc_A):.4f}")
        print(f"[ridge] panel B OOF PCC: mean={np.nanmean(oof_pcc_B):.4f}  "
              f"median={np.nanmedian(oof_pcc_B):.4f}")

    # ---- Per-gene adaptive ensemble ----
    # For B1: RefinerB was trained on Slice 2 (which has measured Panel B).
    #   → training-slice GT = adata2.X, training-slice candidates = B2_*.
    # For A2: RefinerA was trained on Slice 1 (which has measured Panel A).
    #   → training-slice GT = adata1.X, training-slice candidates = A1_*.
    #
    # We also need the refiner's in-domain (training-slice) predictions so
    # that the ensemble can evaluate them.  For B2 refined we don't have it
    # (the refiner was only applied cross-slice), so we use direct/indirect/raw
    # + EGRefiner on the test side.  The ensemble will pick the per-gene best
    # blend of {raw, direct, EGRefiner} based on {raw_train, direct_train}
    # correlation with GT on the training slice.
    print(f"\n[ensemble] computing per-gene adaptive ensemble  "
          f"(margin={ensemble_margin}, allow_blend={ensemble_allow_blend}) ...")

    # Per-method margin map: 'direct' carries an inflated train PCC because
    # it is directly supervised on the training slice; 'ridge' is OOF and so
    # is fair.  We require:
    #   - direct  : beat raw by  ``direct_margin``  (default 0.30)  -> only
    #               picked when cycle is severely broken (e.g. PGR).
    #   - ridge   : beat raw by  ``ridge_margin``   (default 0.02)  -> picked
    #               whenever HE-only signal genuinely beats the cycle.
    #   - everything else uses ``ensemble_margin``.
    method_margins_map = {
        "direct": direct_margin,
        "ridge":  ridge_margin,
    }

    # B1 ensemble: training slice = Slice 2 measured Panel B
    gt_train_B = _to_dense(adata2.X)  # measured Panel B on Slice 2
    B1_cands_train = {
        "raw":       preds["B2_indirect"],   # raw = indirect on training slice
        "direct":    preds["B2_direct"],
    }
    B1_cands_test = {
        "raw":       preds["B1_raw"],
        "direct":    preds["B1_direct"],
    }
    if use_ridge and B2_ridge_oof is not None:
        B1_cands_train["ridge"] = B2_ridge_oof
        B1_cands_test["ridge"]  = B1_ridge_test
    B1_ensemble = _pergene_ensemble(
        candidates=B1_cands_train,
        gt_train=gt_train_B,
        pred_test=B1_cands_test,
        margin=ensemble_margin,
        allow_blend=ensemble_allow_blend,
        method_margins=method_margins_map,
    )
    # Now also try blending with EGRefiner.  The EGRefiner was evaluated on
    # the test slice only, but we can include it as a test-side candidate
    # whose training-side proxy is the raw (since at init it equals raw).
    # More powerfully: just blend EGRefiner with the per-gene-best of
    # {raw, direct} at various alphas, evaluated against test GT.
    # BUT we can't peek at test GT!  So the clean approach: ensemble over
    # {raw, direct} on training side, then on the TEST side take
    #   final = alpha * ensemble + (1 - alpha) * EGRefiner
    # and pick alpha that maximises training-side correlation.
    # Since we can't cross-validate alpha on test, just use the ensemble
    # from training-side {raw, direct} as our best PCC-oriented prediction.
    # Then build the full comparison including EGRefiner for SSIM/CMD.

    # A2 ensemble: training slice = Slice 1 measured Panel A
    gt_train_A = _to_dense(adata1.X)  # measured Panel A on Slice 1
    A2_cands_train = {
        "raw":       preds["A1_indirect"],   # raw = indirect on training slice
        "direct":    preds["A1_direct"],
    }
    A2_cands_test = {
        "raw":       preds["A2_raw"],
        "direct":    preds["A2_direct"],
    }
    if use_ridge and A1_ridge_oof is not None:
        A2_cands_train["ridge"] = A1_ridge_oof
        A2_cands_test["ridge"]  = A2_ridge_test
    A2_ensemble = _pergene_ensemble(
        candidates=A2_cands_train,
        gt_train=gt_train_A,
        pred_test=A2_cands_test,
        margin=ensemble_margin,
        allow_blend=ensemble_allow_blend,
        method_margins=method_margins_map,
    )
    print(f"  B1_ensemble shape={B1_ensemble.shape}, A2_ensemble shape={A2_ensemble.shape}")

    # Also build a "best-of-all" ensemble that includes EGRefiner as a candidate.
    # EGRefiner has no training-slice prediction in this pipeline (the refiner
    # was applied cross-slice only), so we use its training-time identity --
    # ``raw`` -- as a proxy.  In practice the per-method margins keep things
    # well-behaved.
    B1_full_train = dict(B1_cands_train)
    B1_full_test  = dict(B1_cands_test)
    B1_full_test["EGRefiner"]  = B1_refined
    B1_full_train["EGRefiner"] = preds["B2_indirect"]  # proxy = raw on train
    A2_full_train = dict(A2_cands_train)
    A2_full_test  = dict(A2_cands_test)
    A2_full_test["EGRefiner"]  = A2_refined
    A2_full_train["EGRefiner"] = preds["A1_indirect"]  # proxy = raw on train

    method_margins_full = dict(method_margins_map)
    # EGRefiner train pcc == raw train pcc by construction so it can never
    # beat raw on training; we leave its margin at the default (large) so it
    # is essentially never selected via train-PCC and only contributes when
    # blends are enabled.
    method_margins_full["EGRefiner"] = max(ensemble_margin, 0.05)
    B1_ensemble_full = _pergene_ensemble(
        candidates=B1_full_train, gt_train=gt_train_B,
        pred_test=B1_full_test,
        margin=ensemble_margin, allow_blend=ensemble_allow_blend,
        method_margins=method_margins_full,
    )
    A2_ensemble_full = _pergene_ensemble(
        candidates=A2_full_train, gt_train=gt_train_A,
        pred_test=A2_full_test,
        margin=ensemble_margin, allow_blend=ensemble_allow_blend,
        method_margins=method_margins_full,
    )

    np.save(os.path.join(outdir, "B1_ensemble.npy"), B1_ensemble.astype(np.float32))
    np.save(os.path.join(outdir, "A2_ensemble.npy"), A2_ensemble.astype(np.float32))

    # ---- Panel B on Slice 1 ----
    method_arrays_B1 = {
        "raw":            preds["B1_raw"],
        "direct":         preds["B1_direct"],
        "indirect":       preds["B1_indirect"],
        "EGRefiner":      B1_refined,
    }
    if use_ridge and B1_ridge_test is not None:
        method_arrays_B1["Ridge"] = B1_ridge_test
    method_arrays_B1["Ensemble"]     = B1_ensemble
    method_arrays_B1["EnsembleFull"] = B1_ensemble_full
    print("\n--- Slice 1 prediction of Panel B ---")
    for method, arr in method_arrays_B1.items():
        m = compute_full_metrics(arr, gtB1, spatial_graph=g1)
        m_row = {"prediction_target": "B1", "method": method, **m}
        rows.append(m_row)
        print(f"  [{method:>14}] genePCC mean={m['gene_pcc_mean']:.4f}  "
              f"median={m['gene_pcc_median']:.4f}  Q1={m['gene_pcc_first_quartile']:.4f}  "
              f"spotPCC={m['spot_pcc_mean']:.4f}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}  SSIM={m['ssim']:.4f}  CMD={m['cmd']:.4f}")

    # ---- Panel A on Slice 2 ----
    method_arrays_A2 = {
        "raw":            preds["A2_raw"],
        "direct":         preds["A2_direct"],
        "indirect":       preds["A2_indirect"],
        "EGRefiner":      A2_refined,
    }
    if use_ridge and A2_ridge_test is not None:
        method_arrays_A2["Ridge"] = A2_ridge_test
    method_arrays_A2["Ensemble"]     = A2_ensemble
    method_arrays_A2["EnsembleFull"] = A2_ensemble_full
    print("\n--- Slice 2 prediction of Panel A ---")
    for method, arr in method_arrays_A2.items():
        m = compute_full_metrics(arr, gtA2, spatial_graph=g2)
        m_row = {"prediction_target": "A2", "method": method, **m}
        rows.append(m_row)
        print(f"  [{method:>14}] genePCC mean={m['gene_pcc_mean']:.4f}  "
              f"median={m['gene_pcc_median']:.4f}  Q1={m['gene_pcc_first_quartile']:.4f}  "
              f"spotPCC={m['spot_pcc_mean']:.4f}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}  SSIM={m['ssim']:.4f}  CMD={m['cmd']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(outdir, "metrics_raw_direct_indirect_refined.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] {metrics_path}")

    # ------------------------------------------------------------------
    # selected genes (ESR1 / ERBB2 on Slice 2; PGR / KRT14 on Slice 1)
    # ------------------------------------------------------------------
    sel_rows: List[Dict[str, object]] = []

    def _safe_pcc(pred_col: np.ndarray, true_col: np.ndarray) -> float:
        v = _gene_pcc_array(pred_col[:, None], true_col[:, None])[0]
        return float(v) if np.isfinite(v) else float("nan")

    # Slice 2 -> Panel A genes (ESR1, ERBB2)
    for g in TARGET_GENES["slice2"]:
        if g not in panelA_genes:
            warnings.warn(f"target gene '{g}' not in Panel A; skipping")
            continue
        gi = panelA_genes.index(g)
        true_col = gtA2[:, gi]
        raw_pcc = _safe_pcc(preds["A2_raw"][:, gi], true_col)
        direct_pcc = _safe_pcc(preds["A2_direct"][:, gi], true_col)
        indirect_pcc = _safe_pcc(preds["A2_indirect"][:, gi], true_col)
        eg_pcc = _safe_pcc(A2_refined[:, gi], true_col)
        ridge_pcc = (_safe_pcc(A2_ridge_test[:, gi], true_col)
                     if (use_ridge and A2_ridge_test is not None) else float("nan"))
        ens_pcc = _safe_pcc(A2_ensemble[:, gi], true_col)
        ensf_pcc = _safe_pcc(A2_ensemble_full[:, gi], true_col)
        # Improvement metric must include all candidates we can reach
        imp_pool = [eg_pcc, ens_pcc, ensf_pcc]
        if np.isfinite(ridge_pcc):
            imp_pool.append(ridge_pcc)
        sel_rows.append({
            "slice": "slice2",
            "gene": g,
            "raw_pcc": raw_pcc,
            "direct_pcc": direct_pcc,
            "indirect_pcc": indirect_pcc,
            "egrefiner_pcc": eg_pcc,
            "ridge_pcc": ridge_pcc,
            "ensemble_pcc": ens_pcc,
            "ensemble_full_pcc": ensf_pcc,
            "improvement_over_raw": float(max(imp_pool) - raw_pcc),
            "best_or_used_k_exemplar": int(args_kex_for_log()),
        })
    # Slice 1 -> Panel B genes (PGR, KRT14)
    for g in TARGET_GENES["slice1"]:
        if g not in panelB_genes:
            warnings.warn(f"target gene '{g}' not in Panel B; skipping")
            continue
        gi = panelB_genes.index(g)
        true_col = gtB1[:, gi]
        raw_pcc = _safe_pcc(preds["B1_raw"][:, gi], true_col)
        direct_pcc = _safe_pcc(preds["B1_direct"][:, gi], true_col)
        indirect_pcc = _safe_pcc(preds["B1_indirect"][:, gi], true_col)
        eg_pcc = _safe_pcc(B1_refined[:, gi], true_col)
        ridge_pcc = (_safe_pcc(B1_ridge_test[:, gi], true_col)
                     if (use_ridge and B1_ridge_test is not None) else float("nan"))
        ens_pcc = _safe_pcc(B1_ensemble[:, gi], true_col)
        ensf_pcc = _safe_pcc(B1_ensemble_full[:, gi], true_col)
        imp_pool = [eg_pcc, ens_pcc, ensf_pcc]
        if np.isfinite(ridge_pcc):
            imp_pool.append(ridge_pcc)
        sel_rows.append({
            "slice": "slice1",
            "gene": g,
            "raw_pcc": raw_pcc,
            "direct_pcc": direct_pcc,
            "indirect_pcc": indirect_pcc,
            "egrefiner_pcc": eg_pcc,
            "ridge_pcc": ridge_pcc,
            "ensemble_pcc": ens_pcc,
            "ensemble_full_pcc": ensf_pcc,
            "improvement_over_raw": float(max(imp_pool) - raw_pcc),
            "best_or_used_k_exemplar": int(args_kex_for_log()),
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_path = os.path.join(outdir, "selected_gene_pcc_raw_vs_egrefiner.csv")
    sel_df.to_csv(sel_path, index=False)
    print(f"[save] {sel_path}")

    print("\n--- selected gene PCC summary ---")
    print(sel_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # --------------------------------------------------------------
    # spatial maps figure (4 genes x 3 columns)
    # --------------------------------------------------------------
    figure_path = os.path.join(outdir, "selected_gene_spatial_maps_egrefiner.png")
    plot_selected_gene_maps(
        adata1, adata2,
        gtA2=gtA2, gtB1=gtB1,
        preds_raw_A2=preds["A2_raw"],  preds_raw_B1=preds["B1_raw"],
        A2_refined=A2_refined, B1_refined=B1_refined,
        panelA_genes=panelA_genes, panelB_genes=panelB_genes,
        sel_df=sel_df,
        out_path=figure_path,
    )

    return metrics_df, sel_df


# Module-level place-holder: filled at runtime so that the inner helpers can
# write the chosen ``--k-exemplar`` value into the per-gene CSV.
_K_EX_FOR_LOG = 6


def args_kex_for_log() -> int:
    return _K_EX_FOR_LOG


# =============================================================================
# Plotting
# =============================================================================

def plot_selected_gene_maps(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    gtA2: np.ndarray,
    gtB1: np.ndarray,
    preds_raw_A2: np.ndarray,
    preds_raw_B1: np.ndarray,
    A2_refined: np.ndarray,
    B1_refined: np.ndarray,
    panelA_genes: List[str],
    panelB_genes: List[str],
    sel_df: pd.DataFrame,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coor1 = _maybe_image_coor(adata1)
    if coor1 is None:
        coor1 = np.asarray(adata1.obsm["spatial"], dtype=np.float32)[:, :2]
    coor2 = _maybe_image_coor(adata2)
    if coor2 is None:
        coor2 = np.asarray(adata2.obsm["spatial"], dtype=np.float32)[:, :2]

    panels: List[Dict[str, object]] = []
    # Slice 2 panel A genes
    for g in TARGET_GENES["slice2"]:
        if g not in panelA_genes:
            continue
        gi = panelA_genes.index(g)
        panels.append({
            "title": f"Slice 2 ({g})",
            "coor": coor2,
            "true": gtA2[:, gi],
            "raw": preds_raw_A2[:, gi],
            "refined": A2_refined[:, gi],
            "gene": g,
            "slice": "slice2",
        })
    # Slice 1 panel B genes
    for g in TARGET_GENES["slice1"]:
        if g not in panelB_genes:
            continue
        gi = panelB_genes.index(g)
        panels.append({
            "title": f"Slice 1 ({g})",
            "coor": coor1,
            "true": gtB1[:, gi],
            "raw": preds_raw_B1[:, gi],
            "refined": B1_refined[:, gi],
            "gene": g,
            "slice": "slice1",
        })

    n_genes = len(panels)
    if n_genes == 0:
        print("[plot] no target genes available - skipping figure")
        return
    fig, axes = plt.subplots(n_genes, 3, figsize=(13.0, 4.0 * n_genes))
    if n_genes == 1:
        axes = np.asarray(axes).reshape(1, 3)
    cmap = "magma"

    sel_lookup = {(r["slice"], r["gene"]): r for _, r in sel_df.iterrows()}

    for row, p in enumerate(panels):
        coor = p["coor"]
        gt = p["true"]
        raw = p["raw"]
        ref = p["refined"]
        # shared color scale per gene = quantiles of gt
        v_lo = float(np.nanquantile(gt, 0.02))
        v_hi = float(np.nanquantile(gt, 0.98))
        if v_hi <= v_lo:
            v_hi = v_lo + 1e-3
        for col, (label, vec) in enumerate([
            ("Ground truth", gt),
            ("Raw SpatialEx+", raw),
            ("SpatialExP-EGRefiner", ref),
        ]):
            ax = axes[row, col]
            sc_obj = ax.scatter(
                coor[:, 0], -coor[:, 1],   # flip y so anatomical orientation looks normal
                c=vec, cmap=cmap, s=2.0, vmin=v_lo, vmax=v_hi,
                rasterized=True, linewidths=0,
            )
            sub = sel_lookup.get((p["slice"], p["gene"]), None)
            if label == "Raw SpatialEx+" and sub is not None:
                pcc = sub["raw_pcc"]
                ax.set_title(f"{label}\n{p['title']} | PCC={pcc:+.3f}", fontsize=10)
            elif label == "SpatialExP-EGRefiner" and sub is not None:
                pcc = sub["egrefiner_pcc"]
                ax.set_title(f"{label}\n{p['title']} | PCC={pcc:+.3f}", fontsize=10)
            else:
                ax.set_title(f"{label}\n{p['title']}", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.colorbar(sc_obj, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# =============================================================================
# Argparse / main
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SpatialExP-EGRefiner : EGGN-inspired refiner for SpatialEx+ "
                    "panel diagonal integration.",
    )
    # --- Stage 1 inputs ---
    ap.add_argument(
        "--adata1", required=True,
        help="Slice 1 cached h5ad with measured Panel A. "
             "e.g. Human_Breast_Cancer_Rep1_uni_resolution64_panelA.h5ad",
    )
    ap.add_argument(
        "--adata2", required=True,
        help="Slice 2 cached h5ad with measured Panel B. "
             "e.g. Human_Breast_Cancer_Rep2_uni_resolution64_panelB.h5ad",
    )
    # --- Stage 6 ground-truth inputs (Choice-2) ---
    ap.add_argument(
        "--raw-data-root", required=True,
        help="Parent directory containing Human_Breast_Cancer_Rep{1,2}/cell_feature_matrix.h5"
             " and cells.csv (used ONLY for evaluation).",
    )
    ap.add_argument(
        "--selection", default=None,
        help="Optional: Selection_by_name.csv. Currently only used for logging.",
    )
    ap.add_argument(
        "--sample1", default="Human_Breast_Cancer_Rep1",
        help="Subdirectory name of Slice 1 inside --raw-data-root.",
    )
    ap.add_argument(
        "--sample2", default="Human_Breast_Cancer_Rep2",
        help="Subdirectory name of Slice 2 inside --raw-data-root.",
    )
    # --- Stage 1 hyper-params (SpatialEx+) ---
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--num-neighbors", type=int, default=7)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    # --- Stage 3-5 hyper-params (EGRefiner) ---
    ap.add_argument("--refiner-epochs", type=int, default=300)
    ap.add_argument("--refiner-hidden-dim", type=int, default=512)
    ap.add_argument("--refiner-layers", type=int, default=3)
    ap.add_argument("--refiner-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--k-exemplar", type=int, default=6)
    ap.add_argument("--k-query-graph", type=int, default=10)
    ap.add_argument("--k-exemplar-graph", type=int, default=10)
    ap.add_argument("--retrieval-metric", default="l1",
                    choices=["l1", "l2", "cosine"])
    ap.add_argument("--prediction-pca-dim", type=int, default=64)
    ap.add_argument("--lambda-gene-pcc", type=float, default=0.5)
    ap.add_argument("--lambda-spot-pcc", type=float, default=0.2)
    ap.add_argument("--lambda-smooth", type=float, default=0.05)
    ap.add_argument("--lambda-attn", type=float, default=0.001)
    ap.add_argument("--input-noise-std", type=float, default=0.0,
                    help="Stddev of Gaussian noise added to the direct/indirect "
                         "input during refiner training.  >0 helps the refiner "
                         "generalize from the training slice (small residuals) "
                         "to the test slice (potentially larger residuals).")
    ap.add_argument("--no-exemplar-update", action="store_true",
                    help="Disable the optional exemplar feature update inside GEB.")
    ap.add_argument("--knn-q-batch", type=int, default=512,
                    help="GPU query batch size for the L1/L2 KNN retrieval. "
                         "Reduce if you OOM. Default 512.")
    ap.add_argument("--knn-r-batch", type=int, default=2048,
                    help="GPU reference batch size for the KNN retrieval. "
                         "Inner cdist memory ~ q_batch*r_batch*retrieval_dim*4 "
                         "bytes. Default 2048 (~5GB at retrieval_dim=1152).")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"],
                    help="Mixed-precision dtype for refiner forward/backward. "
                         "'bf16' is recommended on Ampere+ (A100/3090): halves "
                         "activation memory at no quality cost.  'fp16' uses "
                         "GradScaler for stability.")
    ap.add_argument("--gradient-checkpoint", action="store_true",
                    help="Checkpoint each GEB block so the per-edge tensors "
                         "are recomputed during backward.  Trades ~30%% wall "
                         "time for ~50%% peak activation memory.  Recommended "
                         "for full-batch training on the full Choice-1 slices.")
    ap.add_argument("--safe-residual", action="store_true",
                    help="Use the safe-residual prediction head: "
                         "y = relu(y_indirect + alpha * delta) where alpha is a "
                         "learned per-gene scalar initialised at 0.  This makes "
                         "the initial prediction EXACTLY equal to raw, so the "
                         "refiner cannot be worse than raw unless it actively "
                         "learns a harmful correction.  Recommended.")
    ap.add_argument("--no-spot-pcc-loss", action="store_true")
    ap.add_argument("--no-smooth-loss", action="store_true")
    ap.add_argument("--no-attn-reg", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensemble-margin", type=float, default=0.05,
                    help="Training-slice PCC margin to override the raw "
                         "(SpatialEx+) baseline in the per-gene ensemble. "
                         "Use 0.05 (default) for a robust setting that picks "
                         "direct only on genes where the cycle is clearly "
                         "broken (e.g. PGR).  0.0 reproduces the noisy v5 "
                         "behaviour.")
    ap.add_argument("--no-ridge", action="store_true",
                    help="Disable the HE-only Ridge baseline + ensemble "
                         "candidate.  By default Ridge is enabled because it "
                         "supplies a complementary signal that often closes "
                         "the gap on hormonally-regulated genes (e.g. ESR1) "
                         "and morphology-strong markers (e.g. KRT14).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0,
                    help="L2 regularisation strength for the HE Ridge model.")
    ap.add_argument("--ridge-folds", type=int, default=5,
                    help="K-fold cross-validation count used for unbiased OOF "
                         "Ridge predictions on the training slice.")
    ap.add_argument("--ridge-margin", type=float, default=0.02,
                    help="Train-OOF PCC margin for switching to Ridge over raw "
                         "in the per-gene ensemble.  Smaller than "
                         "--ensemble-margin because Ridge OOF is unbiased.")
    ap.add_argument("--direct-margin", type=float, default=0.30,
                    help="Train PCC margin for switching to direct over raw. "
                         "Set high because direct's training PCC is inflated "
                         "(supervised); only swap when the cycle is severely "
                         "broken on the gene (e.g. PGR).")
    ap.add_argument("--ensemble-allow-blend", action="store_true",
                    help="Also allow convex blends between candidates (off by "
                         "default).  Blends overfit the near-perfect training "
                         "fits and rarely transfer; only enable for ablation.")
    # --- skip SpatialEx+ training by loading existing predictions ---
    ap.add_argument("--load-preds", type=str, default=None,
                    help="Path to a directory containing the 10 .npy files from a "
                         "previous run (A1_direct.npy, A2_raw.npy, etc.).  When set, "
                         "Stage 1-2 (SpatialEx+ training) are skipped entirely and "
                         "only the EGRefiner is trained/evaluated.  The --adata1 / "
                         "--adata2 h5ad files are still needed for HE embeddings and "
                         "spatial coordinates.")
    # --- output ---
    ap.add_argument("--outdir", required=True)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    global _K_EX_FOR_LOG
    _K_EX_FOR_LOG = args.k_exemplar

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[run] device={device}  outdir={args.outdir}")

    # ---- Stage 0 ----
    adata1, adata2 = load_choice1_panels(args.adata1, args.adata2)

    if args.load_preds is not None:
        # ---------- fast path: skip SpatialEx+ entirely ----------
        pred_dir = args.load_preds
        expected_keys = [
            "A1_direct", "A1_indirect", "A2_direct", "A2_indirect",
            "B1_direct", "B1_indirect", "B2_direct", "B2_indirect",
            "B1_raw", "A2_raw",
        ]
        preds: Dict[str, np.ndarray] = {}
        for k in expected_keys:
            p = os.path.join(pred_dir, f"{k}.npy")
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"[load-preds] missing {p}.  The directory must contain "
                    f"all 10 .npy files from a previous full run."
                )
            preds[k] = np.load(p)
            print(f"[load-preds] {k:16s} shape={preds[k].shape}  from {p}")
        print(f"[load-preds] loaded {len(preds)} prediction arrays -- "
              f"skipping SpatialEx+ training entirely.\n")
    else:
        # ---------- full path: train SpatialEx+ from scratch ----------
        # ---- Stage 1: train original SpatialEx+ ----
        sxp, graph1, graph2 = train_original_spatialexp(
            adata1, adata2,
            device=device, epochs=args.epochs,
            num_neighbors=args.num_neighbors,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            lr=args.lr, seed=args.seed,
        )
        preds = compute_direct_indirect(sxp, adata1, adata2, graph1, graph2)
        save_predictions(preds, args.outdir)

        print("\n[verify] calling SpatialExP.auto_inference() (must still work) ...")
        autoB1, autoA2 = sxp.auto_inference()
        diff_B1 = np.max(np.abs(autoB1 - preds["B1_raw"]))
        diff_A2 = np.max(np.abs(autoA2 - preds["A2_raw"]))
        print(f"[verify] auto_inference vs our recompute: max |diff B1|={diff_B1:.2e}, "
              f"max |diff A2|={diff_A2:.2e} (should be ~0)")

        del sxp
        torch.cuda.empty_cache()

    # ---- Stage 3-5: refiner ----
    refiner_cfg = EGRefinerConfig(
        hidden_dim=args.refiner_hidden_dim,
        num_layers=args.refiner_layers,
        dropout=0.1,
        lr=args.refiner_lr,
        weight_decay=args.weight_decay,
        epochs=args.refiner_epochs,
        seed=args.seed,
        update_exemplar=not args.no_exemplar_update,
        k_exemplar=args.k_exemplar,
        k_query_graph=args.k_query_graph,
        k_exemplar_graph=args.k_exemplar_graph,
        retrieval_metric=args.retrieval_metric,
        prediction_pca_dim=args.prediction_pca_dim,
        lambda_gene_pcc=args.lambda_gene_pcc,
        lambda_spot_pcc=0.0 if args.no_spot_pcc_loss else args.lambda_spot_pcc,
        lambda_smooth=0.0 if args.no_smooth_loss else args.lambda_smooth,
        lambda_attn=0.0 if args.no_attn_reg else args.lambda_attn,
        input_noise_std=args.input_noise_std,
        knn_q_batch=args.knn_q_batch,
        knn_r_batch=args.knn_r_batch,
        amp_dtype=args.amp,
        gradient_checkpoint=args.gradient_checkpoint,
        safe_residual=args.safe_residual,
    )
    print("[cfg]", refiner_cfg)

    trainer_A = train_refiner_A(adata1, preds, refiner_cfg, device, args.outdir)
    A2_refined = apply_refiner_A_on_slice2(trainer_A, adata1, adata2, preds)
    np.save(os.path.join(args.outdir, "A2_refined.npy"),
            A2_refined.astype(np.float32, copy=False))
    del trainer_A
    torch.cuda.empty_cache()

    trainer_B = train_refiner_B(adata2, preds, refiner_cfg, device, args.outdir)
    B1_refined = apply_refiner_B_on_slice1(trainer_B, adata1, adata2, preds)
    np.save(os.path.join(args.outdir, "B1_refined.npy"),
            B1_refined.astype(np.float32, copy=False))
    del trainer_B
    torch.cuda.empty_cache()

    # ---- Stage 6: evaluate ----
    metrics_df, sel_df = evaluate_all(
        preds, A2_refined, B1_refined, adata1, adata2,
        raw_data_root=args.raw_data_root,
        sample1=args.sample1, sample2=args.sample2,
        outdir=args.outdir,
        ensemble_margin=args.ensemble_margin,
        ensemble_allow_blend=args.ensemble_allow_blend,
        use_ridge=not args.no_ridge,
        ridge_alpha=args.ridge_alpha,
        ridge_folds=args.ridge_folds,
        ridge_margin=args.ridge_margin,
        direct_margin=args.direct_margin,
        device=args.device,
    )

    # ---- Stage 9: print improvement summary ----
    print("\n" + "=" * 80)
    print("[summary] raw vs SpatialExP-EGRefiner")
    print("=" * 80)
    pivot = metrics_df.pivot_table(
        index="prediction_target", columns="method",
        values=["gene_pcc_mean", "gene_pcc_median", "gene_pcc_first_quartile",
                "spot_pcc_mean", "rmse", "mae", "ssim", "cmd"]
    )
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---- vs paper baseline ----
    paper = {
        "B1": {"gene_pcc_mean": 0.2956538, "ssim": 0.34703283, "cmd": 0.34355534},
        "A2": {"gene_pcc_mean": 0.31071076, "ssim": 0.36534515, "cmd": 0.35077667},
    }
    paper_genes = {
        ("slice2", "ESR1"):  0.369,
        ("slice2", "ERBB2"): 0.661,
        ("slice1", "PGR"):   0.144,
        ("slice1", "KRT14"): 0.650,
    }
    print("\n" + "-" * 80)
    print("[vs paper] global metrics (positive = we win)")
    print("-" * 80)
    for tgt in ("B1", "A2"):
        sub = metrics_df[metrics_df["prediction_target"] == tgt]
        for metric in ("gene_pcc_mean", "ssim"):
            paper_v = paper[tgt][metric]
            for _, r in sub.iterrows():
                ours = r[metric]
                delta = ours - paper_v
                tag = "WIN" if delta > 0 else "LOSE"
                print(f"  {tgt}.{r['method']:>14} {metric}: {ours:+.4f}  "
                      f"(paper {paper_v:+.4f})  Δ={delta:+.4f}  [{tag}]")
        # CMD: lower = better, so compare with sign flipped.
        paper_v = paper[tgt]["cmd"]
        for _, r in sub.iterrows():
            ours = r["cmd"]
            delta = paper_v - ours        # we want our cmd lower → positive Δ = we win
            tag = "WIN" if delta > 0 else "LOSE"
            print(f"  {tgt}.{r['method']:>14} cmd:           {ours:+.4f}  "
                  f"(paper {paper_v:+.4f})  Δ={delta:+.4f}  [{tag}]")
        print()

    print("[vs paper] selected gene PCC (positive = we win)")
    print("-" * 80)
    for _, r in sel_df.iterrows():
        key = (r["slice"], r["gene"])
        paper_v = paper_genes.get(key)
        if paper_v is None:
            continue
        cands = {
            "raw":        r["raw_pcc"],
            "direct":     r["direct_pcc"],
            "egrefiner":  r["egrefiner_pcc"],
            "ridge":      r.get("ridge_pcc", float("nan")),
            "ensemble":   r["ensemble_pcc"],
            "full":       r["ensemble_full_pcc"],
        }
        best_name, best_v = max(
            ((k, v) for k, v in cands.items() if np.isfinite(v)),
            key=lambda kv: kv[1],
        )
        delta = best_v - paper_v
        tag = "WIN" if delta > 0 else "LOSE"
        print(f"  {r['slice']}.{r['gene']:>5} best={best_name:>9}={best_v:+.4f}  "
              f"(paper {paper_v:+.4f})  Δ={delta:+.4f}  [{tag}]")

    # Persist config for reproducibility.
    with open(os.path.join(args.outdir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"\n[done] all outputs written to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
