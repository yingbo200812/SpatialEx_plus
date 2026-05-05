"""
SpatialEx-pro 的可微损失函数集。

所有损失都作用在稠密的 ``(n_cells, n_genes)`` 张量上，可选 cell mask
（用于 anchor 损失只对可信 cell 计算）+ 可选 per-gene 权重；任一损失
都不会构造 ``(n_cells, n_cells)`` 中间张量，可在全片 HBC (~160K cells)
上直接前向。

本文件的核心是把"训练目标 ≡ 评估指标"这件事做实：

- :func:`weighted_mse`     —— 加权 MSE，监督主路径。
- :func:`pearson_loss`     —— ``mean_g(1 - PCC_g)``，直接优化 PCC 指标。
- :func:`cmd_align_loss`   —— 基因-基因相关阵的 Frobenius 余弦距离，
  与上游 ``Compute_metrics(metric='cmd')`` 同公式，直接优化 CMD 指标。
- :func:`tv_loss_edges`    —— 空间 KNN 上的 L2 总变分，与 SSIM 评估
  在结构层面同向（"邻居的预测应当相似"）。
- :func:`spot_aggregated_mse` —— 沿用 baseline 的伪 spot 低频 MSE，
  作为稳定项保留。
- :func:`make_gene_weights` —— 构造 per-gene 权重向量，按 inv-std 平
  衡基因方差差异并对 marker gene 加权。
"""

from __future__ import annotations

from typing import Optional

import torch


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    gene_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-gene-weighted MSE.

    Parameters
    ----------
    pred, target : (n_cells, n_genes) tensors.
    mask : (n_cells,) bool/float tensor.  If given, cells where ``mask``
        is 0 do not contribute.  Useful for the anchor loss where some
        cells have no trustworthy cross-slice match.
    gene_weights : (n_genes,) tensor.  Multiplied into the squared error
        per gene before reduction.
    """
    se = (pred - target) ** 2
    if gene_weights is not None:
        se = se * gene_weights.view(1, -1)
    if mask is None:
        return se.mean()
    cell_w = mask.view(-1, 1).to(pred.dtype)
    n_active = cell_w.sum().clamp_min(1.0)
    return (se * cell_w).sum() / (n_active * pred.shape[1])


def pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    gene_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-gene Pearson loss = ``mean_g(1 - PCC_g)``.

    PCC is computed on the cells selected by ``mask`` (defaults to all).
    Returned as a scalar.
    """
    if mask is not None:
        m = mask.view(-1).to(torch.bool)
        if m.sum() == 0:
            return pred.new_zeros(())
        p = pred[m]
        t = target[m]
    else:
        p = pred
        t = target
    p_c = p - p.mean(dim=0, keepdim=True)
    t_c = t - t.mean(dim=0, keepdim=True)
    num = (p_c * t_c).sum(dim=0)
    den = torch.sqrt((p_c ** 2).sum(dim=0) * (t_c ** 2).sum(dim=0)) + eps
    pcc = num / den
    loss_g = 1.0 - pcc  # (n_genes,)
    if gene_weights is not None:
        return (loss_g * gene_weights).mean()
    return loss_g.mean()


def _gene_corr(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pearson correlation matrix across the gene dim, computed over cells.

    ``x`` is ``(n_cells, n_genes)``.  Returns a ``(n_genes, n_genes)``
    correlation matrix.  Genes with zero variance get an all-zero row /
    column (so they do not contaminate the Frobenius inner product).
    """
    x_c = x - x.mean(dim=0, keepdim=True)
    sd = x_c.pow(2).sum(dim=0).clamp_min(eps).sqrt()
    x_n = x_c / sd.view(1, -1)
    # Mark zero-variance genes -- their normalised column is exactly 0.
    bad = (sd <= eps)
    if bad.any():
        x_n = x_n.clone()
        x_n[:, bad] = 0.0
    # cells^T @ cells -> (genes, genes), divided by n-1 to match np.corrcoef.
    n = x.shape[0]
    return (x_n.t() @ x_n) / max(n - 1, 1)


def cmd_align_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gene-gene correlation-matrix distance, mirroring the eval metric.

    ``CMD = 1 - <C_pred, C_gt>_F / (||C_pred||_F * ||C_gt||_F)``

    Where ``C_*`` is the (genes, genes) Pearson correlation matrix taken
    across cells.  This is identical (up to numerical issues with
    zero-variance genes) to the upstream
    :func:`SpatialEx.utils.Compute_metrics` implementation with
    ``metric='cmd'``, so minimising it directly drags the predicted
    co-expression structure towards the ground truth.
    """
    Cp = _gene_corr(pred, eps=eps)
    Cg = _gene_corr(target, eps=eps)
    num = (Cp * Cg).sum()
    den = Cp.norm() * Cg.norm() + eps
    return 1.0 - num / den


def tv_loss_edges(
    pred: torch.Tensor,
    edges: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Spatial total-variation on a (2, E) edge index.

    For each edge ``(i, j)`` adds ``||pred_i - pred_j||^2`` (Frobenius
    over the gene dim) to the loss.
    """
    if edges.numel() == 0:
        return pred.new_zeros(())
    src = pred[edges[0]]
    dst = pred[edges[1]]
    diff_sq = (src - dst).pow(2).mean(dim=1)  # (E,) per-edge mean over genes
    if edge_weights is not None:
        diff_sq = diff_sq * edge_weights
    return diff_sq.mean()


def spot_aggregated_mse(
    pred: torch.Tensor,
    target_spot: torch.Tensor,
    agg_mtx_torch_sparse: torch.Tensor,
    gene_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Baseline-style MSE in pseudo-spot space.

    ``agg_mtx_torch_sparse`` is a sparse ``(n_spots, n_cells)`` matrix
    that aggregates cells into spots; ``target_spot`` is the GT in spot
    space.
    """
    pred_spot = torch.sparse.mm(agg_mtx_torch_sparse, pred)
    se = (pred_spot - target_spot) ** 2
    if gene_weights is not None:
        se = se * gene_weights.view(1, -1)
    return se.mean()


def make_gene_weights(
    n_genes: int,
    var_names,
    use_invstd: bool = True,
    invstd_clip: tuple = (0.5, 5.0),
    expr_for_std=None,
    marker_genes=(),
    marker_weight: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Build the (n_genes,) weight vector used by all weighted losses.

    ``expr_for_std`` is a ``(n_cells, n_genes)`` tensor (or numpy array)
    used to estimate the per-gene std for the inverse-std weighting.  If
    ``None`` or ``use_invstd=False`` the base weight is 1.

    Marker genes (matched by name in ``var_names``) get an additional
    ``marker_weight`` multiplier.
    """
    import numpy as np
    base = torch.ones(n_genes, device=device, dtype=torch.float32)
    if use_invstd and expr_for_std is not None:
        if isinstance(expr_for_std, torch.Tensor):
            std = expr_for_std.std(dim=0).cpu().numpy()
        else:
            std = np.asarray(expr_for_std).std(axis=0)
        w = 1.0 / (std + 1e-3)
        w = np.clip(w, invstd_clip[0], invstd_clip[1])
        # Center to have mean 1 so the overall loss scale is unchanged.
        w = w / w.mean()
        base = torch.from_numpy(w.astype(np.float32)).to(device)
    if marker_weight != 1.0 and len(marker_genes) > 0:
        var_list = list(map(str, var_names))
        for g in marker_genes:
            if g in var_list:
                base[var_list.index(g)] *= float(marker_weight)
    return base
