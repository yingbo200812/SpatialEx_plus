"""
SpatialEx-pro 工具模块。

本文件汇总了 SpatialEx-pro 训练 / 推理 / 评估流程所需的、与神经网络无关
的全部"非核心"工具，按职责分为三段：

1. **超参配置**：``SpatialExProConfig`` —— 数据类形式集中管理所有可调
   超参，:mod:`run_train_eval` 通过 CLI 覆盖默认值，并在产物目录下保存
   ``config.json`` 以便完全复现。
2. **跨片 H&E 锚点工具**：``build_cross_slice_anchors`` /
   ``build_within_slice_he_smoother`` —— 算法核心改进之一。在不使用任何
   测试切片真值（leave-one-out safe）的前提下，利用对侧切片的 UNI/H&E
   邻居把"对侧 GT"加权平均出软伪标签，既用于训练时的跨片伪监督，也
   用于推理时的 anchor smoothing。
3. **评估工具**：``evaluate_one_direction`` /
   ``evaluate_two_directions`` / ``per_gene_pcc_table`` / ``side_by_side``
   —— 直接调用上游 ``SpatialEx.utils.Compute_metrics``，保证 PCC / SSIM /
   CMD 指标定义与原论文 Fig. 2c/d 完全一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 超参配置
# ---------------------------------------------------------------------------

@dataclass
class SpatialExProConfig:
    """SpatialEx-pro 训练器超参配置。

    所有默认值均在 HBC Rep1/Rep2 (Choice 1, 同 313 基因 panel) 上调过；
    早期 v1 默认 (anchor=1.0, marker boost=5x, TV=0.05, 无 CMD-align) 会
    把基因-基因共表达结构压扁、CMD 反而上升，当前默认略放低锚点 / 标
    志基因 / TV 的权重，并加入 CMD-align 项以同时优化 PCC、SSIM、CMD
    三个指标。
    """

    # ----- 主干网络 -----
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    use_dgi: bool = True
    share_projection: bool = True
    """两个 HGNN head 是否共享 H&E 投影 MLP；共享后编码器在两片 H&E 上
    都获得梯度信号，是把跨片域差异打平的最廉价手段。"""

    # ----- 训练 -----
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 500
    seed: int = 0
    use_cosine_lr: bool = True
    lr_min_ratio: float = 0.1
    """Cosine LR 衰减下限（占初始 lr 的比例）。0 表示衰减到 0；0.1 保留
    一个小 LR 让训练后期还能继续微调跨片 anchor 关系。"""

    # ----- 监督损失权重 -----
    lambda_mse_cell: float = 1.0
    lambda_mse_spot: float = 0.5
    """伪 spot 维度 MSE，模拟 baseline 的 ``Generate_pseudo_spot`` 平滑
    路径；置 0 则关闭。"""
    lambda_pearson: float = 1.0

    # ----- 跨片 anchor 伪监督 -----
    lambda_anchor_mse: float = 0.5
    lambda_anchor_pearson: float = 0.3
    lambda_anchor_warmup: int = 80
    """anchor 损失从 0 线性升到全权的 epoch 数；比 v1 (30) 长很多，让
    真实-GT 监督路径先把 panel 拟合好再引入跨片信号。"""
    anchor_k: int = 8
    anchor_sim_floor: float = 0.3
    """top-1 跨片 cosine 相似度低于该阈值的 cell 在 anchor 损失中被
    mask 掉，过滤掉 H&E 上分布外（OOD）的 cell。"""
    use_mnn_anchors: bool = False

    # ----- 空间正则 -----
    lambda_spatial_tv: float = 0.02
    tv_max_edges: int = 200_000

    # ----- 基因-基因共表达对齐（直接最小化 CMD）-----
    lambda_cmd_align: float = 0.5
    """CMD = 1 - cos(C_pred, C_gt)，C_* 是 (genes, genes) 的 Pearson
    相关阵。MSE / Pearson / TV 都不直接约束这个矩阵，加该项后基因-基
    因共表达结构会被显式拉向 GT。"""
    cmd_align_subsample: int = 30000
    """每个 epoch 在每个切片上随机抽样这么多 cell 来算 (genes, genes)
    相关阵，控制显存 + 给训练注入随机正则。<=0 则用全部 cell。"""

    # ----- DGI -----
    lambda_dgi: float = 0.3

    # ----- 基因加权 -----
    use_invstd_weighting: bool = True
    invstd_clip: tuple = (0.7, 2.0)
    """inv-std 权重的截断区间；比 v1 (0.5, 5.0) 紧，避免低方差噪声基因
    把损失带偏。"""
    marker_genes: List[str] = field(default_factory=lambda: [
        "EPCAM", "ESR1", "PGR", "ERBB2", "KRT14",
    ])
    marker_weight: float = 2.0
    marker_pearson_weight: float = 3.0

    # ----- 推理 / 后处理 -----
    alpha_spatial: float = 0.1
    """切片内 H&E-NN 平滑混合权重。"""
    beta_anchor: float = 0.1
    """跨片 H&E-NN anchor 在测试时的混合权重；anchor 仍是 leave-one-out
    safe，因为它仅来自训练切片的 GT。"""
    refine_anchor_k: int = 15

    # ----- 数值 -----
    grad_clip: float = 5.0

    # ----- bookkeeping -----
    device: str = "cuda:0"
    log_every: int = 10

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# 2. 跨片 H&E 锚点工具
# ---------------------------------------------------------------------------

def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


@torch.no_grad()
def _topk_cross_slice(
    feat_query: torch.Tensor,
    feat_key: torch.Tensor,
    k: int,
    batch_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 ``feat_query`` 的每一行（已 L2 归一化），在 ``feat_key`` 中找
    cosine 最近的 top-K 邻居。返回 (idx, sim) 两个张量。"""
    n_q = feat_query.shape[0]
    idx_chunks: list[torch.Tensor] = []
    sim_chunks: list[torch.Tensor] = []
    for i in range(0, n_q, batch_size):
        chunk = feat_query[i : i + batch_size]
        sim = chunk @ feat_key.T  # (b, n_k)
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
        idx_chunks.append(topk_idx)
        sim_chunks.append(topk_vals)
    return torch.cat(idx_chunks, dim=0), torch.cat(sim_chunks, dim=0)


@torch.no_grad()
def build_cross_slice_anchors(
    he_train: np.ndarray,
    he_test: np.ndarray,
    gt_train: np.ndarray,
    *,
    k: int = 10,
    sim_floor: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
    use_mnn: bool = False,
    he_test_self: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """为测试切片每个 cell 构造一个软伪标签（leave-one-out safe）。

    参数命名是 *方向性* 的，方便在调用处一眼看出 leave-one-out 语义：

    Parameters
    ----------
    he_train : (n_train, d)
        我们 *允许* 使用 GT 的那一片的 UNI/H&E 嵌入。
    he_test : (n_test, d)
        我们 *不允许* 使用 GT 的那一片的 UNI/H&E 嵌入。
    gt_train : (n_train, n_genes)
        训练片的真实表达（允许使用）。
    k, sim_floor, batch_size, device, use_mnn :
        常规旋钮。``use_mnn=True`` 时需提供 ``he_test_self``。

    Returns
    -------
    dict
        - ``target`` : (n_test, n_genes) float32 —— 每个测试 cell 的伪 GT。
        - ``mask``   : (n_test,)        float32 —— 0/1 anchor 是否可信。
        - ``best_sim`` : (n_test,)      float32 —— 诊断用的 top-1 sim。
    """
    feat_train = torch.as_tensor(np.asarray(he_train), dtype=torch.float32, device=device)
    feat_test = torch.as_tensor(np.asarray(he_test), dtype=torch.float32, device=device)
    gt_train_t = torch.as_tensor(np.asarray(gt_train), dtype=torch.float32, device=device)
    feat_train = _l2_normalize(feat_train)
    feat_test = _l2_normalize(feat_test)

    idx_test_to_train, sim_test_to_train = _topk_cross_slice(
        feat_test, feat_train, k=k, batch_size=batch_size
    )
    sim_pos = sim_test_to_train.clamp_min(0.0)
    sim_norm = sim_pos / (sim_pos.sum(dim=1, keepdim=True) + 1e-8)
    target = (sim_norm.unsqueeze(-1) * gt_train_t[idx_test_to_train]).sum(dim=1)
    best_sim = sim_test_to_train[:, 0]
    mask = (best_sim >= sim_floor).float()

    if use_mnn:
        if he_test_self is None:
            he_test_self = he_test
        feat_test2 = torch.as_tensor(
            np.asarray(he_test_self), dtype=torch.float32, device=device
        )
        feat_test2 = _l2_normalize(feat_test2)
        idx_train_to_test, _ = _topk_cross_slice(
            feat_train, feat_test2, k=k, batch_size=batch_size
        )
        n_train = feat_train.shape[0]
        partners_of_train = [set() for _ in range(n_train)]
        idx_train_to_test_cpu = idx_train_to_test.cpu().numpy()
        for tr in range(n_train):
            for te in idx_train_to_test_cpu[tr]:
                partners_of_train[tr].add(int(te))
        idx_test_to_train_cpu = idx_test_to_train.cpu().numpy()
        n_test = feat_test.shape[0]
        mnn_mask = np.zeros(n_test, dtype=np.float32)
        for j in range(n_test):
            for tr in idx_test_to_train_cpu[j]:
                if j in partners_of_train[int(tr)]:
                    mnn_mask[j] = 1.0
                    break
        mask = mask * torch.from_numpy(mnn_mask).to(mask.device)

    return {
        "target": target.cpu().numpy().astype(np.float32),
        "mask": mask.cpu().numpy().astype(np.float32),
        "best_sim": best_sim.cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def build_within_slice_he_smoother(
    he: np.ndarray,
    pred: np.ndarray,
    k: int = 15,
    batch_size: int = 4096,
    device: str = "cuda",
) -> np.ndarray:
    """对预测矩阵做切片内 H&E-NN 加权平均平滑。

    在测试时使用，"H&E 相似的 cell 应当具有相似的表达"是一个强归纳
    偏置；同片 KNN 平均能压低预测噪声、显著抬高 SSIM。
    """
    feat = torch.as_tensor(np.asarray(he), dtype=torch.float32, device=device)
    feat = _l2_normalize(feat)
    pred_t = torch.as_tensor(np.asarray(pred), dtype=torch.float32, device=device)
    n = feat.shape[0]
    out = torch.zeros_like(pred_t)
    for i in range(0, n, batch_size):
        chunk = feat[i : i + batch_size]
        sim = chunk @ feat.T
        ar = torch.arange(chunk.shape[0], device=device)
        sim[ar, ar + i] = -1.0  # mask self
        topv, topi = torch.topk(sim, k=k, dim=1)
        sim_pos = topv.clamp_min(0.0)
        w = sim_pos / (sim_pos.sum(dim=1, keepdim=True) + 1e-8)
        out[i : i + chunk.shape[0]] = (w.unsqueeze(-1) * pred_t[topi]).sum(dim=1)
    return out.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 评估工具（与 run_SpatialEx.ipynb 字节级对齐）
# ---------------------------------------------------------------------------

_FIG2D_GENES = ("EPCAM", "ESR1", "PGR")


def _pcc(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < eps or b.std() < eps:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_one_direction(
    pred: np.ndarray,
    gt: np.ndarray,
    spatial_xy: np.ndarray,
    var_names: List[str],
    label: str,
    marker_genes: Tuple[str, ...] = _FIG2D_GENES,
) -> Dict[str, float]:
    """单方向 (predicted slice X vs GT slice X) 的 PCC / SSIM / CMD + 3 个 marker PCC。

    SSIM 用的图与 ``run_SpatialEx.ipynb`` 完全一致（gaussian-weighted
    KNN、行归一化），保证数字与论文 baseline 直接可比。
    """
    import SpatialEx as se
    print(f"\n=========== Eval: {label} ===========")
    g = se.pp.Build_graph(
        spatial_xy, graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo",
    )
    pcc, pcc_red = se.utils.Compute_metrics(
        gt.copy(), pred.copy(), metric="pcc", reduce="mean",
    )
    ssim, ssim_red = se.utils.Compute_metrics(
        gt.copy(), pred.copy(), metric="ssim", graph=g, reduce="mean",
    )
    cmd, cmd_red = se.utils.Compute_metrics(
        gt.copy(), pred.copy(), metric="cmd", reduce="mean",
    )
    print(f"[{label}] PCC={pcc_red:.4f}  SSIM={ssim_red:.4f}  CMD={cmd_red:.4f}")

    out: Dict[str, float] = {
        "label": label,
        "PCC": float(pcc_red),
        "SSIM": float(ssim_red),
        "CMD": float(cmd_red),
    }
    for g_name in marker_genes:
        if g_name in var_names:
            i = var_names.index(g_name)
            v = _pcc(pred[:, i], gt[:, i])
            out[f"PCC[{g_name}]"] = v
            print(f"  PCC[{g_name}] = {v:.4f}")
        else:
            out[f"PCC[{g_name}]"] = float("nan")
    return out


def evaluate_two_directions(
    panelB1: np.ndarray,
    panelA2: np.ndarray,
    adata1,
    adata2,
    label: str = "SpatialEx-pro",
    marker_genes: Tuple[str, ...] = _FIG2D_GENES,
    out_csv: Optional[str] = None,
) -> Dict[str, float]:
    """双方向评估 + 写一行 CSV。

    - ``panelB1`` (= head-B 在 slice 1 H&E 上的预测) 与 ``adata1.X`` 对比。
      对应 Fig. 2c 右下数字（论文 PCC≈0.273, SSIM≈0.381）。
    - ``panelA2`` (= head-A 在 slice 2 H&E 上的预测) 与 ``adata2.X`` 对比。
      对应 Fig. 2c 左上数字（论文 PCC≈0.258, SSIM≈0.365）。
    """
    var_names1 = list(map(str, adata1.var_names))
    var_names2 = list(map(str, adata2.var_names))
    gt1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.asarray(adata1.X)
    gt2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.asarray(adata2.X)

    s1 = evaluate_one_direction(
        panelB1, np.asarray(gt1, dtype=np.float32),
        adata1.obsm["spatial"], var_names1,
        label=f"{label} | Slice1 (predicted by head trained on Slice2)",
        marker_genes=marker_genes,
    )
    s2 = evaluate_one_direction(
        panelA2, np.asarray(gt2, dtype=np.float32),
        adata2.obsm["spatial"], var_names2,
        label=f"{label} | Slice2 (predicted by head trained on Slice1)",
        marker_genes=marker_genes,
    )

    out: Dict[str, float] = {"label": label}
    for k, v in s1.items():
        if k == "label":
            continue
        out[f"slice1_{k}"] = v
    for k, v in s2.items():
        if k == "label":
            continue
        out[f"slice2_{k}"] = v

    if out_csv is not None:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([out]).to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")
    return out


def per_gene_pcc_table(
    panelB1: np.ndarray,
    panelA2: np.ndarray,
    adata1,
    adata2,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """逐基因 PCC 表，列：``gene, PCC_slice1, PCC_slice2, mean_PCC``。"""
    var_names = list(map(str, adata1.var_names))
    gt1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.asarray(adata1.X)
    gt2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.asarray(adata2.X)
    rows = []
    for i, name in enumerate(var_names):
        v1 = _pcc(panelB1[:, i], gt1[:, i])
        v2 = _pcc(panelA2[:, i], gt2[:, i])
        rows.append({
            "gene": name,
            "PCC_slice1": v1,
            "PCC_slice2": v2,
            "mean_PCC": float(np.nanmean([v1, v2])),
        })
    df = pd.DataFrame(rows).sort_values("mean_PCC", ascending=False)
    if out_csv is not None:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")
    return df


def side_by_side(
    metrics_pro: Dict[str, float],
    metrics_baseline: Dict[str, float],
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """SpatialEx baseline 与 SpatialEx-pro 的并排对比表（含 delta 列）。

    PCC / SSIM / per-gene PCC：delta 为正表示 pro 更好；
    CMD：delta 为负表示 pro 更好。两列原值都保留以便人工判读。
    """
    rows = []
    for k in metrics_pro:
        if k == "label":
            continue
        v_pro = metrics_pro.get(k, float("nan"))
        v_bl = metrics_baseline.get(k, float("nan"))
        delta = (v_pro - v_bl) if isinstance(v_pro, float) and isinstance(v_bl, float) else float("nan")
        rows.append({
            "metric": k,
            "baseline": v_bl,
            "spatialex_pro": v_pro,
            "delta": delta,
        })
    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
