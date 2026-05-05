"""SpatialEx+-EGRefiner 核心模块（EGGN-inspired Exemplar-Guided Graph Refiner）。

作用
----
本文件实现 SpatialEx+-EGRefiner 算法的核心模型与训练器，是整个 ``gpt``
包的算法层。当 SpatialEx+ 训练完成、direct / indirect 预测被缓存为
``.npy`` 之后，本模块负责为 **缺失目标 panel 的 query 切片** 学习一个
轻量的 refiner，把以下三路信号融合成最终预测：

    1. query 自身的 SpatialEx+ direct 预测 ``y_direct``；
    2. query 自身的 SpatialEx+ indirect 预测 ``y_indirect``；
    3. 来自 **真正测过目标 panel** 的另一切片中、形态学上最相似的 K
       个 *exemplar* 细胞的 **真实表达** ``y_e``。

模型由四个组件构成（对应 EGGN 论文）：

* :class:`ExemplarGraphBuilder`      —— exemplar 检索 + 四向异构图构建
                                       (query<->query, exemplar<->exemplar,
                                       query<-exemplar, exemplar<-query)，
                                       使用 GPU 批量化 KNN 加速。
* :class:`GraphSAGEBlock`            —— 同集合内部消息传递（线性分解
                                       结构以避免 ``torch.cat`` 引发的
                                       OOM）。
* :class:`GEBBlock`                  —— Graph Exemplar Bridging：通过
                                       门控机制把 exemplar 特征与
                                       exemplar 真实表达注入到 query。
* :class:`ExemplarAttentionPrediction` —— 对 K 个 exemplar 做注意力
                                       聚合，并以 ``safe-residual`` 头部
                                       输出 ``y_refined = softplus(gate *
                                       y_direct + (1-gate) * y_indirect +
                                       delta)``。

:class:`EGRefinerTrainer` 把以上组件串起来，**只** 在「真正测过目标 panel
的那一片」上训练，保证全过程不偷看 query 切片缺失 panel 的 ground truth。

工程取舍
--------
* 不依赖 ``torch_geometric`` / ``torch_scatter``；scatter 由
  ``torch.Tensor.index_add_`` 实现，CPU/CUDA 通用。
* 默认开启 mixed precision（``--amp bf16/fp16``）+ 可选 gradient
  checkpoint，在 80GB GPU 上可处理 16w 细胞规模。
* 稀疏邻接乘法显式包在 ``with autocast(enabled=False):`` 中，避免
  ``"addmm_sparse_cuda" not implemented for 'BFloat16'`` 报错。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Scatter primitives (no torch_scatter dependency)
# =============================================================================

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """``out[i] = sum_{j: index[j]==i} src[j]``  with ``out.shape == (dim_size, *src.shape[1:])``.

    Parameters
    ----------
    src : (E, ...) tensor.
    index : (E,) int64 tensor with values in ``[0, dim_size)``.
    dim_size : output dim along axis 0.
    """
    if src.dim() == 1:
        out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """``out[i] = mean_{j: index[j]==i} src[j]``."""
    out_sum = scatter_add(src, index, dim_size)
    ones = torch.ones(index.shape[0], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, ones)
    if src.dim() == 1:
        return out_sum / count.clamp_min(1.0)
    return out_sum / count.clamp_min(1.0).view(-1, *([1] * (src.dim() - 1)))


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int,
                    eps: float = 1e-12) -> torch.Tensor:
    """Per-group softmax: returns weights of the same length as ``scores``.

    For each unique group ``g`` (rows where ``index == g``) the returned
    weights sum to 1.

    Subtracting a global maximum is sufficient for numerical stability and
    avoids any dependency on the (PyTorch 1.13+) ``scatter_reduce_(...,
    reduce='amax')`` path -- only :py:meth:`torch.Tensor.index_add_` is used,
    which is supported on all recent PyTorch versions.
    """
    if scores.numel() > 0:
        # Subtracting a constant from all scores does not change the
        # per-group softmax output, so this is exact.
        scores = scores - scores.detach().max()
    exp_scores = torch.exp(scores)
    sum_per = scatter_add(exp_scores, index, dim_size)
    return exp_scores / (sum_per[index] + eps)


# =============================================================================
# Knn / retrieval helpers
# =============================================================================

def _safe_pca(features: np.ndarray, n_components: int, seed: int = 0) -> np.ndarray:
    """PCA project features to ``n_components`` (capped by min(n_components, dim, n-1))."""
    if features.shape[1] <= n_components:
        return features.astype(np.float32, copy=False)
    n_components = min(n_components, features.shape[1], max(features.shape[0] - 1, 1))
    p = PCA(n_components=n_components, svd_solver="randomized",
            random_state=seed, whiten=False)
    return p.fit_transform(features).astype(np.float32, copy=False)


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def make_retrieval_features(
    he: np.ndarray,
    direct: np.ndarray,
    indirect: np.ndarray,
    pca_dim: int = 64,
    seed: int = 0,
    use_he: bool = True,
) -> np.ndarray:
    """Build a per-cell retrieval feature.

    The feature is the *concatenation* of the (z-scored) UNI/H&E embedding,
    a (PCA-reduced, z-scored) direct prediction and a (PCA-reduced,
    z-scored) indirect prediction.  This is the same feature the EGGN
    paper uses for exemplar retrieval -- in our setting the direct /
    indirect predictions take the role of EGGN's ResNet feature in the
    "retrieve-by-prediction" trick.
    """
    parts: List[np.ndarray] = []
    if use_he and he is not None and he.shape[1] > 0:
        # PCA the HE only if it is huge (UNI = 1024 dim -> usable directly).
        parts.append(_zscore(he))
    if direct is not None and direct.shape[1] > 0:
        d = _safe_pca(direct, pca_dim, seed=seed)
        parts.append(_zscore(d))
    if indirect is not None and indirect.shape[1] > 0:
        d = _safe_pca(indirect, pca_dim, seed=seed + 1)
        parts.append(_zscore(d))
    if not parts:
        raise ValueError("make_retrieval_features: no usable features provided")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _metric_name_for_sklearn(metric: str) -> Tuple[str, dict]:
    metric = metric.lower()
    if metric in ("l1", "manhattan", "cityblock"):
        return "manhattan", {}
    if metric in ("l2", "euclidean"):
        return "euclidean", {}
    if metric == "cosine":
        return "cosine", {}
    raise ValueError(f"unknown retrieval metric: {metric}")


def _knn_topk_sklearn(
    query_features: np.ndarray,
    ref_features: np.ndarray,
    k: int,
    metric: str,
    exclude_self: bool,
) -> np.ndarray:
    """Original CPU path -- fast on low-dim / small problems (e.g. spatial 2D)."""
    sk_metric, sk_kwargs = _metric_name_for_sklearn(metric)
    same = exclude_self and (query_features.shape[0] == ref_features.shape[0])
    n_search = k + 1 if same else k
    n_search = min(n_search, ref_features.shape[0])
    nn = NearestNeighbors(n_neighbors=n_search, metric=sk_metric, **sk_kwargs)
    nn.fit(ref_features)
    _, idx = nn.kneighbors(query_features)
    if same:
        n = idx.shape[0]
        rows = np.arange(n)[:, None]
        mask = (idx != rows)
        order = np.argsort((~mask).astype(np.int8), axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, axis=1)[:, :k]
    return idx.astype(np.int64, copy=False)


def _knn_topk_torch(
    query_features: np.ndarray,
    ref_features: np.ndarray,
    k: int,
    metric: str,
    exclude_self: bool,
    device: str,
    q_batch: int = 512,
    r_batch: int = 4096,
    verbose: bool = True,
) -> np.ndarray:
    """GPU-batched top-k retrieval that supports L1, L2, and cosine.

    Memory layout:
        * cosine -- a single ``(b, n_r)`` similarity matrix per query batch
          (~``b * n_r * 4`` bytes).  Uses one matmul, very fast.
        * L1/L2  -- nested loop over query and reference batches; we keep a
          running ``(b, k)`` top-k accumulator so that no intermediate is
          larger than ``b * r_batch * 4`` bytes.

    Notes
    -----
    With ~160k cells x 1152 retrieval dims, the original sklearn BallTree
    fit/predict path takes >30 minutes on CPU; this GPU path takes ~1 minute
    on a single A100.
    """
    metric = metric.lower()
    if metric not in ("l1", "manhattan", "l2", "euclidean", "cosine"):
        raise ValueError(f"unknown retrieval metric: {metric}")

    same = exclude_self and (query_features.shape[0] == ref_features.shape[0])
    n_q = query_features.shape[0]
    n_r = ref_features.shape[0]
    k_eff = min(k, n_r - (1 if same else 0))

    q = torch.as_tensor(np.ascontiguousarray(query_features, dtype=np.float32),
                        device=device)
    r = torch.as_tensor(np.ascontiguousarray(ref_features, dtype=np.float32),
                        device=device)
    if metric == "cosine":
        q = F.normalize(q, p=2, dim=1)
        r = F.normalize(r, p=2, dim=1)

    out_idx = torch.empty(n_q, k_eff, dtype=torch.int64, device=device)

    if metric == "cosine":
        # one matmul per query-batch (n_r is the "columns" of the dist matrix)
        bq_cosine = max(q_batch, 4096)
        n_batches = (n_q + bq_cosine - 1) // bq_cosine
        for bi, i in enumerate(range(0, n_q, bq_cosine)):
            chunk = q[i : i + bq_cosine]
            bq = chunk.shape[0]
            sim = chunk @ r.T                       # (bq, n_r)
            d = -sim
            if same:
                qi = torch.arange(i, i + bq, device=device).view(-1, 1)
                ri = torch.arange(n_r, device=device).view(1, -1)
                d = torch.where(qi == ri, torch.full_like(d, float("inf")), d)
            _, topk_idx = torch.topk(d, k=k_eff, dim=1, largest=False)
            out_idx[i : i + bq] = topk_idx
            if verbose and (bi % 4 == 0 or bi == n_batches - 1):
                print(f"    [knn-cosine] batch {bi+1}/{n_batches}", flush=True)
    else:
        p_norm = 1.0 if metric in ("l1", "manhattan") else 2.0
        n_q_batches = (n_q + q_batch - 1) // q_batch
        for bi, i in enumerate(range(0, n_q, q_batch)):
            q_chunk = q[i : i + q_batch]
            bq = q_chunk.shape[0]
            best_d = torch.full((bq, k_eff), float("inf"), device=device)
            best_idx = torch.zeros((bq, k_eff), dtype=torch.int64, device=device)
            for j in range(0, n_r, r_batch):
                r_chunk = r[j : j + r_batch]
                br = r_chunk.shape[0]
                # cdist(p=1) is memory-heavy on high-dim, but with bq=512 and
                # br=4096 the (bq, br, d) intermediate stays at ~8 GB which
                # fits on a 24 GB GPU; reduce q_batch / r_batch if you OOM.
                d = torch.cdist(q_chunk, r_chunk, p=p_norm)         # (bq, br)
                if same:
                    qi = torch.arange(i, i + bq, device=device).view(-1, 1)
                    ri = torch.arange(j, j + br, device=device).view(1, -1)
                    d = torch.where(qi == ri, torch.full_like(d, float("inf")), d)
                ref_idx = torch.arange(j, j + br, device=device).expand(bq, -1)
                combined_d = torch.cat([best_d, d], dim=1)
                combined_idx = torch.cat([best_idx, ref_idx], dim=1)
                topk_vals, topk_pos = torch.topk(combined_d, k=k_eff, dim=1,
                                                  largest=False)
                best_d = topk_vals
                best_idx = torch.gather(combined_idx, 1, topk_pos)
            out_idx[i : i + bq] = best_idx
            if verbose and (bi % 4 == 0 or bi == n_q_batches - 1):
                print(f"    [knn-{metric}] q-batch {bi+1}/{n_q_batches}", flush=True)

    return out_idx.detach().cpu().numpy().astype(np.int64, copy=False)


def knn_topk(
    query_features: np.ndarray,
    ref_features: np.ndarray,
    k: int,
    metric: str = "l1",
    exclude_self: bool = False,
    device: str = "cpu",
    q_batch: int = 512,
    r_batch: int = 2048,
) -> np.ndarray:
    """Top-K retrieval with automatic CPU/GPU dispatch.

    Heuristic: switch to the GPU-batched path when CUDA is available *and*
    either the user asked for it (``device`` starts with ``cuda``) or the
    problem is large enough that sklearn's BallTree would be slow
    (``n_q * n_r > 1e7`` rows-of-pairwise).  Set ``device='cpu'`` to force
    the sklearn path.
    """
    use_gpu = (
        torch.cuda.is_available()
        and (str(device).startswith("cuda")
             or query_features.shape[0] * ref_features.shape[0] > 1e7)
    )
    if use_gpu:
        gpu_device = device if str(device).startswith("cuda") else "cuda:0"
        return _knn_topk_torch(
            query_features, ref_features, k=k, metric=metric,
            exclude_self=exclude_self, device=gpu_device,
            q_batch=q_batch, r_batch=r_batch,
        )
    return _knn_topk_sklearn(
        query_features, ref_features, k=k, metric=metric,
        exclude_self=exclude_self,
    )


def build_row_normalized_adj(
    features: np.ndarray,
    k: int,
    metric: str = "l1",
    exclude_self: bool = True,
    add_self_loop: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Build a sparse row-normalized adjacency for mean aggregation.

    The output is a torch sparse COO tensor on ``device`` whose rows sum to
    1 (numerically).  ``GraphSAGEBlock`` applies it as ``A @ h`` to obtain
    mean-aggregated neighbor features.
    """
    n = features.shape[0]
    # 2D spatial coordinates -> sklearn (BallTree is fast); high-dim -> GPU.
    use_cpu = features.shape[1] <= 8 and not str(device).startswith("cuda")
    if use_cpu:
        idx = _knn_topk_sklearn(features, features, k=k, metric=metric,
                                exclude_self=exclude_self)
    else:
        idx = knn_topk(features, features, k=k, metric=metric,
                       exclude_self=exclude_self, device=device)
    rows = np.repeat(np.arange(n), idx.shape[1]).astype(np.int64)
    cols = idx.reshape(-1).astype(np.int64)
    data = np.ones_like(rows, dtype=np.float32)
    if add_self_loop:
        rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
        cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
        data = np.concatenate([data, np.ones(n, dtype=np.float32)])
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = A.tocsr()
    rs = np.asarray(A.sum(axis=1)).reshape(-1)
    rs = np.where(rs > 0, 1.0 / rs, 0.0).astype(np.float32)
    A = sp.diags(rs) @ A
    A = A.tocoo()
    indices = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    values = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to(device)


def build_query_exemplar_edge_index(
    query_retrieval: np.ndarray,
    exemplar_retrieval: np.ndarray,
    k: int,
    metric: str = "l1",
    exclude_self: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a ``(2, n_q * k)`` edge_index ``[query_idx; exemplar_idx]``."""
    n_q = query_retrieval.shape[0]
    idx = knn_topk(query_retrieval, exemplar_retrieval, k=k, metric=metric,
                   exclude_self=exclude_self, device=device)
    src_q = np.repeat(np.arange(n_q, dtype=np.int64), idx.shape[1])
    src_e = idx.reshape(-1).astype(np.int64)
    edge = torch.from_numpy(np.stack([src_q, src_e], axis=0))
    return edge.to(device)


def _build_query_exemplar_edge_with_batch(
    query_retrieval: np.ndarray,
    exemplar_retrieval: np.ndarray,
    k: int,
    metric: str,
    exclude_self: bool,
    device: str,
    q_batch: int,
    r_batch: int,
) -> torch.Tensor:
    """Same as :func:`build_query_exemplar_edge_index` but lets the caller
    tune the GPU batch sizes (used by the trainer)."""
    n_q = query_retrieval.shape[0]
    use_gpu = (
        torch.cuda.is_available()
        and (str(device).startswith("cuda")
             or query_retrieval.shape[0] * exemplar_retrieval.shape[0] > 1e7)
    )
    if use_gpu:
        gpu_device = device if str(device).startswith("cuda") else "cuda:0"
        idx = _knn_topk_torch(
            query_retrieval, exemplar_retrieval, k=k, metric=metric,
            exclude_self=exclude_self, device=gpu_device,
            q_batch=q_batch, r_batch=r_batch,
        )
    else:
        idx = _knn_topk_sklearn(
            query_retrieval, exemplar_retrieval, k=k, metric=metric,
            exclude_self=exclude_self,
        )
    src_q = np.repeat(np.arange(n_q, dtype=np.int64), idx.shape[1])
    src_e = idx.reshape(-1).astype(np.int64)
    edge = torch.from_numpy(np.stack([src_q, src_e], axis=0))
    return edge.to(device)


# =============================================================================
# Building blocks
# =============================================================================

class GraphSAGEBlock(nn.Module):
    """Mean-aggregation GraphSAGE layer.

    Mirrors ``torch_geometric.nn.SAGEConv`` at the operator level:
    ``h' = LeakyReLU( BN( Linear( concat(h, mean_neighbors(h)) ) ) )``.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1,
                 negative_slope: float = 0.2):
        super().__init__()
        # ``Linear(2*in_dim, out_dim)(cat([h, agg]))`` decomposed as
        # ``Linear_self(h) + Linear_neigh(agg)`` -- mathematically identical
        # but avoids the (n, 2*in_dim) intermediate.  At n=161k, hidden=512
        # this saves ~660 MB per layer (×6 layers + backward = a few GB).
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        # ``torch.sparse.mm`` ultimately dispatches to ``addmm``, which IS in
        # the autocast op list.  Under bf16 autocast PyTorch tries to cast
        # both operands to bf16 and then calls ``addmm_sparse_cuda``, which
        # has no bf16 kernel and crashes.  We therefore force the spmm to
        # run in fp32 with autocast disabled, regardless of ``h``'s dtype,
        # and cast the dense aggregate back to ``h.dtype`` for the
        # subsequent linear layers (which DO autocast cleanly).
        h_dtype = h.dtype
        with torch.cuda.amp.autocast(enabled=False):
            agg_fp32 = torch.sparse.mm(adj, h.float())
        agg = agg_fp32.to(h_dtype)
        del agg_fp32
        x = self.lin_self(h) + self.lin_neigh(agg)
        del agg
        if x.shape[0] > 1:
            x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GEBBlock(nn.Module):
    """Graph Exemplar Bridging (EGGN-style) for our cross-panel setting.

    Logic
    -----
    For each query-exemplar edge ``(q, e)`` with ``y_e`` being the *measured*
    target-panel expression of the exemplar:

    1. ``s_e = MLP_expr(y_e)``                    -- project the exemplar
                                                    measured expression into
                                                    the hidden space.
    2. ``diff = h_q - h_e``                        -- query/exemplar mismatch.
    3. ``gate = sigmoid(MLP_gate(diff))``          -- per-channel gate.
    4. ``msg_q = MLP_q(concat(h_q, h_e, s_e)) * gate``
       ``msg_e = MLP_e(concat(h_e, h_q, s_e)) * gate``  (optional)
    5. Aggregate per query / per exemplar with mean, residual-add and
       LayerNorm.  This makes the block stable across multi-layer stacks.
    """

    def __init__(self, hidden_dim: int, expr_dim: int,
                 dropout: float = 0.1, update_exemplar: bool = True):
        super().__init__()
        self.update_exemplar = update_exemplar
        self.proj_y = nn.Linear(expr_dim, hidden_dim, bias=False)
        # ``gate_mlp`` operates on a single (num_edges, hidden) tensor (the
        # ``diff = h_q - h_e``), so no special decomposition is needed.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # The original formulation was
        #     ``Linear(3*hidden -> hidden)(cat([h_q, h_e, s_e]))``
        # which materialises a ``(num_edges, 3*hidden)`` tensor.  At
        # ``num_edges = n*k = 970k`` and ``hidden = 512``, that single
        # intermediate is 6 GB, and with the chained Linear/LeakyReLU/Linear
        # stack each layer ate ~10 GB of activations.  We keep the math
        # exactly identical (an affine map of the concatenation is a sum of
        # affine maps over the parts) but evaluate it as three separate
        # ``Linear(hidden, hidden)`` calls whose outputs are summed in
        # place.  This avoids the (num_edges, 3*hidden) intermediate and
        # roughly halves activation memory per GEB layer.
        self.q_msg_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.q_msg_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_msg_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_msg_act = nn.LeakyReLU(0.2, inplace=True)
        self.q_msg_out = nn.Linear(hidden_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        if update_exemplar:
            self.e_msg_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.e_msg_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.e_msg_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.e_msg_act = nn.LeakyReLU(0.2, inplace=True)
            self.e_msg_out = nn.Linear(hidden_dim, hidden_dim)
            self.e_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_q: torch.Tensor,
        h_e: torch.Tensor,
        y_e: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_q = h_q.shape[0]
        n_e = h_e.shape[0]
        s_e = self.proj_y(y_e)              # (n_e, hidden)

        src_q = edge_index[0]
        src_e = edge_index[1]

        # Pre-gather the per-edge source tensors -- each is (num_edges, hidden).
        h_q_e = h_q.index_select(0, src_q)
        h_e_e = h_e.index_select(0, src_e)
        s_e_e = s_e.index_select(0, src_e)

        # gate = sigmoid(MLP(h_q - h_e)) -- per-edge per-channel mask.
        diff = h_q_e - h_e_e
        gate = torch.sigmoid(self.gate_mlp(diff))
        del diff

        # q_msg: equivalent to ``Linear(3h -> h)(cat([h_q, h_e, s_e]))`` but
        # without the (num_edges, 3*hidden) intermediate.
        msg_q = self.q_msg_q(h_q_e)
        msg_q = msg_q + self.q_msg_e(h_e_e)
        msg_q = msg_q + self.q_msg_s(s_e_e)
        msg_q = self.q_msg_act(msg_q)
        msg_q = self.q_msg_out(msg_q)
        msg_q = msg_q * gate
        out_q = scatter_mean(msg_q, src_q, n_q)
        del msg_q
        out_q = self.dropout(out_q)
        h_q_new = self.q_norm(h_q + out_q)

        h_e_new = h_e
        if self.update_exemplar:
            msg_e = self.e_msg_e(h_e_e)
            msg_e = msg_e + self.e_msg_q(h_q_e)
            msg_e = msg_e + self.e_msg_s(s_e_e)
            msg_e = self.e_msg_act(msg_e)
            msg_e = self.e_msg_out(msg_e)
            msg_e = msg_e * gate
            out_e = scatter_mean(msg_e, src_e, n_e)
            del msg_e
            out_e = self.dropout(out_e)
            h_e_new = self.e_norm(h_e + out_e)

        return h_q_new, h_e_new


class ExemplarAttentionPrediction(nn.Module):
    """Attention-over-K-exemplars head with a learned base/residual decomposition.

    Two modes controlled by ``safe_residual``:

    * **default** (``safe_residual=False``):
      ``y_refined = softplus(gate * y_direct + (1 - gate) * y_indirect + delta)``

    * **safe residual** (``safe_residual=True``):
      ``y_refined = relu(y_indirect + alpha * delta)``
      where ``alpha`` is a **learned per-gene scalar initialised to 0**.  At
      the start of training this is *exactly* ``relu(y_indirect)`` = ``raw``,
      so the refiner **cannot be worse than raw until it learns something
      useful**.  ``alpha`` gradually grows only for genes where the exemplar
      context contains useful complementary information.
    """

    def __init__(self, hidden_dim: int, expr_dim_target: int,
                 dropout: float = 0.1, safe_residual: bool = False):
        super().__init__()
        self.safe_residual = safe_residual
        self.proj_y = nn.Linear(expr_dim_target, hidden_dim, bias=False)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_act = nn.LeakyReLU(0.2, inplace=True)
        self.attn_out = nn.Linear(hidden_dim, 1)
        self.delta_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.delta_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.delta_act = nn.LeakyReLU(0.2, inplace=True)
        self.delta_drop = nn.Dropout(dropout)
        self.delta_out = nn.Linear(hidden_dim, expr_dim_target)
        if safe_residual:
            # Per-gene scaling factor initialised at 0 → starts at raw.
            self.alpha = nn.Parameter(torch.zeros(expr_dim_target))
        else:
            self.gate_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.gate_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_act = nn.LeakyReLU(0.2, inplace=True)
            self.gate_drop = nn.Dropout(dropout)
            self.gate_out = nn.Linear(hidden_dim, expr_dim_target)
        # Zero-init the last linear of the delta head so the initial delta ≈ 0.
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    def forward(
        self,
        h_q: torch.Tensor,
        h_e: torch.Tensor,
        y_e: torch.Tensor,
        edge_index: torch.Tensor,
        y_direct: torch.Tensor,
        y_indirect: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_q = h_q.shape[0]
        s_e = self.proj_y(y_e)

        src_q = edge_index[0]
        src_e = edge_index[1]

        h_q_e = h_q.index_select(0, src_q)
        h_e_e = h_e.index_select(0, src_e)
        s_e_e = s_e.index_select(0, src_e)

        a = self.attn_q(h_q_e)
        a = a + self.attn_e(h_e_e)
        a = a + self.attn_d(h_q_e - h_e_e)
        a = a + self.attn_s(s_e_e)
        a = self.attn_act(a)
        scores = self.attn_out(a).squeeze(-1)
        del a
        weights = scatter_softmax(scores, src_q, n_q)
        weighted = weights.unsqueeze(-1) * h_e_e
        context = scatter_add(weighted, src_q, n_q)
        del h_q_e, h_e_e, s_e_e, weighted

        d = self.delta_q(h_q) + self.delta_c(context)
        d = self.delta_act(d)
        d = self.delta_drop(d)
        delta = self.delta_out(d)
        del d

        if self.safe_residual:
            # alpha starts at 0 → y_refined starts at relu(y_indirect) = raw.
            # As alpha grows the model mixes in the learned correction.
            gate = self.alpha.sigmoid().unsqueeze(0)     # for logging compat
            y_refined = F.relu(y_indirect + self.alpha * delta)
        else:
            g = self.gate_q(h_q) + self.gate_c(context)
            g = self.gate_act(g)
            g = self.gate_drop(g)
            gate = torch.sigmoid(self.gate_out(g))
            del g
            y_base = gate * y_direct + (1.0 - gate) * y_indirect
            y_refined = F.softplus(y_base + delta)
        return y_refined, weights, gate


class EGRefiner(nn.Module):
    """Full SpatialExP-EGRefiner network.

    Inputs at forward time:

    * ``query_he``         -- ``(n_q, d_he)``
    * ``query_direct``     -- ``(n_q, expr_dim_target)`` SpatialEx+ direct
    * ``query_indirect``   -- ``(n_q, expr_dim_target)`` SpatialEx+ indirect
    * ``exemplar_he``      -- ``(n_e, d_he)``
    * ``exemplar_direct``  -- ``(n_e, expr_dim_target)``
    * ``exemplar_indirect``-- ``(n_e, expr_dim_target)``
    * ``exemplar_expr``    -- ``(n_e, expr_dim_target)`` *measured* target panel
    * ``adj_qq``           -- sparse query-query adjacency (row-normalized)
    * ``adj_ee``           -- sparse exemplar-exemplar adjacency (row-norm.)
    * ``edge_qe``          -- ``(2, E)`` query-exemplar edge_index
    """

    def __init__(
        self,
        d_he: int,
        expr_dim_target: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        update_exemplar: bool = True,
        gradient_checkpoint: bool = False,
        safe_residual: bool = False,
    ):
        super().__init__()
        self.d_he = d_he
        self.expr_dim_target = expr_dim_target
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gradient_checkpoint = gradient_checkpoint
        self.safe_residual = safe_residual

        # query side: HE + direct + indirect  -> hidden
        q_in = d_he + 2 * expr_dim_target
        self.q_proj = nn.Sequential(
            nn.Linear(q_in, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
        )
        # exemplar side: HE + direct + indirect + measured expression -> hidden
        e_in = d_he + 3 * expr_dim_target
        self.e_proj = nn.Sequential(
            nn.Linear(e_in, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
        )

        self.q_sage = nn.ModuleList([GraphSAGEBlock(hidden_dim, hidden_dim, dropout)
                                     for _ in range(num_layers)])
        self.e_sage = nn.ModuleList([GraphSAGEBlock(hidden_dim, hidden_dim, dropout)
                                     for _ in range(num_layers)])
        self.geb = nn.ModuleList([GEBBlock(hidden_dim, expr_dim_target,
                                           dropout=dropout,
                                           update_exemplar=update_exemplar)
                                  for _ in range(num_layers)])

        self.head = ExemplarAttentionPrediction(
            hidden_dim, expr_dim_target, dropout, safe_residual=safe_residual,
        )

    def encode(
        self,
        query_he: torch.Tensor,
        query_direct: torch.Tensor,
        query_indirect: torch.Tensor,
        exemplar_he: torch.Tensor,
        exemplar_direct: torch.Tensor,
        exemplar_indirect: torch.Tensor,
        exemplar_expr: torch.Tensor,
        adj_qq: torch.sparse.Tensor,
        adj_ee: torch.sparse.Tensor,
        edge_qe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_q = self.q_proj(torch.cat([query_he, query_direct, query_indirect], dim=-1))
        h_e = self.e_proj(torch.cat(
            [exemplar_he, exemplar_direct, exemplar_indirect, exemplar_expr],
            dim=-1,
        ))
        use_ckpt = self.gradient_checkpoint and self.training
        for q_sage, e_sage, geb in zip(self.q_sage, self.e_sage, self.geb):
            h_q = q_sage(h_q, adj_qq)
            h_e = e_sage(h_e, adj_ee)
            if use_ckpt:
                # Checkpoint only the GEB block: the per-edge tensors inside
                # GEB (h_q[src_q], h_e[src_e], s_e[src_e], etc.) are by far
                # the largest activations, and GEB only contains LayerNorm
                # / Linear / Dropout (no BatchNorm), so re-running it inside
                # backward does not corrupt running statistics.
                h_q, h_e = torch.utils.checkpoint.checkpoint(
                    geb, h_q, h_e, exemplar_expr, edge_qe,
                    use_reentrant=False,
                )
            else:
                h_q, h_e = geb(h_q, h_e, exemplar_expr, edge_qe)
        return h_q, h_e

    def forward(
        self,
        query_he: torch.Tensor,
        query_direct: torch.Tensor,
        query_indirect: torch.Tensor,
        exemplar_he: torch.Tensor,
        exemplar_direct: torch.Tensor,
        exemplar_indirect: torch.Tensor,
        exemplar_expr: torch.Tensor,
        adj_qq: torch.sparse.Tensor,
        adj_ee: torch.sparse.Tensor,
        edge_qe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_q, h_e = self.encode(
            query_he, query_direct, query_indirect,
            exemplar_he, exemplar_direct, exemplar_indirect, exemplar_expr,
            adj_qq, adj_ee, edge_qe,
        )
        y_refined, weights, gate = self.head(
            h_q, h_e, exemplar_expr, edge_qe, query_direct, query_indirect
        )
        return y_refined, weights, gate


# =============================================================================
# Loss helpers
# =============================================================================

def gene_pcc_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  eps: float = 1e-6) -> torch.Tensor:
    """1 - mean per-gene Pearson correlation across cells.

    Genes with ~zero variance on either side are skipped.  ``mask`` is an
    optional boolean cell mask (only the masked rows participate).
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pc = pred - pred.mean(dim=0, keepdim=True)
    tc = target - target.mean(dim=0, keepdim=True)
    num = (pc * tc).sum(dim=0)
    den = torch.sqrt((pc ** 2).sum(dim=0) * (tc ** 2).sum(dim=0) + eps)
    pcc = num / (den + eps)
    valid = (pred.std(dim=0) > eps) & (target.std(dim=0) > eps)
    if not valid.any():
        return pred.new_zeros(())
    return 1.0 - pcc[valid].mean()


def spot_pcc_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  eps: float = 1e-6) -> torch.Tensor:
    """1 - mean per-cell Pearson correlation across genes."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.shape[1] < 2 or pred.shape[0] == 0:
        return pred.new_zeros(())
    pc = pred - pred.mean(dim=1, keepdim=True)
    tc = target - target.mean(dim=1, keepdim=True)
    num = (pc * tc).sum(dim=1)
    den = torch.sqrt((pc ** 2).sum(dim=1) * (tc ** 2).sum(dim=1) + eps)
    pcc = num / (den + eps)
    valid = (pred.std(dim=1) > eps) & (target.std(dim=1) > eps)
    if not valid.any():
        return pred.new_zeros(())
    return 1.0 - pcc[valid].mean()


def graph_smoothness_loss(pred: torch.Tensor, adj: torch.sparse.Tensor,
                          max_edges: int = 200_000) -> torch.Tensor:
    """``mean ||y_i - mean_neighbors(y_j)||^2`` over rows.

    Equivalent (up to a constant) to a TV penalty on the spatial graph.
    """
    # see ``GraphSAGEBlock.forward``: torch.sparse.mm dispatches to addmm
    # which auto-casts under bf16, but the sparse bf16 kernel does not
    # exist.  Force fp32 with autocast disabled.
    p_dtype = pred.dtype
    with torch.cuda.amp.autocast(enabled=False):
        smoothed = torch.sparse.mm(adj, pred.float())
    smoothed = smoothed.to(p_dtype)
    return ((pred - smoothed) ** 2).mean()


def attention_entropy_loss(weights: torch.Tensor,
                            edge_index: torch.Tensor,
                            n_q: int,
                            eps: float = 1e-12) -> torch.Tensor:
    """Negative mean entropy of attention weights per query.

    ``weights`` is the per-edge attention output of ``scatter_softmax``.
    The returned value is ``-H(p)`` so that minimizing it pushes attention
    *away* from a single-exemplar collapse (i.e. encourages higher entropy).
    """
    src_q = edge_index[0]
    contrib = -(weights * torch.log(weights.clamp_min(eps)))
    H = scatter_add(contrib, src_q, n_q)
    # ignore queries that have no edges (shouldn't happen in our setup)
    return -H.mean()


# =============================================================================
# Trainer
# =============================================================================

@dataclass
class EGRefinerConfig:
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    seed: int = 0
    update_exemplar: bool = True
    # retrieval
    k_exemplar: int = 6
    k_query_graph: int = 10
    k_exemplar_graph: int = 10
    retrieval_metric: str = "l1"
    prediction_pca_dim: int = 64
    # KNN batched-cdist sizes (reduce if you OOM on small GPUs).
    # Memory of the inner ``cdist(p=1)`` is about
    # ``knn_q_batch * knn_r_batch * retrieval_dim * 4`` bytes.
    # Defaults: 512 * 2048 * 1152 * 4 ≈ 4.8 GB intermediate.  On a 24 GB
    # GPU you can safely bump to (512, 4096) or (1024, 2048) for speed.
    knn_q_batch: int = 512
    knn_r_batch: int = 2048
    # Memory-saving controls for the refiner forward/backward.
    #     amp_dtype             -- 'none' / 'bf16' / 'fp16'.  bf16 autocast
    #                              halves activation memory at full
    #                              numerical safety on Ampere+; fp16 also
    #                              works but needs GradScaler (we provide it).
    #     gradient_checkpoint   -- when True, checkpoint each GEB block so
    #                              the per-edge tensors are recomputed in
    #                              backward instead of being kept in memory.
    amp_dtype: str = "none"
    gradient_checkpoint: bool = False
    safe_residual: bool = False
    # loss weights
    lambda_mse: float = 1.0
    lambda_gene_pcc: float = 0.5
    lambda_spot_pcc: float = 0.2
    lambda_smooth: float = 0.05
    lambda_attn: float = 0.001
    # train/val split
    val_frac: float = 0.2
    # input perturbation (helps generalization to the cross-slice regime)
    input_noise_std: float = 0.0
    # log
    log_every: int = 10


class EGRefinerTrainer:
    """Train a single direction (Panel A or Panel B) refiner.

    The trainer is symmetric: pass in the query (slice that has the panel
    measured) and exemplar arrays.  At inference time we *re-instantiate* the
    forward graph with a different query / exemplar pair (e.g. query =
    Slice 2, exemplar = Slice 1) and call :meth:`predict`.

    Crucially, the trainer **never** consumes the target slice's missing-panel
    ground truth -- we only ever see the panel that the *training* slice has
    measured.
    """

    def __init__(
        self,
        cfg: EGRefinerConfig,
        train_he: np.ndarray,
        train_direct: np.ndarray,
        train_indirect: np.ndarray,
        train_expr: np.ndarray,                  # measured target panel on training slice
        train_spatial: Optional[np.ndarray] = None,
        device: str = "cuda:0",
        name: str = "RefinerA",
    ):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.name = name

        self.train_he = np.asarray(train_he, dtype=np.float32)
        self.train_direct = np.asarray(train_direct, dtype=np.float32)
        self.train_indirect = np.asarray(train_indirect, dtype=np.float32)
        self.train_expr = np.asarray(train_expr, dtype=np.float32)
        self.train_spatial = (
            np.asarray(train_spatial, dtype=np.float32)
            if train_spatial is not None else None
        )

        n = self.train_he.shape[0]
        assert self.train_direct.shape == (n, self.train_expr.shape[1]), (
            f"[{name}] direct shape mismatch: got {self.train_direct.shape}, "
            f"expected ({n}, {self.train_expr.shape[1]})"
        )
        assert self.train_indirect.shape == (n, self.train_expr.shape[1]), (
            f"[{name}] indirect shape mismatch: got {self.train_indirect.shape}, "
            f"expected ({n}, {self.train_expr.shape[1]})"
        )

        self._set_seed(cfg.seed)

        # ---------- precompute graphs (training side) ----------
        # Retrieval features are HE + PCA(direct) + PCA(indirect).
        retr = make_retrieval_features(
            he=self.train_he,
            direct=self.train_direct,
            indirect=self.train_indirect,
            pca_dim=cfg.prediction_pca_dim,
            seed=cfg.seed,
        )
        # query-query: prefer spatial coordinates if available (much smoother),
        # fall back to retrieval features otherwise.
        if self.train_spatial is not None:
            qq_feat = self.train_spatial
            ee_feat = self.train_spatial
        else:
            qq_feat = retr
            ee_feat = retr
        print(f"[{self.name}] building query-query KNN graph "
              f"(k={cfg.k_query_graph}, dim={qq_feat.shape[1]}) ...", flush=True)
        self.adj_qq = build_row_normalized_adj(
            qq_feat, k=cfg.k_query_graph, metric="l2",
            exclude_self=True, add_self_loop=True, device=str(self.device),
        )
        print(f"[{self.name}] building exemplar-exemplar KNN graph "
              f"(k={cfg.k_exemplar_graph}) ...", flush=True)
        # During training, query==exemplar, so the two graphs are identical.
        self.adj_ee = self.adj_qq

        print(f"[{self.name}] building query-exemplar edges "
              f"(k={cfg.k_exemplar}, metric={cfg.retrieval_metric}, "
              f"dim={retr.shape[1]}) on {self.device} -- "
              f"this is the slow KNN; should be ~1-2 min on a modern GPU "
              f"with q_batch={cfg.knn_q_batch}, r_batch={cfg.knn_r_batch}.",
              flush=True)
        self.edge_qe = _build_query_exemplar_edge_with_batch(
            retr, retr, k=cfg.k_exemplar, metric=cfg.retrieval_metric,
            exclude_self=True, device=str(self.device),
            q_batch=cfg.knn_q_batch, r_batch=cfg.knn_r_batch,
        )

        # ---------- tensors on device ----------
        self.t_he = torch.from_numpy(self.train_he).to(self.device)
        self.t_direct = torch.from_numpy(self.train_direct).to(self.device)
        self.t_indirect = torch.from_numpy(self.train_indirect).to(self.device)
        self.t_expr = torch.from_numpy(self.train_expr).to(self.device)

        # ---------- model ----------
        self.model = EGRefiner(
            d_he=self.train_he.shape[1],
            expr_dim_target=self.train_expr.shape[1],
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            update_exemplar=cfg.update_exemplar,
            gradient_checkpoint=cfg.gradient_checkpoint,
            safe_residual=cfg.safe_residual,
        ).to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        # ---------- mixed precision ----------
        amp = str(cfg.amp_dtype).lower()
        if amp not in ("none", "bf16", "fp16"):
            raise ValueError(f"amp_dtype must be 'none' / 'bf16' / 'fp16', got {amp!r}")
        self.amp_dtype = {
            "none": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[amp]
        # GradScaler is needed for fp16 (loss values get under/overflow); bf16
        # has fp32-equivalent dynamic range and does not need scaling.
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.amp_dtype is torch.float16 and str(self.device).startswith("cuda"):
            self.scaler = torch.cuda.amp.GradScaler()
        print(f"[{self.name}] memory mode: amp={amp}, "
              f"gradient_checkpoint={cfg.gradient_checkpoint}, "
              f"hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}, "
              f"update_exemplar={cfg.update_exemplar}", flush=True)

        # ---------- train / val split ----------
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(cfg.val_frac * n)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        self.train_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.val_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        print(f"[{self.name}] split: train={int(self.train_mask.sum())} "
              f"val={int(self.val_mask.sum())}")

        # ---------- log ----------
        self.train_log: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _maybe_perturb(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.input_noise_std <= 0 or not self.model.training:
            return x
        return x + torch.randn_like(x) * self.cfg.input_noise_std

    # ------------------------------------------------------------------
    # training loop (full-batch)
    # ------------------------------------------------------------------
    def train(self) -> List[Dict[str, float]]:
        cfg = self.cfg
        n = self.t_he.shape[0]
        best_val_pcc = float("-inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None

        amp_enabled = (
            self.amp_dtype is not None and str(self.device).startswith("cuda")
        )
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype)
            if amp_enabled else nullcontext()
        )

        for epoch in range(cfg.epochs):
            self.model.train()
            self.optim.zero_grad()

            q_d = self._maybe_perturb(self.t_direct)
            q_i = self._maybe_perturb(self.t_indirect)
            e_d = self._maybe_perturb(self.t_direct)
            e_i = self._maybe_perturb(self.t_indirect)

            with amp_ctx:
                y_pred, attn_w, gate = self.model(
                    self.t_he, q_d, q_i,
                    self.t_he, e_d, e_i, self.t_expr,
                    self.adj_qq, self.adj_ee, self.edge_qe,
                )

                mse = F.mse_loss(y_pred[self.train_mask], self.t_expr[self.train_mask])
                loss_gene = gene_pcc_loss(y_pred, self.t_expr, mask=self.train_mask)
                if cfg.lambda_spot_pcc > 0:
                    loss_spot = spot_pcc_loss(y_pred, self.t_expr, mask=self.train_mask)
                else:
                    loss_spot = y_pred.new_zeros(())
                if cfg.lambda_smooth > 0:
                    loss_smooth = graph_smoothness_loss(y_pred, self.adj_qq)
                else:
                    loss_smooth = y_pred.new_zeros(())
                if cfg.lambda_attn > 0:
                    loss_attn = attention_entropy_loss(attn_w, self.edge_qe, n)
                else:
                    loss_attn = y_pred.new_zeros(())

                loss = (
                    cfg.lambda_mse * mse
                    + cfg.lambda_gene_pcc * loss_gene
                    + cfg.lambda_spot_pcc * loss_spot
                    + cfg.lambda_smooth * loss_smooth
                    + cfg.lambda_attn * loss_attn
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optim.step()

            log_row = {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "mse": float(mse.item()),
                "gene_pcc_loss": float(loss_gene.item()),
                "spot_pcc_loss": float(loss_spot.item()),
                "smooth_loss": float(loss_smooth.item()),
                "attn_entropy_loss": float(loss_attn.item()),
            }

            # ---- validation (full graph, mask the val cells) ----
            val_pcc = self._eval_val(y_pred)
            log_row["val_gene_pcc_mean"] = val_pcc

            self.train_log.append(log_row)

            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}

            if (epoch + 1) % max(1, cfg.log_every) == 0 or epoch == 0:
                print(
                    f"[{self.name}][ep {epoch+1:4d}/{cfg.epochs}] "
                    f"L={log_row['loss']:.4f}  mse={log_row['mse']:.4f}  "
                    f"gPCC={log_row['gene_pcc_loss']:.4f}  "
                    f"sPCC={log_row['spot_pcc_loss']:.4f}  "
                    f"smo={log_row['smooth_loss']:.4f}  "
                    f"attnH={log_row['attn_entropy_loss']:.4f}  "
                    f"valGenePCC={log_row['val_gene_pcc_mean']:.4f}  "
                    f"best={best_val_pcc:.4f}"
                )

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.name}] restored best checkpoint (val gene PCC = {best_val_pcc:.4f})")
        return self.train_log

    @torch.no_grad()
    def _eval_val(self, y_pred: torch.Tensor) -> float:
        if not self.val_mask.any():
            return float("nan")
        pred = y_pred[self.val_mask]
        target = self.t_expr[self.val_mask]
        eps = 1e-6
        pc = pred - pred.mean(dim=0, keepdim=True)
        tc = target - target.mean(dim=0, keepdim=True)
        num = (pc * tc).sum(dim=0)
        den = torch.sqrt((pc ** 2).sum(dim=0) * (tc ** 2).sum(dim=0) + eps)
        pcc = num / (den + eps)
        valid = (pred.std(dim=0) > eps) & (target.std(dim=0) > eps)
        if not valid.any():
            return float("nan")
        return float(pcc[valid].mean().item())

    # ------------------------------------------------------------------
    # cross-slice inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        query_he: np.ndarray,
        query_direct: np.ndarray,
        query_indirect: np.ndarray,
        exemplar_he: np.ndarray,
        exemplar_direct: np.ndarray,
        exemplar_indirect: np.ndarray,
        exemplar_expr: np.ndarray,
        query_spatial: Optional[np.ndarray] = None,
        exemplar_spatial: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the trained refiner on a fresh (query, exemplar) pair.

        Used at *test time* to apply RefinerA (trained on Slice 1) on
        ``query=Slice 2``, ``exemplar=Slice 1``, etc.  Build a brand-new
        heterogeneous graph for the new query/exemplar arrays and forward.
        """
        cfg = self.cfg
        device = self.device
        self.model.eval()

        query_he = np.asarray(query_he, dtype=np.float32)
        query_direct = np.asarray(query_direct, dtype=np.float32)
        query_indirect = np.asarray(query_indirect, dtype=np.float32)
        exemplar_he = np.asarray(exemplar_he, dtype=np.float32)
        exemplar_direct = np.asarray(exemplar_direct, dtype=np.float32)
        exemplar_indirect = np.asarray(exemplar_indirect, dtype=np.float32)
        exemplar_expr = np.asarray(exemplar_expr, dtype=np.float32)

        # Build retrieval features for both sides.
        q_retr = make_retrieval_features(query_he, query_direct, query_indirect,
                                         pca_dim=cfg.prediction_pca_dim,
                                         seed=cfg.seed)
        e_retr = make_retrieval_features(exemplar_he, exemplar_direct,
                                         exemplar_indirect,
                                         pca_dim=cfg.prediction_pca_dim,
                                         seed=cfg.seed + 1)

        if query_spatial is not None:
            qq_feat = np.asarray(query_spatial, dtype=np.float32)
        else:
            qq_feat = q_retr
        if exemplar_spatial is not None:
            ee_feat = np.asarray(exemplar_spatial, dtype=np.float32)
        else:
            ee_feat = e_retr

        print(f"[{self.name}] (predict) query-query KNN graph "
              f"k={cfg.k_query_graph} dim={qq_feat.shape[1]}", flush=True)
        adj_qq = build_row_normalized_adj(
            qq_feat, k=cfg.k_query_graph, metric="l2",
            exclude_self=True, add_self_loop=True, device=str(device),
        )
        print(f"[{self.name}] (predict) exemplar-exemplar KNN graph "
              f"k={cfg.k_exemplar_graph} dim={ee_feat.shape[1]}", flush=True)
        adj_ee = build_row_normalized_adj(
            ee_feat, k=cfg.k_exemplar_graph, metric="l2",
            exclude_self=True, add_self_loop=True, device=str(device),
        )
        # query-exemplar: cross-slice retrieval (no self-exclusion: rows belong
        # to different slices and may share index numbers, but they are NOT
        # the same cell).
        print(f"[{self.name}] (predict) query-exemplar KNN edges "
              f"k={cfg.k_exemplar} metric={cfg.retrieval_metric} "
              f"dim={q_retr.shape[1]}", flush=True)
        edge_qe = _build_query_exemplar_edge_with_batch(
            q_retr, e_retr, k=cfg.k_exemplar,
            metric=cfg.retrieval_metric, exclude_self=False,
            device=str(device),
            q_batch=cfg.knn_q_batch, r_batch=cfg.knn_r_batch,
        )

        t_q_he = torch.from_numpy(query_he).to(device)
        t_q_d = torch.from_numpy(query_direct).to(device)
        t_q_i = torch.from_numpy(query_indirect).to(device)
        t_e_he = torch.from_numpy(exemplar_he).to(device)
        t_e_d = torch.from_numpy(exemplar_direct).to(device)
        t_e_i = torch.from_numpy(exemplar_indirect).to(device)
        t_e_x = torch.from_numpy(exemplar_expr).to(device)

        amp_enabled = (
            self.amp_dtype is not None and str(self.device).startswith("cuda")
        )
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype)
            if amp_enabled else nullcontext()
        )
        with amp_ctx:
            y_pred, _, _ = self.model(
                t_q_he, t_q_d, t_q_i,
                t_e_he, t_e_d, t_e_i, t_e_x,
                adj_qq, adj_ee, edge_qe,
            )
        return y_pred.float().detach().cpu().numpy()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)


# =============================================================================
# convenience helper to dump train logs as csv
# =============================================================================

def save_train_log(log: List[Dict[str, float]], path: str) -> None:
    import csv
    if not log:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(log[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in log:
            w.writerow(row)
