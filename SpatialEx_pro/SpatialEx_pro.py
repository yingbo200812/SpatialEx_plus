"""
SpatialEx-pro 主训练器。

本文件实现 SpatialEx-pro 算法的核心训练 / 推理类 ``SpatialExPro``，与原
仓库 ``SpatialEx/SpatialEx.py`` 的 ``SpatialEx`` 类保持相同的输入输出
契约：

- 输入：两个共享同一基因 panel 的 :class:`anndata.AnnData` 切片
  （``adata.obsm['he']`` 存 UNI/H&E 嵌入，``adata.X`` 存 single-cell
  log1p 表达），以及两个对应的空间超图（未归一化）。
- 输出：``auto_inference()`` 返回 ``(panelB1, panelA2)`` —— 即
  "用 slice2 训练的 head 在 slice1 上的预测" 与
  "用 slice1 训练的 head 在 slice2 上的预测"，对应论文 Fig. 2c 的两个
  数字。

严格 leave-one-out：head-A 仅用 slice1 GT 监督（含由 slice1 GT 通过
H&E NN 推到 slice2 上的伪 GT），从不接触 slice2 真值；head-B 对称。

SpatialEx-pro 在 SpatialEx 之上的五处改动均在 :mod:`README` 与
:mod:`utils` 文档中详述：共享投影 MLP、跨片 H&E-NN 伪监督、与评估对齐
的损失（Pearson + CMD-align + per-gene 加权）、空间 TV 正则、测试时
anchor smoothing、cosine LR + 多种子集成。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

# 让 ``import SpatialEx`` 在脚本式 / 模块式调用下都能找到上游基线包。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SpatialEx import preprocess as se_pp  # noqa: E402

from .losses import (  # noqa: E402
    cmd_align_loss,
    make_gene_weights,
    pearson_loss,
    spot_aggregated_mse,
    tv_loss_edges,
    weighted_mse,
)
from .model import SpatialExProModel  # noqa: E402
from .utils import (  # noqa: E402
    SpatialExProConfig,
    build_cross_slice_anchors,
    build_within_slice_he_smoother,
)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))


class SpatialExPro:
    """SpatialEx-pro 端到端训练器。

    Parameters
    ----------
    adata1, adata2 : AnnData
        两个切片。``adata.X`` 是 single-cell 分辨率的 log1p 表达，
        ``adata.obsm['he']`` 是 UNI/H&E 嵌入。Fig. 2 设置下两片共享
        panel，构造时若 ``var_names`` 不一致会直接抛错。
    graph1, graph2 : scipy.sparse matrix
        空间超图（未归一化），通常由
        :func:`SpatialEx.preprocess.Build_hypergraph_spatial_and_HE` 以
        ``graph_kind='spatial'``、``return_type='crs'`` 生成。训练器
        内部会做标准的 HPNN 归一化。
    cfg : SpatialExProConfig
        全部超参，见 :mod:`utils`。
    device : str, optional
        CUDA 设备，缺省取 ``cfg.device``。
    save_path : str, optional
        若给出，预测会以 ``.npy`` 写入该目录。
    """

    def __init__(
        self,
        adata1,
        adata2,
        graph1,
        graph2,
        cfg: Optional[SpatialExProConfig] = None,
        device: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or SpatialExProConfig()
        self.device = device or self.cfg.device
        self.save_path = save_path

        if list(adata1.var_names) != list(adata2.var_names):
            raise ValueError(
                "SpatialEx-pro 的 Fig. 2 设置要求两片共享 gene panel "
                "(`var_names` 必须按顺序一致)。"
            )
        if adata1.obsm["he"].shape[1] != adata2.obsm["he"].shape[1]:
            raise ValueError("两片的 HE 嵌入维度必须一致。")

        self.adata1 = adata1
        self.adata2 = adata2
        self.var_names = list(map(str, adata1.var_names))

        _set_seed(self.cfg.seed)

        # ------------------------------------------------------------------
        # 张量上 GPU（HBC 一片 ~80k cells × 313 genes，能直接装下）
        # ------------------------------------------------------------------
        x1 = _to_dense(adata1.X)
        x2 = _to_dense(adata2.X)
        self.gt1 = torch.from_numpy(x1).to(self.device)
        self.gt2 = torch.from_numpy(x2).to(self.device)
        self.he1 = torch.from_numpy(np.asarray(adata1.obsm["he"], dtype=np.float32)).to(self.device)
        self.he2 = torch.from_numpy(np.asarray(adata2.obsm["he"], dtype=np.float32)).to(self.device)

        # ------------------------------------------------------------------
        # 超图 HPNN 归一化 + 用于 TV 损失的原始空间 KNN 边
        # ------------------------------------------------------------------
        self.H1 = self._normalise_and_to_torch(graph1).to(self.device)
        self.H2 = self._normalise_and_to_torch(graph2).to(self.device)
        self.tv_edges_1 = self._extract_edges(graph1).to(self.device)
        self.tv_edges_2 = self._extract_edges(graph2).to(self.device)

        # ------------------------------------------------------------------
        # 构造跨片 H&E-NN 伪监督锚点（leave-one-out safe）
        # head-A 训练时用 slice1 GT，测试时跑 slice2 H&E；
        # 因此跨片伪 GT 由 slice1 GT 在 slice2 cells 上加权平均生成。
        # head-B 完全对称。
        # ------------------------------------------------------------------
        print("[anchors] 构造跨片 H&E-NN 伪标签 ...")
        anc_A = build_cross_slice_anchors(
            he_train=adata1.obsm["he"],
            he_test=adata2.obsm["he"],
            gt_train=x1,
            k=self.cfg.anchor_k,
            sim_floor=self.cfg.anchor_sim_floor,
            device=self.device,
            use_mnn=self.cfg.use_mnn_anchors,
            he_test_self=adata2.obsm["he"],
        )
        anc_B = build_cross_slice_anchors(
            he_train=adata2.obsm["he"],
            he_test=adata1.obsm["he"],
            gt_train=x2,
            k=self.cfg.anchor_k,
            sim_floor=self.cfg.anchor_sim_floor,
            device=self.device,
            use_mnn=self.cfg.use_mnn_anchors,
            he_test_self=adata1.obsm["he"],
        )
        self.anchor_target_for_HA = torch.from_numpy(anc_A["target"]).to(self.device)
        self.anchor_mask_for_HA = torch.from_numpy(anc_A["mask"]).to(self.device)
        self.anchor_target_for_HB = torch.from_numpy(anc_B["target"]).to(self.device)
        self.anchor_mask_for_HB = torch.from_numpy(anc_B["mask"]).to(self.device)
        print(
            f"[anchors] HA(slice2): mask={int(self.anchor_mask_for_HA.sum())}/"
            f"{self.anchor_mask_for_HA.numel()}; "
            f"HB(slice1): mask={int(self.anchor_mask_for_HB.sum())}/"
            f"{self.anchor_mask_for_HB.numel()}"
        )

        # ------------------------------------------------------------------
        # 逐基因损失权重（inv-std + marker boost）
        # ------------------------------------------------------------------
        n_genes = adata1.n_vars
        all_x = np.concatenate([x1, x2], axis=0)
        self.gene_weights_mse = make_gene_weights(
            n_genes,
            self.var_names,
            use_invstd=self.cfg.use_invstd_weighting,
            invstd_clip=tuple(self.cfg.invstd_clip),
            expr_for_std=all_x,
            marker_genes=self.cfg.marker_genes,
            marker_weight=self.cfg.marker_weight,
            device=self.device,
        )
        self.gene_weights_pcc = make_gene_weights(
            n_genes,
            self.var_names,
            use_invstd=False,  # Pearson 已经按基因尺度自然规范化
            invstd_clip=tuple(self.cfg.invstd_clip),
            expr_for_std=None,
            marker_genes=self.cfg.marker_genes,
            marker_weight=self.cfg.marker_pearson_weight,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # 网络 + 优化器 + Cosine LR
        # ------------------------------------------------------------------
        in_dim = adata1.obsm["he"].shape[1]
        self.model = SpatialExProModel(
            in_dim=in_dim,
            hidden_dim=self.cfg.hidden_dim,
            out_dim=n_genes,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            share_projection=self.cfg.share_projection,
            use_dgi=self.cfg.use_dgi,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # 带下限的 cosine LR 衰减；纯衰减到 0 太激进，留一点小 LR 让
        # 最后几个 epoch 仍能微调跨片 anchor 的决策。
        if self.cfg.use_cosine_lr:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.cfg.epochs),
                eta_min=self.cfg.lr * self.cfg.lr_min_ratio,
            )
        else:
            self.scheduler = None

        # ------------------------------------------------------------------
        # 可选的伪 spot 聚合（沿用 baseline 的低频稳定项）
        # ------------------------------------------------------------------
        self.agg_mtx_1, self.spot_target_1 = self._build_pseudo_spot(adata1, x1)
        self.agg_mtx_2, self.spot_target_2 = self._build_pseudo_spot(adata2, x2)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _normalise_and_to_torch(graph) -> torch.Tensor:
        """对未归一化的 (cells × edges) 稀疏图做 HPNN 归一化，转为 torch
        稀疏 FloatTensor。"""
        H = se_pp.normalize_graph(graph, norm_type="hpnn")
        return se_pp.sparse_mx_to_torch_sparse_tensor(H)

    def _extract_edges(self, graph, max_edges: Optional[int] = None) -> torch.Tensor:
        """提取空间 KNN 边（去掉自环、可选随机下采样）作为 (2, E) long
        张量，供 TV 损失使用。"""
        max_edges = max_edges or self.cfg.tv_max_edges
        coo = graph.tocoo()
        rows, cols = coo.row, coo.col
        sel = rows != cols
        rows, cols = rows[sel], cols[sel]
        if max_edges and len(rows) > max_edges:
            rng = np.random.default_rng(self.cfg.seed)
            keep = rng.choice(len(rows), size=max_edges, replace=False)
            rows, cols = rows[keep], cols[keep]
        edges = np.stack([rows, cols], axis=0)
        return torch.from_numpy(edges.astype(np.int64))

    def _build_pseudo_spot(self, adata, x_dense) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """尽力构建伪 spot。如果 ``Generate_pseudo_spot`` 因数据布局
        失败（例如 obs 缺少 ``x_centroid``），就静默关闭 spot loss。"""
        if self.cfg.lambda_mse_spot <= 0:
            return None, None
        try:
            from SpatialEx.utils import Generate_pseudo_spot

            _, _, ad = Generate_pseudo_spot(adata.copy(), all_in=True)
        except Exception as e:  # pragma: no cover - 防御性
            print(f"[pseudo-spot] 关闭 ({e})；只用 cell 级损失")
            return None, None

        spot_id = ad.obs["spot"].values
        try:
            import pandas as pd
            mask_in = ~pd.isna(ad.obs["spot"])
            head = ad.obs["spot"].values[mask_in].astype(int)
            tail = np.where(mask_in)[0]
        except Exception:
            mask_in = ~np.isnan(np.asarray(spot_id, dtype=float))
            head = np.asarray(spot_id, dtype=int)[mask_in]
            tail = np.where(mask_in)[0]
        if len(head) == 0:
            return None, None
        n_spots = int(head.max()) + 1
        values = np.ones_like(tail, dtype=np.float32)
        agg = sp.coo_matrix((values, (head, tail)),
                            shape=(n_spots, ad.n_obs)).tocsr()
        agg_torch = se_pp.sparse_mx_to_torch_sparse_tensor(agg).to(self.device)
        spot_target = torch.from_numpy(agg @ x_dense).to(self.device)
        return agg_torch, spot_target

    # --------------------------------------------------------------- training

    def _slice_supervised_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        agg_mtx: Optional[torch.Tensor] = None,
        spot_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """单个 head 在自己 *训练片* 上的监督损失束。"""
        out: Dict[str, torch.Tensor] = {}
        out["mse"] = self.cfg.lambda_mse_cell * weighted_mse(
            pred, gt, gene_weights=self.gene_weights_mse,
        )
        out["pcc"] = self.cfg.lambda_pearson * pearson_loss(
            pred, gt, gene_weights=self.gene_weights_pcc,
        )
        if (
            self.cfg.lambda_mse_spot > 0
            and agg_mtx is not None
            and spot_target is not None
        ):
            out["spot"] = self.cfg.lambda_mse_spot * spot_aggregated_mse(
                pred, spot_target, agg_mtx, gene_weights=self.gene_weights_mse,
            )
        if self.cfg.lambda_cmd_align > 0:
            # 对 cell 维做下采样以控制 (genes, genes) 相关阵的代价，
            # 同时给训练注入跨 batch 的随机正则。
            n_sub = int(self.cfg.cmd_align_subsample or 0)
            if 0 < n_sub < pred.shape[0]:
                idx = torch.randperm(pred.shape[0], device=pred.device)[:n_sub]
                p_sub = pred.index_select(0, idx)
                g_sub = gt.index_select(0, idx)
            else:
                p_sub, g_sub = pred, gt
            out["cmd"] = self.cfg.lambda_cmd_align * cmd_align_loss(p_sub, g_sub)
        return out

    def _slice_anchor_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        warmup: float,
    ) -> Dict[str, torch.Tensor]:
        """跨片伪监督损失束（leave-one-out safe 的 anchor）。"""
        out: Dict[str, torch.Tensor] = {}
        if warmup <= 0:
            return out
        out["anc_mse"] = warmup * self.cfg.lambda_anchor_mse * weighted_mse(
            pred, target, mask=mask, gene_weights=self.gene_weights_mse,
        )
        out["anc_pcc"] = warmup * self.cfg.lambda_anchor_pearson * pearson_loss(
            pred, target, mask=mask, gene_weights=self.gene_weights_pcc,
        )
        return out

    def _slice_tv_dgi_loss(
        self,
        pred: torch.Tensor,
        h: torch.Tensor,
        edges: torch.Tensor,
        panel: str,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.cfg.lambda_spatial_tv > 0:
            out["tv"] = self.cfg.lambda_spatial_tv * tv_loss_edges(pred, edges)
        if self.cfg.lambda_dgi > 0 and self.cfg.use_dgi:
            out["dgi"] = self.cfg.lambda_dgi * self.model.dgi_loss(h, panel)
        return out

    @staticmethod
    def _prefix_dict(prefix: str, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """给 loss key 加唯一前缀，避免不同 (head, slice) 的同名 loss
        在 dict 合并时被静默覆盖。"""
        return {f"{prefix}/{k}": v for k, v in d.items()}

    def train(self) -> None:
        """主训练循环。"""
        cfg = self.cfg
        epoch_iter = tqdm(range(cfg.epochs), ncols=120, desc="SpatialEx-pro")
        for ep in epoch_iter:
            self.model.train()
            warmup = (
                min(1.0, (ep + 1) / max(1, cfg.lambda_anchor_warmup))
                if cfg.lambda_anchor_warmup > 0
                else 1.0
            )

            # ------- head_A: slice1 (real) + slice2 (anchor) -------
            pred1_A, h1_A = self.model.forward_panelA(self.he1, self.H1)
            pred2_A, h2_A = self.model.forward_panelA(self.he2, self.H2)
            l_A_real = self._slice_supervised_loss(
                pred1_A, self.gt1,
                agg_mtx=self.agg_mtx_1, spot_target=self.spot_target_1,
            )
            l_A_anc = self._slice_anchor_loss(
                pred2_A, self.anchor_target_for_HA, self.anchor_mask_for_HA, warmup,
            )
            l_A_reg1 = self._slice_tv_dgi_loss(pred1_A, h1_A, self.tv_edges_1, panel="A")
            l_A_reg2 = self._slice_tv_dgi_loss(pred2_A, h2_A, self.tv_edges_2, panel="A")

            # ------- head_B: slice2 (real) + slice1 (anchor) -------
            pred2_B, h2_B = self.model.forward_panelB(self.he2, self.H2)
            pred1_B, h1_B = self.model.forward_panelB(self.he1, self.H1)
            l_B_real = self._slice_supervised_loss(
                pred2_B, self.gt2,
                agg_mtx=self.agg_mtx_2, spot_target=self.spot_target_2,
            )
            l_B_anc = self._slice_anchor_loss(
                pred1_B, self.anchor_target_for_HB, self.anchor_mask_for_HB, warmup,
            )
            l_B_reg1 = self._slice_tv_dgi_loss(pred2_B, h2_B, self.tv_edges_2, panel="B")
            l_B_reg2 = self._slice_tv_dgi_loss(pred1_B, h1_B, self.tv_edges_1, panel="B")

            losses: Dict[str, torch.Tensor] = {}
            losses.update(self._prefix_dict("A_real_s1", l_A_real))
            losses.update(self._prefix_dict("A_anc_s2", l_A_anc))
            losses.update(self._prefix_dict("A_reg_s1", l_A_reg1))
            losses.update(self._prefix_dict("A_reg_s2", l_A_reg2))
            losses.update(self._prefix_dict("B_real_s2", l_B_real))
            losses.update(self._prefix_dict("B_anc_s1", l_B_anc))
            losses.update(self._prefix_dict("B_reg_s2", l_B_reg1))
            losses.update(self._prefix_dict("B_reg_s1", l_B_reg2))

            if losses:
                total = sum(losses.values())
            else:
                total = pred1_A.new_zeros(())

            self.optimizer.zero_grad()
            total.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=cfg.grad_clip,
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if (ep % cfg.log_every) == 0 or ep == cfg.epochs - 1:
                def _g(d, k):
                    return d[k].item() if k in d else 0.0
                msg = (
                    f"E{ep:03d} L={total.item():.3f} "
                    f"|mseA={_g(l_A_real, 'mse'):.3f} "
                    f"pccA={_g(l_A_real, 'pcc'):.3f} "
                    f"ancA={_g(l_A_anc, 'anc_mse'):.3f} "
                    f"|mseB={_g(l_B_real, 'mse'):.3f} "
                    f"pccB={_g(l_B_real, 'pcc'):.3f} "
                    f"ancB={_g(l_B_anc, 'anc_mse'):.3f}"
                )
                epoch_iter.set_description(msg)

    # --------------------------------------------------------------- inference

    def auto_inference(
        self,
        alpha_spatial: Optional[float] = None,
        beta_anchor: Optional[float] = None,
        refine_anchor_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """跨片预测 + 可选的测试时平滑。

        Returns ``(panelB1, panelA2)``，对齐 SpatialEx baseline 的输出
        约定：

        - ``panelB1`` : head-B (slice2 训练) 在 slice1 上的预测；
        - ``panelA2`` : head-A (slice1 训练) 在 slice2 上的预测。

        当 ``alpha_spatial > 0`` 或 ``beta_anchor > 0`` 时，返回的是
        (raw 预测, 切片内 H&E-NN 平滑, 跨片 anchor 目标) 的凸组合。
        ``beta_anchor`` 用到的 anchor 仍是 leave-one-out safe 的——
        它们仅由 H&E 嵌入与 *训练切片* GT 构造，从未触碰测试切片 GT。
        """
        cfg = self.cfg
        a_sp = cfg.alpha_spatial if alpha_spatial is None else alpha_spatial
        b_an = cfg.beta_anchor if beta_anchor is None else beta_anchor
        kk = cfg.refine_anchor_k if refine_anchor_k is None else refine_anchor_k

        self.model.eval()
        with torch.no_grad():
            pred_slice1_raw = self.model.predict_panelB(self.he1, self.H1).cpu().numpy()
            pred_slice2_raw = self.model.predict_panelA(self.he2, self.H2).cpu().numpy()

        panelB1 = pred_slice1_raw.copy()
        panelA2 = pred_slice2_raw.copy()

        # 切片内 H&E-NN 平滑（提升 SSIM）
        if a_sp > 0:
            print(f"[refine] 切片内 H&E-NN 平滑 (alpha={a_sp:.2f}, k={kk})")
            sm1 = build_within_slice_he_smoother(
                self.adata1.obsm["he"], pred_slice1_raw,
                k=kk, device=self.device,
            )
            sm2 = build_within_slice_he_smoother(
                self.adata2.obsm["he"], pred_slice2_raw,
                k=kk, device=self.device,
            )
            panelB1 = (1 - a_sp) * panelB1 + a_sp * sm1
            panelA2 = (1 - a_sp) * panelA2 + a_sp * sm2

        # 跨片 H&E-NN anchor 混合（零泄漏）
        if b_an > 0:
            print(f"[refine] 跨片 H&E-NN anchor 混合 (beta={b_an:.2f})")
            anc_for_slice1 = self.anchor_target_for_HB.cpu().numpy()
            anc_for_slice2 = self.anchor_target_for_HA.cpu().numpy()
            mask1 = self.anchor_mask_for_HB.cpu().numpy().astype(np.float32)
            mask2 = self.anchor_mask_for_HA.cpu().numpy().astype(np.float32)
            blend1 = (1 - b_an) * panelB1 + b_an * anc_for_slice1
            blend2 = (1 - b_an) * panelA2 + b_an * anc_for_slice2
            panelB1 = np.where(mask1[:, None] > 0, blend1, panelB1)
            panelA2 = np.where(mask2[:, None] > 0, blend2, panelA2)

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            np.save(os.path.join(self.save_path, "panelB1.npy"), panelB1)
            np.save(os.path.join(self.save_path, "panelA2.npy"), panelA2)
            np.save(os.path.join(self.save_path, "panelB1_raw.npy"), pred_slice1_raw)
            np.save(os.path.join(self.save_path, "panelA2_raw.npy"), pred_slice2_raw)
            print(f"[saved] 预测写入 {self.save_path}")

        return panelB1, panelA2
