"""
SpatialEx-pro 网络结构。

相对于上游 :class:`SpatialEx.model.Model`，本模块的关键改动：

- H&E 投影 MLP 可在两个 HGNN head 间 **共享**（由
  :attr:`SpatialExProConfig.share_projection` 控制）。共享后编码器在
  两片切片上都接收梯度信号，是把切片域差异打平的最廉价手段。
- HGNN 块与预测头之间引入残差连接，使多损失联合训练更稳定。
- 单独的 :class:`DGIHead` 提供与原 ``Predictor_dgi`` 等价的自监督对
  比目标，但与监督路径解耦，避免监督梯度被对比梯度稀释。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class HGNNLayer(nn.Module):
    """One sparse hypergraph convolution: ``y = act(H @ (W x))``.

    The hypergraph is normalised externally (HPNN), so this layer is just
    a sparse-mm followed by a non-linearity.  Equivalent to one of the
    layers inside :class:`SpatialEx.model.HGNN` but with the activation
    inside the layer for clarity.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1,
                 activation: str = "prelu") -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if activation == "prelu":
            self.act = nn.PReLU(out_dim)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        x = self.linear(self.dropout(x))
        x = torch.sparse.mm(H, x)
        return self.act(x)


class HGNNBlock(nn.Module):
    """Stack of HGNN layers used as a single slice's regression head."""

    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1, activation: str = "prelu") -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList(
            [HGNNLayer(hidden_dim, hidden_dim, dropout=dropout,
                       activation=activation)
             for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, H) + x  # residual
        return x


class PredictorHead(nn.Module):
    """``hidden -> hidden -> out_dim`` with residual + LeakyReLU output.

    The output activation ``LeakyReLU(0.1)`` matches the SpatialEx
    baseline so that predicted expression stays mostly non-negative
    (log1p(counts) is non-negative).
    """

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.norm(h + F.leaky_relu(self.fc1(h), 0.1))
        return F.leaky_relu(self.fc2(z), 0.1)


class DGIHead(nn.Module):
    """Cosine-margin DGI head identical in spirit to the baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, z: torch.Tensor, z_shuffled: torch.Tensor) -> torch.Tensor:
        c = z.mean(dim=0, keepdim=True)
        c_exp = c.expand_as(z)
        nb = z.shape[0]
        device = z.device
        lbl_pos = torch.ones(nb, device=device)
        lbl_neg = -torch.ones(nb, device=device)
        return (
            self.criterion(z, c_exp, lbl_pos)
            + self.criterion(z_shuffled, c_exp, lbl_neg)
        )


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class SpatialExProModel(nn.Module):
    """Two-headed SpatialEx_pro regressor.

    Two HGNN heads, optionally sharing their input MLP.  Each head has
    an independent predictor: ``H&E -> hidden -> HGNN -> predictor -> expr``.

    The model exposes two forward modes:

    - :meth:`forward_panelA` / :meth:`forward_panelB`: predict the panel
      using the *corresponding* HGNN head.
    - :meth:`forward_cross`: feed slice 2's H&E through the panel-A head
      (and slice 1's H&E through the panel-B head) -- this is what the
      cross-slice anchor loss uses.

    All heads see the **same** ``in_dim`` because UNI embeddings are
    fixed-dim across slices.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 313,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_projection: bool = True,
        use_dgi: bool = True,
    ) -> None:
        super().__init__()
        self.share_projection = share_projection
        self.use_dgi = use_dgi

        def _make_proj() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(hidden_dim),
            )

        if share_projection:
            self.proj_shared = _make_proj()
            self.proj_A = self.proj_B = self.proj_shared
        else:
            self.proj_A = _make_proj()
            self.proj_B = _make_proj()

        self.hgnn_A = HGNNBlock(hidden_dim, num_layers=num_layers, dropout=dropout)
        self.hgnn_B = HGNNBlock(hidden_dim, num_layers=num_layers, dropout=dropout)

        self.head_A = PredictorHead(hidden_dim, out_dim)
        self.head_B = PredictorHead(hidden_dim, out_dim)

        if use_dgi:
            self.dgi_A = DGIHead()
            self.dgi_B = DGIHead()

    # -------- supervised forward --------

    def _forward(
        self,
        proj: nn.Module,
        hgnn: HGNNBlock,
        head: PredictorHead,
        x: torch.Tensor,
        H: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = proj(x)
        h = hgnn(h0, H)
        y = head(h)
        return y, h

    def forward_panelA(self, x: torch.Tensor, H: torch.Tensor):
        """Apply the Panel-A head.  Used both for slice-1 supervised
        forward and for slice-2 cross-slice anchor forward."""
        return self._forward(self.proj_A, self.hgnn_A, self.head_A, x, H)

    def forward_panelB(self, x: torch.Tensor, H: torch.Tensor):
        """Apply the Panel-B head.  Used both for slice-2 supervised
        forward and for slice-1 cross-slice anchor forward."""
        return self._forward(self.proj_B, self.hgnn_B, self.head_B, x, H)

    # -------- DGI forward --------

    def dgi_loss(self, h: torch.Tensor, panel: str) -> torch.Tensor:
        if not self.use_dgi:
            return h.new_zeros(())
        idx = torch.randperm(h.shape[0], device=h.device)
        h_shuffled = h[idx]
        if panel == "A":
            return self.dgi_A(h, h_shuffled)
        return self.dgi_B(h, h_shuffled)

    # -------- inference --------

    @torch.no_grad()
    def predict_panelA(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            y, _ = self.forward_panelA(x, H)
        finally:
            if was_training:
                self.train()
        return y

    @torch.no_grad()
    def predict_panelB(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            y, _ = self.forward_panelB(x, H)
        finally:
            if was_training:
                self.train()
        return y
