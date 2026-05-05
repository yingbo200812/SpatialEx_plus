"""
SpatialEx-pro：基于 SpatialEx 的跨片 H&E → 单细胞表达翻译算法（增强版）。

包对外暴露：

- :class:`SpatialExPro`        —— 主训练器（端到端训练 + 跨片推理）。
- :class:`SpatialExProConfig`  —— 全部超参的 dataclass。

核心改进概览（详见 :file:`README.md`）：

1. 共享 H&E 投影 MLP，让编码器看到两片图像，得到切片不变的潜空间。
2. 跨片 H&E-NN 软伪监督（leave-one-out safe），把"训练片的 GT 沿 H&E
   邻居外推到对侧片"作为额外监督。
3. 训练目标对齐三大评估指标：MSE + Pearson + CMD-align（gene-gene 相
   关阵对齐）+ inv-std + marker boost。
4. 空间 TV 正则提升 SSIM。
5. 测试时 anchor smoothing：切片内 H&E-NN 平滑 + 跨片 anchor 凸混合。
6. Cosine LR 衰减 + 多种子集成。

CLI 入口与服务器一键脚本位于本包目录下：

- :file:`run_train_eval.py`  —— Python CLI（最完整）
- :file:`SpatialEx-pro.sh`   —— 服务器一键运行（推荐）
- :file:`make_plots.py`      —— 生成与论文 baseline 的对比图
"""

from .SpatialEx_pro import SpatialExPro
from .utils import SpatialExProConfig

__all__ = ["SpatialExPro", "SpatialExProConfig"]
