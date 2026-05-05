"""SpatialEx+-EGRefiner: SpatialEx+ 的 Exemplar-Guided Graph Refiner 增强版。

本包在 *不修改* 原版 :mod:`SpatialEx` / :mod:`spatialex+_plus` 的前提下，
基于 EGGN (Exemplar-Guided Graph Network) 思想为 SpatialEx+ 的 panel
diagonal integration 任务构建一个轻量 refiner，并配套提供：

* 多种子 baseline 搜索（:mod:`gpt.seed_search`）
* per-gene final 决策的后处理（:mod:`gpt.postprocess`）
* 柱状图与全切片可视化（:mod:`gpt.plot_bars` / :mod:`gpt.plot_full_slice`）

模块布局
--------

    refiner.py        EGRefiner 模型 + EGRefinerTrainer + EGRefinerConfig
    pipeline.py       端到端流水线脚本，同时是被其它子模块复用的工具层
    seed_search.py    多种子搜索最优 SpatialEx+ baseline
    postprocess.py    复用缓存预测做 Ridge / stacking / per-gene 决策
    plot_bars.py      读 postprocess 落盘的 CSV 画柱状图
    plot_full_slice.py 重训 SpatialEx+ 后推理 spot 外细胞，画全切片图
"""

from .refiner import (
    EGRefiner,
    EGRefinerConfig,
    EGRefinerTrainer,
    save_train_log,
)

__all__ = [
    "EGRefiner",
    "EGRefinerConfig",
    "EGRefinerTrainer",
    "save_train_log",
]
