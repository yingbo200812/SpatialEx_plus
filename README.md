# Baseline

本目录整理了 SpatialEx 相关实验的 baseline 代码、改进模型与复现实验结果，主要用于对比原版 SpatialEx / SpatialEx+ 与本项目改进方法在 HBC 空间多组学任务上的表现。

## 目录结构

```text
baseline/
├── SpatialEx/                  # 原版 SpatialEx / SpatialEx+ 核心代码
├── run_SpatialEx/              # 原版 SpatialEx 与 SpatialEx+ 的运行 notebook
├── SpatialEx_pro/              # SpatialEx-pro：跨切片 H&E -> 表达预测改进版
└── SpatialEx+-EGRefiner/       # SpatialEx+-EGRefiner：SpatialEx+ 后处理增强版
```

## 主要内容

### 1. SpatialEx

`SpatialEx/` 保存原始 SpatialEx 方法的核心实现，包括模型、预处理和工具函数。该部分主要作为 baseline，不做额外算法修改。

### 2. run_SpatialEx

`run_SpatialEx/` 中包含用于复现原版 SpatialEx / SpatialEx+ 的 notebook：

- `run_SpatialEx.ipynb`
- `run_SpatialEx+.ipynb`

这些 notebook 用于生成原始 baseline 结果，并为后续改进方法提供公平对照。

### 3. SpatialEx-pro

`SpatialEx_pro/` 是在 SpatialEx 基础上改进的跨切片预测方法，目标是复现并提升 SpatialEx 论文 Fig. 2c/d 中的 leave-one-slice-out 任务表现。

主要改进包括：

- 共享 H&E 投影编码器，减少跨切片特征域偏移；
- 引入跨切片 H&E 最近邻软伪监督；
- 使用 Pearson loss、CMD alignment loss 和空间平滑约束，使训练目标更接近评估指标；
- 对 marker gene 进行加权优化；
- 支持多随机种子集成与推理后处理。

运行入口：

```bash
bash SpatialEx_pro/SpatialEx-pro.sh
```

详细说明见：

```text
SpatialEx_pro/README.md
```

### 4. SpatialEx+-EGRefiner

`SpatialEx+-EGRefiner/` 是针对 SpatialEx+ panel diagonal integration 任务的增强版本。该方法不修改原版 SpatialEx+ 主体，而是在其预测结果上加入 exemplar-guided refiner 与后处理策略。

主要功能包括：

- 基于 H&E 特征在另一切片中检索形态相似细胞；
- 使用真实测量 panel 作为 exemplar 信息辅助预测缺失 panel；
- 对 raw / direct / ridge / stack 等预测结果进行 per-gene 决策；
- 生成全局指标、目标基因 PCC 和空间可视化结果。

详细说明见：

```text
SpatialEx+-EGRefiner/README.md
```

## 结果文件

本目录中已经包含部分复现实验结果，例如：

```text
SpatialEx_pro/results/
├── metrics_spatialex_pro.csv
├── metrics_baseline.csv
├── compare.csv
├── per_gene_pcc.csv
└── config.json

SpatialEx+-EGRefiner/results/
├── final_metrics_vs_paper.csv
├── final_selected_gene_pcc.csv
├── decisions_panelA.csv
├── decisions_panelB.csv
└── config.json
```

这些文件记录了模型在 HBC 数据集上的主要评估指标，包括 PCC、SSIM、CMD 以及部分 marker gene 的预测相关性。

## 环境依赖

建议使用与 SpatialEx 原项目一致的 Python 环境。主要依赖包括：

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- Scanpy
- scikit-learn
- Matplotlib

如果在服务器上运行，请根据实际数据路径修改脚本中的 `h5ad`、raw data 和输出目录路径。

## 使用说明

如果只想查看已经复现出的结果，可以直接读取 `results/` 下的 CSV 文件。

如果需要重新运行实验，请先确认：

1. HBC 数据集路径已经正确配置；
2. GPU / CUDA 环境可用；
3. 脚本中的输入输出路径已根据本地或服务器环境修改；
4. 当前工作目录位于仓库根目录或脚本要求的位置。

推荐先阅读各子目录 README，再运行对应脚本。

## 说明

本目录用于保存 baseline 复现、方法改进和结果对比代码。原版 SpatialEx 代码主要作为对照基线保留，改进方法集中在 `SpatialEx_pro/` 和 `SpatialEx+-EGRefiner/` 中。
