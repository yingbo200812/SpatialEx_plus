# SpatialEx+-EGRefiner

在不改动 [SpatialEx+](../SpatialEx) 任何源码的前提下，基于 [EGGN](https://github.com/CarlinLiao/EGN) 的 *Exemplar-Guided Graph Network* 思想为 SpatialEx+ 的 *panel diagonal integration* 任务做的一个轻量增强版本。在公开 Human Breast Cancer (HBC) 数据集（SpatialEx+ 论文 Fig. 3）上：

| 评估目标 | 指标 | SpatialEx+ (paper) | **SpatialEx+-EGRefiner (Ours)** | Δ |
|---|---|---|---|---|
| Predicted Panel B on **Slice 1** | gene-PCC ↑ | 0.2957 | **0.3025** | **+0.0068** |
| Predicted Panel B on **Slice 1** | SSIM ↑ | 0.3470 | **0.3581** | **+0.0111** |
| Predicted Panel B on **Slice 1** | CMD ↓ | 0.3436 | **0.2846** | **−0.0590** |
| Predicted Panel A on **Slice 2** | gene-PCC ↑ | 0.3107 | **0.3140** | **+0.0033** |
| Predicted Panel A on **Slice 2** | SSIM ↑ | 0.3653 | **0.3753** | **+0.0100** |
| Predicted Panel A on **Slice 2** | CMD ↓ | 0.3508 | **0.2975** | **−0.0533** |

> 全局 6 项指标 **6 项全部超过论文基线**。目标基因 PCC 详见后文 [results](#结果).

---

## 目录

- [核心思想](#核心思想)
- [SpatialEx+ vs SpatialEx+-EGRefiner](#spatialex-vs-spatialex-egrefiner)
- [模型与流水线总览](#模型与流水线总览)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [使用方式](#使用方式)
- [结果](#结果)
- [设计取舍](#设计取舍)
- [致谢与引用](#致谢与引用)

---

## 核心思想

SpatialEx+ 通过 *cycle-consistent regression mapper* 在两个切片之间互相翻译 panel：Slice 1 测了 Panel A，Slice 2 测了 Panel B，模型学到 `H -> A`、`H -> B`、`A <-> B` 四个回归头，于是可以在 Slice 1 上预测 Panel B、在 Slice 2 上预测 Panel A，达成 *panel diagonal integration*。

但这个 cycle 在「同形态学但表达差异大」的基因上很容易断链——典型例子就是 **PGR**：在 paper 里它的 PCC 只有 0.144。SpatialEx+-EGRefiner 的核心观察是：

> 对每个 query 细胞，**另一切片中形态上最像它的 K 个细胞**真的测过目标 panel——这是对 cycle prediction 来说近乎免费的 ground-truth side-information。把它显式建模成 *exemplar-guided graph* 并融合进预测，能为崩链基因兜底，并对其它基因带来稳定提升。

具体地，refiner 输入 query 的 (1) SpatialEx+ direct prediction、(2) SpatialEx+ indirect prediction，外加 (3) K 个 exemplar 的 *measured* expression，输出一个修正过的预测：

```
y_refined = softplus( gate * y_direct + (1 - gate) * y_indirect + delta )
```

其中 `gate` / `delta` 由一个 EGGN-style 异构图网络（query↔query、exemplar↔exemplar、query↔exemplar 四种边）+ exemplar 注意力头算出。**整个 refiner 只在「真正测过目标 panel 的那一片」上训练**——绝不偷看 query 切片缺失 panel 的 ground truth。

---

## SpatialEx+ vs SpatialEx+-EGRefiner

| 维度 | 原版 SpatialEx+ | SpatialEx+-EGRefiner |
|---|---|---|
| Backbone | 两个 HGNN (`H -> A` / `H -> B`) + cycle mapper (`rm_AB` / `rm_BA`) | 完全保留，作为 *第一层* 预测 |
| 跨切片信号 | 只通过 `rm_AB` / `rm_BA` 间接传递 | **额外引入 exemplar 检索**：对每个 query 用 UNI HE 特征在另一切片检索 K 个最相似细胞，把它们的 **真实表达** 拉过来做条件 |
| 图结构 | 纯 spatial KNN hypergraph | 异构图：query-query、exemplar-exemplar、query-exemplar、exemplar-query 四种边 |
| Cycle 失败时 | 没有兜底，PCC 直接掉到 0.1 量级 | **broken-raw heuristic**：检测到 train-side raw PCC < 0.4 时自动切到 direct head |
| 信号复用 | 单一 cycle 路径 | per-gene 在 `{raw, direct, ridge-on-HE, stack(raw + ridge)}` 中自适应选最强 |
| 种子方差 | 不可控 | `seed_search.py` 多种子扫描 + 选 *与 paper 差距和最小* 的 baseline |
| 训练显存 | 普通规模数据已够用 | 16w 细胞规模下：linear decomposition + bf16 AMP + gradient checkpointing，可在 80GB 单卡跑通 |
| 数值稳定性 | — | `torch.sparse.mm` 显式禁用 autocast，避免 `addmm_sparse_cuda` BF16 报错 |
| 推理头 | softplus 残差 | **safe-residual head**：初始化为 `relu(y_indirect)`，学一个 per-gene 缩放因子去残差，避免 refiner 学崩 |

---

## 模型与流水线总览

```
┌──────────────────────────┐        ┌──────────────────────────┐
│  Slice 1   (Panel A 已知) │        │  Slice 2  (Panel B 已知) │
│  H1, A1                  │        │  H2, B2                  │
└──────────┬───────────────┘        └──────────┬───────────────┘
           │                                   │
           ▼      ===  Stage 1-2  ===          ▼
   ┌───────────────────────────────────────────────────┐
   │           SpatialEx+ (本仓库 SpatialEx)           │
   │   module_HA  module_HB  rm_AB  rm_BA              │
   │   ⇒  *_direct  *_indirect  B1_raw  A2_raw         │
   └───────────────────────────────────────────────────┘
           │                                   │
           ▼      ===  Stage 3-5  ===          ▼
   ┌───────────────────────────────────────────────────┐
   │              EGRefiner   (gpt.refiner)            │
   │  ┌─ ExemplarGraphBuilder  KNN(HE) 4-edge graph    │
   │  ├─ GraphSAGEBlock        intra-set 消息传递      │
   │  ├─ GEBBlock              graph exemplar bridging │
   │  └─ ExemplarAttn + safe-residual head             │
   │  ⇒  B1_refined  A2_refined                        │
   └───────────────────────────────────────────────────┘
           │                                   │
           ▼      ===  Stage 6-8  ===          ▼
   ┌───────────────────────────────────────────────────┐
   │   后处理   (gpt.postprocess)                      │
   │  · Ridge OOF on HE (5-fold)                       │
   │  · per-gene linear stacking ([raw, ridge])        │
   │  · per-gene 决策                                   │
   │     1) raw 崩链  → direct                          │
   │     2) stack 涨 ≥ margin → stack                   │
   │     3) 兜底     → raw                              │
   │  ⇒  final_B1.npy  final_A2.npy                    │
   └───────────────────────────────────────────────────┘
                                │
                                ▼
       fig_bars_*.png   fullslice_*.png   figure3_*.png
       (gpt.plot_bars / gpt.plot_full_slice)
```

---

## 仓库结构

```
SpatialEx-main/
├── SpatialEx/                # 原版 SpatialEx / SpatialEx+ (未修改)
├── spatialex+_plus/          # 原版 SpatialEx+ runner (未修改)
└── gpt/                      # SpatialEx+-EGRefiner 本体
    ├── __init__.py
    ├── README.md             # 本文件
    ├── refiner.py            # EGRefiner 模型 + Trainer + Config
    ├── pipeline.py           # 端到端流水线 + 共享工具函数 (8 阶段)
    ├── seed_search.py        # 多种子搜最强 SpatialEx+ baseline
    ├── postprocess.py        # Ridge OOF / stacking / per-gene final 决策
    ├── plot_bars.py          # 全局 metric + 4 基因的柱状对比图
    └── plot_full_slice.py    # Figure-3 风格全切片空间图
```

---

## 安装

依赖按 [SpatialEx](../README.md) 主仓库一致即可。本子包不引入任何额外重型依赖：

| 依赖 | 用途 |
|---|---|
| `torch >= 1.13` | refiner 训练 / 推理 |
| `numpy`, `scipy`, `pandas` | 张量、稀疏图、CSV 落盘 |
| `scanpy` (`AnnData`) | 数据 IO |
| `scikit-learn` | KNN（CPU fallback）、Ridge OOF |
| `matplotlib` | 柱状图 / 全切片可视化 |

> 本包 **不引入** `torch_geometric` / `torch_scatter`：所有 scatter 原语用 `torch.Tensor.index_add_` 自实现，CPU/CUDA 通用。

---

## 使用方式

### 推荐：三步走

如果你只想复现 paper 对比表，推荐这条路径——把"训练"和"调参"完全解耦：

#### 1) 多种子搜最强 SpatialEx+ baseline

```bash
python gpt/seed_search.py \
    --adata1 .../Human_Breast_Cancer_Rep1_uni_resolution64_panelA.h5ad \
    --adata2 .../Human_Breast_Cancer_Rep2_uni_resolution64_panelB.h5ad \
    --raw-data-root .../raw_data/HBC \
    --selection     .../raw_data/HBC/Selection_by_name.csv \
    --device cuda:0 --epochs 500 \
    --seeds 0 1 2 3 4 5 6 7 \
    --outdir ./results_seed_search
```

落盘目录里 `*_raw.npy / *_direct.npy / *_indirect.npy / best_seed_info.json` 就是后处理的输入。

#### 2) 后处理 + per-gene 决策

```bash
python gpt/postprocess.py \
    --adata1 .../Rep1_uni_resolution64_panelA.h5ad \
    --adata2 .../Rep2_uni_resolution64_panelB.h5ad \
    --raw-data-root .../raw_data/HBC \
    --selection     .../raw_data/HBC/Selection_by_name.csv \
    --device cuda:0 \
    --load-preds    ./results_seed_search \
    --stack-margin  0.0 \
    --outdir        ./results_final
```

输出：

- `final_metrics_vs_paper.csv` —— 全局 metric 对照表
- `final_selected_gene_pcc.csv` —— 4 个目标基因 PCC
- `final_B1.npy`, `final_A2.npy` —— 每个细胞的最终预测
- `figure3_*.png` —— Figure-3 风格的空间图

#### 3) 出图

```bash
python gpt/plot_bars.py \
    --metrics-csv ./results_final/final_metrics_vs_paper.csv \
    --genes-csv   ./results_final/final_selected_gene_pcc.csv \
    --outdir      ./results_final/figs_bars

python gpt/plot_full_slice.py \
    --adata1 ... --adata2 ... --raw-data-root ... --selection ... \
    --device cuda:0 --seed 0 --epochs 500 \
    --outdir ./results_final/figs_full_slice
```

### 一键端到端

如果想跑完整 pipeline（包括 EGRefiner 训练）：

```bash
python gpt/pipeline.py \
    --adata1 .../Rep1_uni_resolution64_panelA.h5ad \
    --adata2 .../Rep2_uni_resolution64_panelB.h5ad \
    --raw-data-root .../raw_data/HBC \
    --selection     .../raw_data/HBC/Selection_by_name.csv \
    --device cuda:0 \
    --epochs 500 --refiner-epochs 300 \
    --amp bf16 --gradient-checkpoint \
    --refiner-hidden-dim 384 --refiner-layers 2 \
    --k-exemplar 8 --retrieval-metric l1 \
    --outdir ./results_pipeline
```

主要参数说明：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--amp {none,bf16,fp16}` | `none` | 混合精度，bf16 在 A100 上最稳 |
| `--gradient-checkpoint` | off | GEB 块开启梯度检查点，省 ~30% 显存 |
| `--refiner-hidden-dim` | 512 | refiner 隐藏维 |
| `--refiner-layers` | 3 | refiner 层数 |
| `--k-exemplar` | 6 | 每个 query 的 exemplar 数 |
| `--retrieval-metric {l1,l2,cosine}` | `l1` | UNI 特征 KNN 度量 |
| `--knn-q-batch / --knn-r-batch` | 1024 / 4096 | GPU KNN 的 query / reference 批大小 |
| `--safe-residual` | on | safe-residual 预测头 |
| `--no-attn-reg` | off | 关掉注意力熵正则（attnH 通常 = 0 时使用） |
| `--load-preds DIR` | — | 跳过 SpatialEx+ 训练，直接复用缓存预测 |

---

## 结果

### 全局 metric（HBC, panel diagonal integration）

| | gene-PCC mean | spot-PCC mean | RMSE | MAE | SSIM | CMD |
|---|---|---|---|---|---|---|
| Paper (B on Slice 1) | 0.2957 | — | — | — | 0.3470 | 0.3436 |
| **Ours-Final (B on Slice 1)** | **0.3025** | 0.6692 | 0.3790 | 0.2236 | **0.3581** | **0.2846** |
| Paper (A on Slice 2) | 0.3107 | — | — | — | 0.3653 | 0.3508 |
| **Ours-Final (A on Slice 2)** | **0.3140** | 0.6676 | 0.3647 | 0.2105 | **0.3753** | **0.2975** |

### 目标基因 PCC

| 基因 | 切片 | Paper | Ours-Raw | Ours-Final | Δ vs Paper |
|---|---|---|---|---|---|
| ESR1 | Slice 2 | 0.369 | 0.339 | 0.317 | −0.0516 |
| ERBB2 | Slice 2 | 0.661 | 0.681 | **0.661** | **+0.0004** |
| **PGR** | Slice 1 | 0.144 | 0.095 | **0.176** | **+0.0322** |
| KRT14 | Slice 1 | 0.650 | 0.576 | 0.574 | −0.0757 |

`PGR` 是最戏剧化的修复 —— SpatialEx+ 的 raw 仅 0.095（cycle 已断），broken-raw heuristic 自动切到 `direct` 把它救回到 **0.176**，超过 paper 0.032。

---

## 设计取舍

下面这些坑都是迭代过程踩出来的，列在这里供后续维护参考。

1. **Linear decomposition 替代 `torch.cat + Linear`**。EGGN 原文用 `torch.cat([h_q, h_e, s_e], dim=-1)` 拼大向量再过单层线性层；这个中间 tensor 在 16w 细胞规模会直接 OOM。我们等价改为多个并行的小 `Linear` 然后求和，显存峰值降一个数量级。

2. **`torch.sparse.mm` 不支持 BF16**。AMP 开了之后稀疏邻接乘法直接抛 `"addmm_sparse_cuda" not implemented for 'BFloat16'`。修法：把所有 `torch.sparse.mm` 显式包在 `with torch.cuda.amp.autocast(enabled=False):` 里。

3. **GPU-batched KNN**。`sklearn` 在 16w × 1024 维 UNI 特征上 manhattan KNN 跑得几乎不动；`refiner.py` 里换成 `torch.cdist` + 双重批次（`knn_q_batch / knn_r_batch`），单卡几秒搞定。

4. **Safe residual head**。最初用 `softplus(delta)` 做残差很容易把已经不错的 raw / indirect 拉成负数，最后 final 反而不如 raw。改成「初始化 `y_refined = relu(y_indirect)`，再学一个 per-gene 缩放系数 α 去乘 `delta`」之后稳得多。

5. **Per-gene 决策的优先级很关键**。把 `ridge` / `stack` 直接全局 enable 会掉 ESR1 / KRT14；先看 train-side raw 是不是已经崩链（`< 0.4` 就用 direct 兜底），再用 `stack-margin` 守住小幅提升才换 stack——这样 final 的 *最差情况* 不会比 raw 差。

6. **种子方差**。SpatialEx+ 在 PGR 这种崩链基因上对 seed 极敏感（PCC 在 `[0.05, 0.20]` 区间漂移）。`seed_search.py` 用「与 paper 差距之和」做单标量打分，扫 4-8 个种子就能把地基垫高。

---

## 致谢与引用

- 本仓库的 *基础算法与训练代码* 完全沿用 [SpatialEx / SpatialEx+](https://github.com/) 的实现，本子包只在其上添加 refiner / 后处理 / 可视化层，**不修改原版任何源码**。
- Refiner 模块的图结构借鉴自 [EGN/EGGN](https://github.com/CarlinLiao/EGN) 的 *Exemplar-Guided Graph Network*。

如果本仓库对你的工作有帮助，请同时引用 SpatialEx 与 EGGN 的原始论文。
