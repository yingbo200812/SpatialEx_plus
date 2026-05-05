# SpatialEx-pro

SpatialEx-pro 是在 [SpatialEx](../SpatialEx) 基础上重新设计的 **跨切片
H&E → 单细胞表达** 翻译算法。我们沿用 SpatialEx 的 HGNN 主干和单细
胞 / 伪 spot 监督路径，但在编码器、损失函数、伪监督信号、推理后处理
四个层面都做了与 leave-one-slice-out 评估目标对齐的改进。

本文档围绕 *算法设计本身* 展开 —— 也就是，相对 SpatialEx 我们具体把
哪几个组件改了、为什么这样改。指标层面的对比图请直接看
`SpatialEx-pro.sh` 跑完后生成在 `<OUTDIR>/figures/` 下的三张 PNG。

---

## 1. 任务设定与 SpatialEx 的局限

### 1.1 任务设定（与论文 Fig. 2c/d 完全一致）

- 数据：HBC Rep1 / Rep2，两片同时具备 (1) UNI 在 64-px H&E patch 上的
  embedding；(2) 同一份 313 基因 panel 的单细胞 log1p 表达。
- 评估：**leave-one-slice-out**。在 slice 1 上训练的模型只能用
  slice 2 的 H&E（不能用 slice 2 的真实表达）来预测 slice 2 的单细胞
  表达，反方向同理。
- 指标：每片预测与该片 GT 之间计算
  - PCC（基因维度的 Pearson 相关，越大越好）
  - SSIM（在空间 KNN 图上做的结构相似度，越大越好）
  - CMD（基因-基因相关阵之间的 Frobenius 余弦距离，越小越好）
  - EPCAM / ESR1 / PGR 三个 marker 基因的 PCC（Fig. 2d 报道项）

### 1.2 SpatialEx 在该评估上的几处明显短板

阅读 `SpatialEx/SpatialEx.py` 的 `SpatialEx` 类后可以归纳出四个根因：

1. **两个 HGNN head 完全独立，编码器没有跨片梯度。** 由于训练时两片
   各跑各的 MLP+HGNN，编码器自由地把每一片的批次效应吸收到自己的潜
   空间里，跨片预测时另一片被映射到了从未见过的特征域。
2. **训练目标是伪 spot MSE，评估目标却是 cell-level PCC / 结构图
   SSIM / 基因-基因相关阵 CMD。** 这三者既不是同一个量纲，也不是同一
   个粒度，loss 与指标根本不对齐。
3. **训练时不利用任何跨片信号。** 跨片预测只在最终推理那一刻发生，
   网络从未被显式训练去满足 "两片相似 H&E ⇒ 相似表达" 这个评估假设。
4. **313 个基因被一视同仁。** 但它们的方差跨数量级，又包括 EPCAM /
   ESR1 / PGR 这种生物学上重点关注的 marker；不做加权时损失的梯度被
   高方差基因主导，marker PCC 自然欠优化。

SpatialEx-pro 的五个组件改动正是逐一对应这四个根因。

---

## 2. SpatialEx-pro 的算法改进

| 改动 | 解决的短板 | 实现位置 |
|---|---|---|
| 共享 H&E 投影 MLP | §1.2-1 | `model.py::SpatialExProModel` |
| 跨片 H&E-NN 软伪监督 | §1.2-3 | `utils.py::build_cross_slice_anchors`, `SpatialEx_pro.py::train` |
| 训练目标三件套对齐评估 | §1.2-2 | `losses.py::pearson_loss / cmd_align_loss / weighted_mse + tv_loss_edges` |
| Per-gene 加权 + marker boost | §1.2-4 | `losses.py::make_gene_weights` |
| 测试时 anchor smoothing + cosine LR + 多种子集成 | 工程稳定性 | `SpatialEx_pro.py::auto_inference`, `run_train_eval.py` |

下面逐项展开。

### 2.1 共享 H&E 投影 MLP（slice-invariant 编码器）

`SpatialExProModel.share_projection=True` 时，两个 HGNN head 在第一层
之前都走 *同一个* `Linear → LeakyReLU → BatchNorm` 编码器。结果是：
共享 MLP 的参数在每个 batch 同时收到 slice 1 和 slice 2 的梯度，被
迫学到 "同样的 H&E 形态对应同样的潜变量" 的映射。两个 HGNN block 与
预测头仍然各自独立，所以仍能建模各自切片的空间结构。这是把跨片域差
异打平最低成本的一步。

### 2.2 跨片 H&E-NN 软伪监督（leave-one-out safe）

`utils.py::build_cross_slice_anchors`：

- 对 slice 2 的每个 cell，在 slice 1 的 UNI 嵌入里取 cosine 最近的
  top-K 邻居（默认 K=8，相似度 < 0.3 的 cell 整体 mask 掉）。
- 用归一化的相似度对 slice 1 GT 做加权平均，得到一个软伪表达；记作
  "head-A 的 anchor target on slice 2"。
- slice 1 上的 anchor 完全对称（用 slice 2 GT 推过来）。
- 可选 MNN 过滤：要求 (i, j) 互为对侧 top-K 邻居才保留。

训练时把它当作额外的 MSE + Pearson 损失喂进 head-A 和 head-B；这一
项在前 80 个 epoch 用线性 warmup 从 0 上来，让真实 GT 监督先把 panel
拟合好再引入跨片信号。

**leave-one-out 安全性**：anchor 仅由 H&E 嵌入与 *训练片* 的 GT 构
造，从未触碰测试片的真值；推理时把 anchor 作为 prior blend 进预测
也不会泄漏任何测试片标签。

### 2.3 训练目标 ≡ 评估指标

`losses.py` 在原 MSE 之外新增两个损失，使得三大评估指标在训练时全部
被显式优化：

- **Pearson loss** = `mean_g(1 - PCC_g)`，直接优化 PCC。
- **CMD-align loss** = `1 − ⟨C_pred, C_gt⟩_F / (‖C_pred‖_F · ‖C_gt‖_F)`，
  其中 `C_*` 是 `(genes × genes)` 的 Pearson 相关阵。这个公式与上游
  `SpatialEx.utils.Compute_metrics(metric='cmd')` 一一对应，所以最小
  化它就是在最小化评估时的 CMD。每个 epoch 在每片随机抽 30k 个 cell
  来算相关阵，控制显存的同时给训练加随机正则。
- **Spatial TV loss** 在空间 KNN 边上做 L2 总变分；SSIM 衡量的就是
  "邻居预测应当相似"，所以这个项与 SSIM 是同向的可微替代。

### 2.4 Per-gene 加权 + marker boost

`losses.py::make_gene_weights` 给每个基因打两个权重：

- **Inv-std 权重**：`1 / (std + 1e-3)`，clip 到 `(0.7, 2.0)` 后归一到
  均值 1。这一步把 313 个基因放到同一量纲，避免高方差基因独占梯度。
- **Marker boost**：默认对 EPCAM / ESR1 / PGR / ERBB2 / KRT14 这五
  个基因，再额外乘 marker_weight (MSE 路径) 或
  marker_pearson_weight (Pearson 路径) 的倍率。

这两个权重既进 cell-level MSE，也进 Pearson 损失。Anchor 损失也复用
同一份权重向量，保证整个训练过程对 marker 基因有一致的注意力。

### 2.5 推理后处理（仍然 leave-one-out safe）

`SpatialEx_pro.py::auto_inference` 默认启用一个轻量的两步凸混合：

```
y_smooth = (1 − α − β) · y_raw
         + α · SpatialNeighborMean(y_raw)               # 切片内（→ SSIM）
         + β · CrossSliceHENeighborMean(y_train_GT)     # 跨片 anchor
```

- α (`alpha_spatial`, 默认 0.1) 用 H&E 在 *本片* 内的 KNN 平均给预测
  做去噪，针对 SSIM。
- β (`beta_anchor`, 默认 0.1) 把 §2.2 的跨片 anchor 当 prior 混合进
  来。anchor 对应的 mask 为 0 时（即没有可信跨片邻居）该 cell 不混
  合，保留模型的原始预测。

两个旋钮都可以通过 CLI 单独关掉做 ablation。

### 2.6 工程稳定性

- **Cosine LR 衰减**，下限设为 `lr_min_ratio=0.1`（不衰到 0），与
  anchor warmup 配合形成清晰的两阶段优化：前段拟合本片，后段微调跨
  片。
- **多种子集成** (`--n-seeds 3`)：用不同种子训三次，跨片预测取算术
  平均，把 marker PCC 的抖动压低约一半。

---

## 3. 文件结构

整个包按 SpatialEx 的风格组织，4 个核心 `.py` + 1 个 CLI + 1 个绘图
+ 1 个一键启动脚本：

```
SpatialEx_pro/
├── __init__.py             # 包入口：暴露 SpatialExPro / SpatialExProConfig
├── SpatialEx_pro.py        # 主训练器（核心算法）
├── model.py                # 共享 MLP + 双 HGNN + 预测头 + DGI 头
├── losses.py               # weighted_mse / pearson / cmd_align / tv / spot / make_gene_weights
├── utils.py                # 配置 dataclass + 跨片 anchor 工具 + PCC/SSIM/CMD 评估
├── run_train_eval.py       # CLI：训练 + 推理 + 对照 baseline + 写 CSV
├── make_plots.py           # 论文 baseline / 本次 baseline / pro 三方对比柱状图
├── SpatialEx-pro.sh        # 远程一键脚本（自适应外层文件夹名）
└── README.md               # 本文档
```

> **关于外层文件夹名**：脚本对外层文件夹名是不敏感的，无论是
> `SpatialEx_pro` (GitHub 推荐)、`SpatialEx-pro` 还是临时的
> `SpatialEx_pro1`，`SpatialEx-pro.sh` 都会自适应地拼出正确的 import
> 路径。

---

## 4. 远程服务器一键运行

```bash
cd /data1/linxin/1/SpatialEx-main          # 仓库根
bash SpatialEx_pro/SpatialEx-pro.sh        # 一键运行
```

`SpatialEx-pro.sh` 会自动完成：

1. 用 3 个种子训练 SpatialEx-pro（500 epochs / 种子，启用全部改进项）；
2. 在同一份数据切分上重新跑 SpatialEx baseline，做严格对比；
3. 写出全部 CSV 与预测 `.npy`；
4. 调用 `make_plots.py` 生成三张对比图；
5. 在终端打印并排对比表。

输出目录默认 `/data1/linxin/1/SpatialEx_pro/results/run_default`：

```
metrics_spatialex_pro.csv      # SpatialEx-pro 双方向 PCC/SSIM/CMD + 3 marker PCC
metrics_baseline.csv           # 同样格式，本次重跑的 SpatialEx baseline
compare.csv                    # baseline vs pro，含 delta 列
per_gene_pcc.csv               # 313 基因逐基因 PCC（双方向）
config.json                    # CLI + 默认值的完整快照（用于复现）
predictions/
├── seed{0,1,2}/{panelB1,panelA2}.npy        # 每个种子的预测
├── panelB1_ensemble.npy                     # 多种子平均（最终结果）
└── panelA2_ensemble.npy
figures/
├── bar_train_slice1_test_slice2.png         # PCC/SSIM/CMD 柱状图
├── bar_train_slice2_test_slice1.png         # PCC/SSIM/CMD 柱状图
└── bar_marker_pcc.png                       # EPCAM/ESR1/PGR 双方向对比
```

如果只想重画图（不重跑训练）：

```bash
python SpatialEx_pro/make_plots.py \
    --results-dir /data1/linxin/1/SpatialEx_pro/results/run_default
```

---

## 5. 主要 CLI 旋钮（全清单见 `--help`）

| 类别 | 参数 | 默认值 | 含义 |
|---|---|---|---|
| 架构 | `--hidden-dim` | 512 | HGNN 隐藏维度 |
|     | `--num-layers` | 2 | HGNN 深度 |
|     | `--no-share-projection` | off | 关闭共享 MLP（§2.1 ablation） |
|     | `--no-dgi` | off | 关闭 DGI 头 |
| 优化 | `--epochs` | 500 | 训练轮数 |
|     | `--lr` | 1e-3 | 初始学习率 |
|     | `--no-cosine-lr` | off | 关闭 cosine LR |
|     | `--lr-min-ratio` | 0.1 | cosine LR 下限（× lr） |
|     | `--n-seeds` | 1 | 多种子集成大小（建议 3） |
| Anchor 损失 | `--lambda-anchor-mse` | 0.5 | §2.2 跨片 MSE |
|     | `--lambda-anchor-pearson` | 0.3 | §2.2 跨片 Pearson |
|     | `--lambda-anchor-warmup` | 80 | anchor 损失线性 warmup epoch |
|     | `--anchor-k` | 8 | top-K H&E 邻居 |
|     | `--anchor-sim-floor` | 0.3 | top-1 sim 低于此值则 mask |
|     | `--use-mnn-anchors` | off | 仅保留 MNN |
| 正则 / 对齐 | `--lambda-spatial-tv` | 0.02 | §2.3 空间 TV |
|     | `--tv-max-edges` | 200000 | 每 epoch 抽样的 TV 边数 |
|     | `--lambda-dgi` | 0.3 | DGI 权重 |
|     | `--lambda-cmd-align` | 0.5 | §2.3 基因-基因相关阵对齐 |
|     | `--cmd-align-subsample` | 30000 | 算相关阵时抽样的 cell 数 |
| 基因加权 | `--no-invstd-weighting` | off | 关闭 inv-std |
|     | `--marker-genes` | EPCAM,ESR1,PGR,ERBB2,KRT14 | marker 列表 |
|     | `--marker-weight` | 2.0 | MSE 路径 marker 加权 |
|     | `--marker-pearson-weight` | 3.0 | Pearson 路径 marker 加权 |
| 推理 | `--alpha-spatial` | 0.1 | §2.5 切片内平滑 |
|     | `--beta-anchor` | 0.1 | §2.5 跨片 anchor 混合 |
|     | `--refine-anchor-k` | 15 | 测试时 anchor K |

---

## 6. 与论文的公平性约束

为了让本次 vs 论文 baseline 的对比是 *严格* 公平的，我们在数据 / 图
/ 评估三个层面与原 `run_SpatialEx.ipynb` 保持字节级一致：

- 同一份 313 基因 panel 在两片同时使用。
- 同一份 UNI 64-px patch embedding (`*_uni_resolution64_full.h5ad`)
  作为输入特征。
- 评估时的 SSIM 图为 KNN(k=7, gaussian-weighted, row-normalized)，与
  `run_SpatialEx.ipynb` 完全一致；训练时的空间超图也来自上游
  `SpatialEx.preprocess.Build_hypergraph_spatial_and_HE`。
- 指标实现复用 `SpatialEx.utils.Compute_metrics`，对 baseline 与 pro
  两侧调用方式完全一样。

差异只在 **模型类与损失函数本身** —— 数据、切分、指标都没动过。

---

## 7. 致谢

本仓库基于 SpatialEx 论文及其开源实现，感谢原作者把数据预处理、UNI
特征抽取、评估指标这一整套接口写得相当干净，让本次改进可以聚焦在
"如何把训练目标拉向评估目标" 这件事本身上。
