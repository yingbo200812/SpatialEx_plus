#!/usr/bin/env bash
# ============================================================
# SpatialEx-pro 远程服务器一键运行脚本。
#
# 评估目标：复现 SpatialEx 论文 Fig. 2c/d
#   （HBC Rep1 / Rep2，313 基因 panel，leave-one-slice-out 跨片预测）。
#
# 论文 baseline 数字（用于对比图的参考柱）：
#   Train Slice1 -> Test Slice2   PCC=0.2576  SSIM=0.3654  CMD=0.2149
#   Train Slice2 -> Test Slice1   PCC=0.2733  SSIM=0.3809  CMD=0.2033
#   Marker PCC (Slice2)            EPCAM=0.756  ESR1=0.317  PGR=0.113
#
# 本脚本流程：
#   1) 跑 SpatialEx-pro 训练 + 推理（默认 3 种子集成）；
#   2) 在同一份数据切分上跑 SpatialEx baseline，做严格并排对比；
#   3) 调 make_plots.py 生成主指标 + marker PCC 对比图；
#   4) 终端打印对比表。
#
# 全部产物默认写到：
#   /data1/linxin/1/SpatialEx_pro/results/run_default
#
# 注：脚本会自动根据 *自身所在文件夹名* 决定 import 路径，因此把外层
# 文件夹改名为 SpatialEx_pro / SpatialEx_pro1 / SpatialEx-pro 都不影响。
# ============================================================

set -e

# ---- 自动定位包目录（兼容 SpatialEx_pro / SpatialEx_pro1 / SpatialEx-pro）
PKG_DIR="$(dirname "$(readlink -f "$0")")"
PKG_NAME="$(basename "$PKG_DIR")"
REPO_ROOT="$(dirname "$PKG_DIR")"

# ---- CUDA 显存优化 ----------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 数据 / 输出路径 --------------------------------------------
H5AD_ROOT="/data1/linxin/1/SpatialEx/data/Human_breast_cancer_small/Human_breast_cancer_small"
RAW_ROOT="/data1/linxin/1/SpatialEx/data/raw_data/HBC"

ADATA1="$H5AD_ROOT/Human_Breast_Cancer_Rep1/Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad"
ADATA2="$H5AD_ROOT/Human_Breast_Cancer_Rep2/Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad"

OUTDIR="/data1/linxin/1/SpatialEx_pro/results/run_default"
DEVICE="cuda:0"
EPOCHS=500
N_SEEDS=3                           # 多种子集成

mkdir -p "$OUTDIR"

echo "======================================================="
echo " SpatialEx-pro :: Fig. 2c/d cross-slice evaluation"
echo "  Pkg dir      : $PKG_DIR"
echo "  Slice1 h5ad  : $ADATA1"
echo "  Slice2 h5ad  : $ADATA2"
echo "  Out dir      : $OUTDIR"
echo "  GPU          : $DEVICE"
echo "  Epochs       : $EPOCHS"
echo "  Seeds        : $N_SEEDS"
echo "======================================================="

cd "$REPO_ROOT"

python "$PKG_NAME/run_train_eval.py" \
    --h5ad-root      "$H5AD_ROOT"   \
    --raw-data-root  "$RAW_ROOT"    \
    --adata1         "$ADATA1"      \
    --adata2         "$ADATA2"      \
    --device         "$DEVICE"      \
    --epochs         "$EPOCHS"      \
    --n-seeds        "$N_SEEDS"     \
    --outdir         "$OUTDIR"      \
    --run-baseline                  \
    \
    --hidden-dim     512            \
    --num-layers     2              \
    --dropout        0.1            \
    \
    --lambda-mse-cell        1.0    \
    --lambda-mse-spot        0.5    \
    --lambda-pearson         1.0    \
    \
    --lambda-anchor-mse      0.5    \
    --lambda-anchor-pearson  0.3    \
    --lambda-anchor-warmup   80     \
    --anchor-k               8      \
    --anchor-sim-floor       0.3    \
    \
    --lambda-spatial-tv      0.02   \
    --tv-max-edges           200000 \
    --lambda-dgi             0.3    \
    \
    --lambda-cmd-align       0.5    \
    --cmd-align-subsample    30000  \
    \
    --marker-genes           "EPCAM,ESR1,PGR,ERBB2,KRT14" \
    --marker-weight          2.0    \
    --marker-pearson-weight  3.0    \
    \
    --alpha-spatial          0.1    \
    --beta-anchor            0.1    \
    --refine-anchor-k        15

echo ""
echo "======================================================="
echo " 训练完成。产物文件："
ls -lh "$OUTDIR"

# ---- 生成对比图 -----------------------------------------------
echo ""
echo " 生成对比图 ..."
python "$PKG_NAME/make_plots.py" --results-dir "$OUTDIR"

echo ""
echo "=== SpatialEx-pro 指标 ==="
OUTDIR="$OUTDIR" python - <<'PY'
import os, pandas as pd
out = os.environ['OUTDIR']
pro_csv = os.path.join(out, 'metrics_spatialex_pro.csv')
bl_csv  = os.path.join(out, 'metrics_baseline.csv')
cmp_csv = os.path.join(out, 'compare.csv')

if os.path.isfile(pro_csv):
    print(">>> SpatialEx-pro <<<")
    print(pd.read_csv(pro_csv).T.to_string(header=False))
if os.path.isfile(bl_csv):
    print("\n>>> SpatialEx baseline (this run) <<<")
    print(pd.read_csv(bl_csv).T.to_string(header=False))
if os.path.isfile(cmp_csv):
    print("\n>>> Side-by-side (delta = pro - baseline) <<<")
    df = pd.read_csv(cmp_csv)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
PY
echo "======================================================="
