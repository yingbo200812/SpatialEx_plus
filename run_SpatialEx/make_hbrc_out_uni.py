import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialEx as se
from SpatialEx import utils as se_utils


def patch_uni_path(uni_dir: str):
    """在不改源码的情况下，把 utils.create_ImageEncoder 里 UNI 的 local_dir 替换掉。"""
    import timm
    orig = se_utils.create_ImageEncoder

    def new_create_ImageEncoder(model_name='resnet50', pretrained=True, frozen=True):
        if model_name.lower() == 'uni':
            print(f"[patched] Loading UNI from: {uni_dir}")
            model = timm.create_model(
                "vit_large_patch16_224",
                img_size=224, patch_size=16, init_values=1e-5,
                num_classes=0, dynamic_img_size=True,
            )
            state = torch.load(os.path.join(uni_dir, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state, strict=True)
            if frozen:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
            return model
        return orig(model_name=model_name, pretrained=pretrained, frozen=frozen)

    se_utils.create_ImageEncoder = new_create_ImageEncoder
    import SpatialEx.preprocess as se_pp
    se_pp.create_ImageEncoder = new_create_ImageEncoder


def make_one(save_root: str, rep_tag: str, resolution: int, device: str, batch_size: int):
    coor_path = os.path.join(save_root, f"HBRC_{rep_tag}_cell_coor.csv")
    img_path = os.path.join(save_root, f"Xenium_FFPE_Human_Breast_Cancer_{rep_tag}_he_image.ome.tif")
    out_npy = os.path.join(save_root, f"HBRC_{rep_tag}_Out_uni.npy")

    print(f"\n=========== {rep_tag} ===========")
    print("coor :", coor_path)
    print("img  :", img_path)
    print("npy  :", out_npy)

    out_spatial = pd.read_csv(coor_path, index_col=0)
    n0 = out_spatial.shape[0]

    patch_radius = int(resolution / 2.0)

    img, scale = se.pp.Read_HE_image(img_path)
    H, W = img.shape[:2]

    # 对图像四周做 zero-padding（各扩展 patch_radius 像素），
    # 这样所有细胞（包括边缘处的）都能提取完整 patch，无需过滤
    img_padded = np.pad(img,
                        ((patch_radius, patch_radius),
                         (patch_radius, patch_radius),
                         (0, 0)),
                        mode='constant', constant_values=0)
    print(f"原始图像 {H}x{W} -> padding 后 {img_padded.shape[0]}x{img_padded.shape[1]}")

    # 构造 AnnData，坐标整体偏移 +patch_radius 以适配 padded 图像
    adata = sc.AnnData(np.zeros((n0, 1), dtype=np.float32))
    coords = out_spatial[["image_col", "image_row"]].values.astype(np.int64)
    coords = coords + patch_radius
    adata.obsm["image_coor"] = coords

    # 提取 UNI patch embedding
    he_patches, adata = se.pp.Tiling_HE_patches(resolution, adata, img_padded, key="image_coor")
    adata = se.pp.Extract_HE_patches_representaion(
        he_patches, adata=adata,
        image_encoder="uni", device=device, store_key="he",
        img_batch_size=batch_size,
    )

    emb = np.asarray(adata.obsm["he"])
    print(f"细胞总数 {n0} -> embedding {emb.shape}")

    if emb.shape[0] != n0:
        print(f"[警告] Tiling 内部仍丢弃了部分细胞: {n0} -> {emb.shape[0]}")

    np.save(out_npy, emb)
    print("saved:", out_npy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="包含 Human_Breast_Cancer_Rep1/ 与 Human_Breast_Cancer_Rep2/ 的父目录")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patch-uni-path", default=None,
                    help="若不改 SpatialEx/utils.py，可在此指定 UNI 权重所在目录")
    args = ap.parse_args()

    if args.patch_uni_path:
        patch_uni_path(args.patch_uni_path)

    for rep in ("Rep1", "Rep2"):
        save_root = os.path.join(args.root, f"Human_Breast_Cancer_{rep}")
        if not os.path.isdir(save_root):
            print(f"[skip] not found: {save_root}")
            continue
        make_one(save_root, rep, args.resolution, args.device, args.batch_size)


if __name__ == "__main__":
    main()