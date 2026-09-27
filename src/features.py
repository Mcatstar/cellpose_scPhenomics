"""
features.py —— 从大图生成 Cellpose 训练数据。

流程（中间产物都在 INTERIM_DATA_DIR, 最终落到 PROCESSED_DATA_DIR/train/）:
  Step1  ROI 归一化: xxx_img.tif + xxx_roi_crop.tif → INTERIM/xxx_img_norm.tif
  Step2  切片：      INTERIM/xxx_img_norm.tif + xxx_masks.tif
                     → INTERIM/xxx_y{y}-x{x}_img.tif / _masks.tif
  Step3  尺寸统一：  INTERIM/*_y*-x*_img.tif / _masks.tif
                     → PROCESSED/train/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tifffile import imread, imwrite
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.padding import compute_target_size, pad_to_size, tif_shape
from src.utils.roi import normalize_roi
from src.utils.tiling import iter_tiles

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    crop_size: int = 512,
    stride: Optional[int] = None,
    min_mask_fraction: float = 0.0,
    multiple_of: int = 16,
) -> None:
    logger.info("Generating features from dataset...")
    logger.info(f"input path: {input_path}")
    logger.info(f"output path: {output_path}")

    train_path = output_path / "train"
    train_path.mkdir(parents=True, exist_ok=True)

    stride = stride if stride is not None else crop_size

    # ------------------------------------------------------------------
    # Step 1: ROI 归一化
    # ------------------------------------------------------------------
    img_path_list = sorted(
        p
        for p in input_path.iterdir()
        if p.is_file()
        and p.name.endswith("_img.tif")
        and not p.name.endswith("_img_norm.tif")
    )

    for img_path in tqdm(img_path_list, desc="ROI Normalizing"):
        roi_path = img_path.with_name(
            img_path.name.replace("_img.tif", "_roi_crop.tif")
        )
        img = imread(img_path)
        roi = imread(roi_path)
        img_norm = normalize_roi(img, roi)
        imwrite(
            input_path / img_path.name.replace("_img.tif", "_img_norm.tif"),
            img_norm.astype("float32"),
        )

    # ------------------------------------------------------------------
    # Step 2: 切片
    # ------------------------------------------------------------------
    n_tiles_total = 0
    pairs_found = 0

    for img_norm_path in tqdm(
        sorted(input_path.glob("*_img_norm.tif")), desc="Tiling"
    ):
        base_name = img_norm_path.name.replace("_img_norm.tif", "")
        mask_path = input_path / f"{base_name}_masks.tif"
        if not mask_path.exists():
            logger.warning(f"缺少 mask, 跳过：{img_norm_path.name}")
            continue

        img = imread(img_norm_path)
        mask = imread(mask_path)

        if img.shape[-2:] != mask.shape[-2:]:
            logger.warning(
                f"形状不匹配, 跳过：{img_norm_path.name} "
                f"{img.shape} vs {mask.shape}"
            )
            continue

        H, W = img.shape[-2:]

        # 小图：整张写出，交给 Step 3 padding 对齐
        if H < crop_size or W < crop_size:
            if min_mask_fraction > 0 and (mask > 0).mean() < min_mask_fraction:
                logger.info(f"{img_norm_path.name}: mask 比例不足, 跳过")
                continue
            name = f"{base_name}_y0-x0"
            imwrite(input_path / f"{name}_img.tif", img)
            imwrite(input_path / f"{name}_masks.tif", mask)
            n_tiles_total += 1
            pairs_found += 1
            continue

        # 正常滑窗
        n = 0
        for y, x, img_tile, mask_tile in iter_tiles(
            img, mask, crop_size, stride, min_mask_fraction
        ):
            name = f"{base_name}_y{y}-x{x}"
            imwrite(input_path / f"{name}_img.tif", img_tile)
            imwrite(input_path / f"{name}_masks.tif", mask_tile)
            n += 1
        n_tiles_total += n
        pairs_found += 1

    logger.info(
        f"切片完成：{pairs_found} 对, 共 {n_tiles_total} 个 tile → {input_path}"
    )

    # ------------------------------------------------------------------
    # Step 3: 尺寸统一
    # ------------------------------------------------------------------
    tile_img_paths = sorted(input_path.glob("*_y*-x*_img.tif"))
    tile_mask_paths = sorted(input_path.glob("*_y*-x*_masks.tif"))
    all_tile_paths = tile_img_paths + tile_mask_paths

    if not all_tile_paths:
        logger.warning("未找到任何 tile, 退出")
        return

    shapes = [
        tif_shape(p)
        for p in tqdm(all_tile_paths, desc="读取尺寸", unit=" file", leave=False)
    ]
    target_h, target_w, needs_pad = compute_target_size(
        shapes, multiple_of=multiple_of
    )

    if needs_pad:
        logger.info(f"发现尺寸不一致, 统一 pad 到 ({target_h}, {target_w})")
    else:
        logger.info(f"尺寸已一致 ({target_h}, {target_w}), 仅做复制")

    for path in tqdm(all_tile_paths, desc="写入 train", unit=" file"):
        arr = imread(path)
        arr = pad_to_size(arr, target_h, target_w)
        imwrite(train_path / path.name, arr)

    logger.success(f"已输出 {len(all_tile_paths)} 个文件 → {train_path}")


if __name__ == "__main__":
    app()