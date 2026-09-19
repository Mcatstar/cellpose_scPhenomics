"""
features.py —— 从大图生成 Cellpose 训练数据。

流程（所有中间产物都在 INTERIM_DATA_DIR, 仅最终数据落到 PROCESSED_DATA_DIR) :
  Step1  ROI 归一化：  xxx_img.tif + xxx_roi_crop.tif
                       →  INTERIM/xxx_img_norm.tif
  Step2  切片：        INTERIM/xxx_img_norm.tif + xxx_masks.tif
                       →  INTERIM/xxx_y{y}-x{x}_img.tif / _masks.tif
  Step3  尺寸统一：    INTERIM/*_y*-x*_img.tif / _masks.tif
                       →  PROCESSED/train/  (尺寸一致时 pad 自动 no-op)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from loguru import logger
from numpy.typing import NDArray
from tifffile import imread, imwrite
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import Padder, Tiler

app = typer.Typer()


def select_channels(arr: NDArray, channels: list[int]) -> NDArray:
    """只保留指定通道索引, 2D 数组原样返回"""
    if arr.ndim < 3:
        return arr
    return arr[channels]

# ROI 归一化（features 专属逻辑）
def normalize_roi(img: NDArray, roi: NDArray, lower: int = 1, upper: int = 99) -> NDArray:
    """ROI 内 1-99 百分位归一化到 [0, 1]; ROI 外置 0"""
    img = img.astype(np.float32)

    if img.ndim == 2:
        vals = img[roi > 0]
        if vals.size == 0:
            return np.zeros_like(img)
        low, high = np.percentile(vals, [lower, upper])
        high = max(high, low + 1e-6)
        out = np.clip((img - low) / (high - low), 0, 1)
        out[roi == 0] = 0
        return out

    out = np.zeros_like(img)
    for c in range(img.shape[0]):
        vals = img[c][roi > 0]
        if vals.size == 0:
            continue
        low, high = np.percentile(vals, [lower, upper])
        high = max(high, low + 1e-6)
        out[c] = np.clip((img[c] - low) / (high - low), 0, 1)
        out[c][roi == 0] = 0
    return out


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

    # 定义所需数据字典
    CHANNEL_MAP: dict[str, dict[str, int]] = {
        "mito":  {"img": 0, "masks": 0},
        "lipid": {"img": 1, "masks": 1},
    }
    IMG_CHANNELS: list[int] = [v["img"] for v in CHANNEL_MAP.values()]

    # ROI 归一化
    img_path_list = sorted(
        p
        for p in input_path.iterdir()
        if p.is_file()
        and p.name.endswith("_img.tif")
        and not p.name.endswith("_img_norm.tif")
        and "_y" not in p.stem.split("_img")[0][-6:]
    )

    for img_path in tqdm(img_path_list, desc="ROI Normlizing"):
        roi_path = img_path.with_name(img_path.name.replace("_img.tif", "_roi_crop.tif"))
        img = imread(img_path)
        roi = imread(roi_path)
        img = select_channels(img, IMG_CHANNELS)
        img_norm = normalize_roi(img, roi)
        img_norm_path = input_path / img_path.name.replace("_img.tif", "_img_norm.tif")
        imwrite(img_norm_path, img_norm.astype(np.float32))

    # 切片
    tiler = Tiler(
        crop_size=crop_size,
        stride=crop_size,  # 滑动步长，此时0%重叠区域
        min_mask_fraction=min_mask_fraction,
    )

    n_tiles_total = 0
    pairs_found = 0
    for img_norm_path in tqdm(sorted(input_path.glob("*_img_norm.tif")), desc="Tiling"):
        base_name = img_norm_path.name.replace("_img_norm.tif", "")
        mask_path = input_path / f"{base_name}_masks.tif"
        if not mask_path.exists():
            logger.warning(f"缺少 mask, 跳过：{img_norm_path.name}")
            continue
        n = tiler.tile(
            img_norm_path,
            mask_path,
            input_path,
            base_name=base_name,
            img_suffix="_img.tif",
            mask_suffix="_masks.tif",
        )
        n_tiles_total += n
        pairs_found += 1

    logger.info(f"切片完成：{pairs_found} 对，共 {n_tiles_total} 个 tile → {INTERIM_DATA_DIR}")

    # 尺寸统一
    tile_img_path_list = sorted(input_path.glob("*_y*-x*_img.tif"))
    tile_mask_path_list = sorted(input_path.glob("*_y*-x*_masks.tif"))
    all_tile_path_list = tile_img_path_list + tile_mask_path_list

    padder = Padder(multiple_of=multiple_of)
    changed = padder.unify(all_tile_path_list, output_path=train_path)

    if changed:
        logger.success(f"已完成 padding 并输出到 {train_path}")
    else:
        logger.success(f"尺寸已一致，已复制 {len(all_tile_path_list)} 个文件 → {train_path}")


if __name__ == "__main__":
    app()
