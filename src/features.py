from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tifffile import imread, imwrite
from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def normalize_roi(img_crop: NDArray, roi_crop: NDArray, lower=1, upper=99):
    """ROI区域内的归一化
    img_crop: ROI 裁切后的图像, (H, W) 或 (C, H, W)
    roi_crop: 同尺寸 ROI 掩码，(H, W), roi>0 为有效区域
    返回: 归一化后的图像, ROI 外为 0
    """
    img = img_crop.astype(np.float32)

    if img.ndim == 2:
        vals = img[roi_crop > 0]
        if vals.size == 0:
            return np.zeros_like(img)
        low, high = np.percentile(vals, [lower, upper])
        high = max(high, low + 1e-6)
        out = np.clip((img - low) / (high - low), 0, 1)
        out[roi_crop == 0] = 0
        return out
    else:
        out = np.zeros_like(img)
        for c in range(img.shape[0]):
            vals = img[c][roi_crop > 0]
            if vals.size == 0:
                continue
            low, high = np.percentile(vals, [lower, upper])
            high = max(high, low + 1e-6)
            out[c] = np.clip((img[c] - low) / (high - low), 0, 1)
            out[c][roi_crop == 0] = 0
        return out


def pad_to_size(img_2d: NDArray, target_h: int, target_w: int) -> NDArray:
    """把 2D (H, W) padding 到 (1, target_h, target_w)，左上角对齐"""
    out = np.zeros((1, target_h, target_w), dtype=img_2d.dtype)
    out[0, :img_2d.shape[0], :img_2d.shape[1]] = img_2d
    return out


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    # 定义频道字典, 含义[在_img.tif的频道, 在_masks.tif的频道]
    dict_ch = {
        "mito": [0, 0],
        "lipid": [1, 1]
    }

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    # Load image-------------------------------------
    tmp_list: list[Path] = list(Path(input_path).iterdir())
    img_list: list[Path] = []
    for tmp in tmp_list:
        if re.search('_img.tif', str(tmp)):
            img_list.append(tmp) # img_list have full path, not name, e.g. "/path/to/img.tif"
            
    logger.info("images found:\n"+"\n".join(map(lambda x: str(x.name), img_list)))
    logger.info(f"Number of images found: {len(img_list)}")
    # img_index = int(input('Select image: '))
    for img_index in tqdm(range(len(img_list)), total=len(img_list), desc="Features Generation"):
        # Select image and load it
        img_crop_path = input_path / f"{img_list[img_index].name}"
        roi_crop_path = input_path / f"{img_list[img_index].name.replace("_img.tif", "_roi_crop.tif")}"
        img_crop = imread(img_crop_path)
        roi_crop = imread(roi_crop_path)
        img_norm = normalize_roi(img_crop, roi_crop)
        img_norm_path = INTERIM_DATA_DIR / f"{img_list[img_index].name.replace("_img.tif", "_img_norm.tif")}"
        imwrite(img_norm_path, img_norm)
        logger.info(f"Successfully normlizing the No.{img_index+1} image in total {len(img_list)}")

    box_records_path = input_path / "box_records.csv"
    box_records = pd.read_csv(box_records_path)
    H_max = box_records.loc[:, "crop_h"].max()
    W_max = box_records.loc[:, "crop_w"].max()
    H_max = int(np.ceil(H_max / 16) * 16)
    W_max = int(np.ceil(W_max / 16) * 16)
    N = len(box_records)
    
    for i, (_, row) in tqdm(enumerate(box_records.iterrows()), total=N, desc="Padding Generation"):
        img_norm_path = INTERIM_DATA_DIR / row["name"].replace(".tif", "_img_norm.tif")
        masks_path = INTERIM_DATA_DIR / row["name"].replace(".tif", "_masks.tif")
        img_norm = imread(img_norm_path)
        masks = imread(masks_path)
        for cat, (src_img_ch, src_mask_ch) in dict_ch.items():
            img_2d  = img_norm[src_img_ch]        # (H_i, W_i)
            mask_2d = masks[src_mask_ch]     # (H_i, W_i)

            img_pad  = pad_to_size(img_2d,  H_max, W_max)   # (1, H_max, W_max)
            mask_pad = pad_to_size(mask_2d, H_max, W_max)   # (1, H_max, W_max)
            img_i_path = output_path / "train" / f"{cat}" / row["name"].replace(".tif", f"_img.tif")
            masks_i_path = output_path / "train" / f"{cat}" / row["name"].replace(".tif", f"_masks.tif")
            imwrite(img_i_path, img_pad.astype(np.float32))
            imwrite(masks_i_path, mask_pad.astype(np.uint16))
        logger.info(f"Successfully padding the No.{i+1} image in total {N}")
    logger.success("Features generation complete.")
    # -----------------------------------------

if __name__ == "__main__":
    app()
