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


def best_grid(n: int, h: int, w: int, align: int = 16):
    """确定拼接大数组的尺寸
    n: 图数量
    h, w: 单元格高、宽
    align: 画布尺寸向上取整到 align 的倍数
    返回: (R, C, canvas_h, canvas_w, ratio)
    R	行数
    C	列数
    ch	对齐后的画布高度 = R * h 向上取整到 align 的倍数
    cw	对齐后的画布宽度 = C * w 向上取整到 align 的倍数
    ratio	max(ch,cw) / min(ch,cw)，越接近 1 越却近于正方形
    """
    Rs = np.arange(1, n + 1)                    # 所有候选行数
    Cs = np.ceil(n / Rs).astype(int)            # 对应的列数
    chs = Rs * h                                # 画布高
    cws = Cs * w                                # 画布宽
    ratios = np.maximum(chs, cws) / np.minimum(chs, cws)

    i = np.argmin(ratios)                       # 比例最接近 1 的索引
    R, C = int(Rs[i]), int(Cs[i])
    ch, cw = int(chs[i]), int(cws[i])

    ch = int(np.ceil(ch / align) * align)
    cw = int(np.ceil(cw / align) * align)
    return R, C, ch, cw, max(ch, cw) / min(ch, cw)


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
    number_of_boxes = len(box_records)
    H_max = box_records.loc[:, "crop_h"].max()
    W_max = box_records.loc[:, "crop_w"].max()
    R, C, ch, cw, ratio = best_grid(number_of_boxes, H_max, W_max)
    logger.info(f"""Number of boxes: {number_of_boxes}, 
                max crop_h: {H_max}, max crop_w: {W_max}, 
                best grid: {R}x{C}, canvas size: {ch}x{cw}, ratio: {ratio:.2f}""")

    n_ch = len(dict_ch)
    N = len(box_records)
    big_img = np.zeros((n_ch, ch, cw), dtype=np.float32)
    big_masks = np.zeros((n_ch, ch, cw), dtype=np.uint16)
    for i, (_, row) in tqdm(enumerate(box_records.iterrows()), total=N, desc="Padding Generation"):
        img_norm_path = INTERIM_DATA_DIR / row["name"].replace(".tif", "_img_norm.tif")
        masks_path = INTERIM_DATA_DIR / row["name"].replace(".tif", "_masks.tif")
        img_norm = imread(img_norm_path)
        masks = imread(masks_path)
        r, c = divmod(i, C)
        y_off, x_off = r * H_max, c * W_max
        h_i, w_i = img_norm.shape[-2], img_norm.shape[-1]
        if img_norm.ndim == 2:
            img_norm = img_norm[np.newaxis, ...]    # (1, H, W)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        for ch_idx, (_, (src_img_ch, src_mask_ch)) in enumerate(dict_ch.items()):
            big_img[ch_idx, y_off:y_off+h_i, x_off:x_off+w_i]   = img_norm[src_img_ch]
            big_masks[ch_idx, y_off:y_off+h_i, x_off:x_off+w_i] = masks[src_mask_ch]
        logger.info(f"Successfully normlizing the No.{i+1} image in total {N}")
    big_img_path = output_path / "train" / "big_img.tif"
    big_masks_path = output_path / "train" / "big_masks.tif"
    imwrite(big_img_path, big_img)
    imwrite(big_masks_path, big_masks)
    logger.success("Features generation complete.")
    # -----------------------------------------

if __name__ == "__main__":
    app()
