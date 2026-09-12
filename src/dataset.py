from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as nd 
from tifffile import imread, imwrite
import napari
from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, IMAGES_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def roi_cut(img: NDArray, masks: NDArray, roi: NDArray | None=None):
    """
    img: 原始图像，形状 (H, W) 或 (C, H, W)
    masks: 标记掩码或识别目标，形状 (H, W) 或 (C, H, W)
    roi: 工作区域或识别区域掩码标记，形状 (H, W), roi>0 为识别区域
    返回：(img_crop, masks_crop, box)
    """
    if roi is None:
        # 全图均为识别区域，不裁剪，不置零
        img_crop = img.copy()
        masks_crop = masks.copy()
        box = (0, img.shape[-2], 0, img.shape[-1])
        return img_crop, masks_crop, box

    ys, xs = np.where(roi > 0)
    if len(ys) == 0:
        # 无识别区域，全图均不识别
        img_crop = np.zeros_like(img)
        masks_crop = np.zeros_like(masks)
        box = (0, 0, 0, 0)
        return img_crop, masks_crop, box

    # 确定裁切范围
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    logger.info(f"box: y={y0}:{y1}, x={x0}:{x1}")
    roi_crop = roi[y0:y1, x0:x1]
    
    # 裁切 置零
    if img.ndim == 2:
        img_crop = img[y0:y1, x0:x1].copy()
        img_crop[roi_crop == 0] = 0
    else:
        img_crop = np.stack(
            [np.where(roi_crop == 0, 0, img[c, y0:y1, x0:x1]).astype(img.dtype)
            for c in range(img.shape[0])], 
            axis=0
        )
    if masks.ndim == 2:
        masks_crop = masks[y0:y1, x0:x1].copy()
        masks_crop[roi_crop == 0] = 0
    else:
        masks_crop = np.stack(
            [np.where(roi_crop == 0, 0, masks[c, y0:y1, x0:x1]).astype(masks.dtype)
            for c in range(masks.shape[0])], 
            axis=0
        )
    return img_crop, masks_crop, (y0, y1, x0, x1)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = IMAGES_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    # Load image-------------------------------------
    tmp_list: list[Path] = list(Path(input_path).iterdir())
    img_list: list[Path] = []
    for tmp in tmp_list:
        if re.search('.tif', str(tmp)):
            img_list.append(tmp) # img_list have full path, not name, e.g. "/path/to/img.tif"
            
    logger.info("images found:\n"+"\n".join(map(lambda x: str(x.name), img_list)))
    logger.info(f"Number of images found: {len(img_list)}")
    # img_index = int(input('Select image: '))
    for img_index in tqdm(range(len(img_list)), total=len(img_list), desc="Dataset Generation"):
        # Select image and load it
        logger.info(f"Selected image: {img_list[img_index].name}")
        img_path: Path = img_list[img_index] # img_list have full path
        logger.info(f"Image path: {img_path}")
        img = imread(img_path)
        logger.info(img.shape)

        # Cutting image----------------------------------
        # Mask work area, _masks.tif and _roi.tif in /data/external
        masks_path = EXTERNAL_DATA_DIR / f"{img_list[img_index].name}_masks.tif"
        masks = imread(masks_path) if masks_path.exists() else np.zeros_like(img.shape)
        roi_path = EXTERNAL_DATA_DIR / f"{img_list[img_index].name}_roi.tif"
        roi = imread(roi_path) if roi_path.exists() else None
        img_crop, masks_crop, _ = roi_cut(img, masks, roi)

        imwrite(INTERIM_DATA_DIR / f"{img_list[img_index].name.replace(".tif", "_img.tif")}", np.uint16(img_crop))
        imwrite(INTERIM_DATA_DIR / f"{img_list[img_index].name.replace(".tif", "_masks.tif")}", np.uint16(masks_crop))


    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
