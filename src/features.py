"""
features.py —— 把 INTERIM_DATA_DIR 里已标注的 *_img.tif / *_mito_labels.tif /
*_ld_labels.tif 转移到 PROCESSED_DATA_DIR

规则：
1. 已标注数据（两个标签都存在）：
   - 根据 img 的形状裁剪标签（左上角对齐，因为 dataset 阶段切掉了黑边）
   - 80/20 随机划分到 train 和 test, 标签文件重命名为 *_masks.tif
2. 仅推理数据（缺少任一标签）：
   - 全部放入 infer, 不裁剪
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from tifffile import imread, imwrite
from tqdm import tqdm

from src.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    logger.info("Generating features from dataset...")
    logger.info(f"input path: {input_path}")
    logger.info(f"output path: {output_path}")

    train_dir = output_path / "train"
    test_dir = output_path / "test"
    infer_dir = output_path / "infer"

    metadata = pd.read_csv(input_path / "metadata.csv")
    logger.info(f"metadata 记录数: {len(metadata)}")

    # ---------- 分类 ----------
    labeled, inference_only = [], []
    for img_name in metadata["img_name"]:
        img_path = input_path / img_name
        if not img_path.exists():
            logger.warning(f"图像不存在, 跳过: {img_name}")
            continue

        base = img_name[: -len("_img.tif")]
        has_mito = (EXTERNAL_DATA_DIR / f"{base}_mito_labels.tif").exists()
        has_ld = (EXTERNAL_DATA_DIR / f"{base}_ld_labels.tif").exists()
        (labeled if has_mito and has_ld else inference_only).append(img_path)

    logger.info(
        f"已标注样本: {len(labeled)}, 仅推理样本: {len(inference_only)}"
    )

    # ---------- 已标注：80/20 划分 + 裁剪 + 重命名为 masks ----------
    train_imgs, test_imgs = (
        train_test_split(labeled, train_size=train_ratio, random_state=seed)
        if labeled else ([], [])
    )

    for img_list, target_dir in (
        (train_imgs, train_dir),
        (test_imgs, test_dir),
    ):
        for img_path in tqdm(img_list, desc=f"已标注 → {target_dir.name}"):
            base = img_path.name[: -len("_img.tif")]
            h, w = imread(img_path).shape[:2]

            shutil.copy2(img_path, target_dir / img_path.name)
            for tag in ("_mito", "_ld"):
                src = EXTERNAL_DATA_DIR / f"{base}{tag}_labels.tif"
                dst = target_dir / src.name.replace("_labels", "_masks")
                imwrite(dst, imread(src)[:h, :w])

    # ---------- 仅推理：原样转移到 infer ----------
    for img_path in tqdm(inference_only, desc="仅推理 → infer"):
        base = img_path.name[: -len("_img.tif")]
        shutil.copy2(img_path, infer_dir / img_path.name)
        for tag in ("_mito", "_ld"):
            src = EXTERNAL_DATA_DIR / f"{base}{tag}_labels.tif"
            if src.exists():
                shutil.copy2(src, infer_dir / src.name)

    logger.success(
        f"完成: train={len(train_imgs)}, test={len(test_imgs)}, "
        f"infer={len(inference_only)} → {output_path}"
    )


if __name__ == "__main__":
    app()