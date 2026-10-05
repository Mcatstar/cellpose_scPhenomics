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
) -> None:
    pass

if __name__ == "__main__":
    app()