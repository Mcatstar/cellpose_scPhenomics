"""
utils/tiling.py —— 2D 大图滑窗切片器。

- 只做 XY 平面滑窗，输出尺寸严格 = crop_size * crop_size。
- image 与 mask 共用同一套 (y, x) 坐标，一一对应。
- 不做任何归一化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from tifffile import imread, imwrite


class Tiler:
    """2D 滑窗切片器。"""

    def __init__(
        self,
        crop_size: int = 512,
        stride: Optional[int] = None,
        min_mask_fraction: float = 0.0,
    ):
        self.crop_size = crop_size
        self.stride = stride if stride is not None else crop_size
        self.min_mask_fraction = min_mask_fraction

    # ---------------------------------------------------------------- coords
    def tile_coords(self, length: int) -> list[int]:
        """沿某一维度切片的起始坐标，末尾自动对齐边界。"""
        c, s = self.crop_size, self.stride
        if length < c:
            return []
        coords = list(range(0, length - c + 1, s))
        if coords[-1] != length - c:
            coords.append(length - c)
        return coords

    # ------------------------------------------------------------------ iter
    def iter_tiles(
        self, img: NDArray, mask: NDArray
    ) -> Iterator[tuple[int, int, NDArray, NDArray]]:
        """以相同坐标同时切分 image 和 mask，yield (y, x, img_tile, mask_tile)。"""
        c = self.crop_size
        for y in self.tile_coords(img.shape[0]):
            for x in self.tile_coords(img.shape[1]):
                img_tile = img[y : y + c, x : x + c]
                mask_tile = mask[y : y + c, x : x + c]
                if (
                    self.min_mask_fraction > 0
                    and (mask_tile > 0).mean() < self.min_mask_fraction
                ):
                    continue
                yield y, x, img_tile, mask_tile

    # ------------------------------------------------------------------ save
    def save_pair(
        self,
        img_path: Path,
        mask_path: Path,
        output_path: Path,
        base_name: str,
        img_suffix: str = "_img.tif",
        mask_suffix: str = "_masks.tif",
    ) -> int:
        """对一对 (image, mask) 切片并保存，返回切片数。"""
        img = imread(img_path)
        mask = imread(mask_path)

        if img.shape[:2] != mask.shape[:2]:
            logger.warning(
                f"形状不匹配，跳过：{img_path.name} {img.shape} vs "
                f"{mask_path.name} {mask.shape}"
            )
            return 0

        output_path.mkdir(parents=True, exist_ok=True)
        n = 0
        for y, x, img_tile, mask_tile in self.iter_tiles(img, mask):
            name = f"{base_name}_y{y}-x{x}"
            imwrite(output_path / f"{name}{img_suffix}", img_tile)
            imwrite(output_path / f"{name}{mask_suffix}", mask_tile)
            n += 1
        return n

    # ------------------------------------------------------------ find_pairs
    @staticmethod
    def find_pairs(
        input_path: Path,
        img_suffix: str,
        mask_suffix: str,
    ) -> list[tuple[Path, Path]]:
        """在目录中查找成对的 (image, mask)，缺失 mask 的跳过。"""
        pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(input_path.glob(f"*{img_suffix}")):
            base = img_path.name[: -len(img_suffix)]
            mask_path = input_path / f"{base}{mask_suffix}"
            if not mask_path.exists():
                logger.warning(f"缺少 mask，跳过：{img_path.name}")
                continue
            pairs.append((img_path, mask_path))
        return pairs