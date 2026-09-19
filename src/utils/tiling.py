"""
utils/tiling.py —— 2D 大图滑窗切片器

- 约定：图像为 (H, W) 或 (C, H, W), 空间维永远是最后两维
- 只做 XY 平面滑窗, 输出尺寸严格 = crop_size × crop_size
- image 与 mask 共用同一套 (y, x) 坐标, 一一对应
- 不做任何归一化
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from tifffile import imread, imwrite


class Tiler:
    """2D 滑窗切片器(支持 (H, W) 与 (C, H, W))"""

    def __init__(
        self,
        crop_size: int = 512,
        stride: Optional[int] = None,
        min_mask_fraction: float = 0.0,
    ):
        self.crop_size = crop_size
        self.stride = stride if stride is not None else crop_size
        self.min_mask_fraction = min_mask_fraction

    # coords
    def tile_coords(self, length: int) -> list[int]:
        """沿某一维度切片的起始坐标, 末尾自动对齐边界"""
        c, s = self.crop_size, self.stride
        if length < c:
            return []
        coords = list(range(0, length - c + 1, s))
        if coords[-1] != length - c:
            coords.append(length - c)
        return coords

    # iter
    def iter_tiles(
        self, img: NDArray, mask: NDArray
    ) -> Iterator[tuple[int, int, NDArray, NDArray]]:
        """以相同坐标同时切分 image 和 mask, yield (y, x, img_tile, mask_tile)

        约定：空间维为最后两维；前置维 (如通道) 用 ... 原样保留
        - 2D  (H, W)    → tile (crop, crop)
        - 3D  (C, H, W) → tile (C, crop, crop)
        """
        c = self.crop_size
        H, W = img.shape[-2], img.shape[-1]
        for y in self.tile_coords(H):
            for x in self.tile_coords(W):
                img_tile = img[..., y : y + c, x : x + c]
                mask_tile = mask[..., y : y + c, x : x + c]
                if (
                    self.min_mask_fraction > 0
                    and (mask_tile > 0).mean() < self.min_mask_fraction
                ):
                    continue
                yield y, x, img_tile, mask_tile

    # save
    def tile(
        self,
        img_path: Path,
        mask_path: Path,
        output_path: Path,
        base_name: str,
        img_suffix: str = "_img.tif",
        mask_suffix: str = "_masks.tif",
    ) -> int:
        """对一对 (image, mask) 切片并保存, 返回切片数

        - 正常：滑窗切成若干 crop_size * crop_size 的 tile
        - 大图 < crop_size: 不切, 整张作为唯一的 y0-x0 tile 写出, 
          交由后续 padding 统一尺寸
        """
        img = imread(img_path)
        mask = imread(mask_path)

        # 只比较空间维
        if img.shape[-2:] != mask.shape[-2:]:
            logger.warning(
                f"形状不匹配，跳过：{img_path.name} {img.shape} vs "
                f"{mask_path.name} {mask.shape}"
            )
            raise ValueError

        H, W = img.shape[-2:]
        output_path.mkdir(parents=True, exist_ok=True)

        # ---- 小图：整张写出，不切 ----
        if H < self.crop_size or W < self.crop_size:
            if (
                self.min_mask_fraction > 0
                and (mask > 0).mean() < self.min_mask_fraction
            ):
                logger.info(
                    f"{img_path.name}: mask 有效像素比例不足 "
                    f"{self.min_mask_fraction}，跳过"
                )
                return 0

            name = f"{base_name}_y0-x0"
            imwrite(output_path / f"{name}{img_suffix}", img)
            imwrite(output_path / f"{name}{mask_suffix}", mask)
            logger.info(
                f"{img_path.name}: ({H}, {W}) < crop_size="
                f"{self.crop_size}，整张写出为 y0-x0"
            )
            return 1

        # ---- 正常：滑窗切 ----
        n = 0
        for y, x, img_tile, mask_tile in self.iter_tiles(img, mask):
            name = f"{base_name}_y{y}-x{x}"
            imwrite(output_path / f"{name}{img_suffix}", img_tile)
            imwrite(output_path / f"{name}{mask_suffix}", mask_tile)
            n += 1
        return n


    # find_pairs
    @staticmethod
    def find_pairs(
        input_path: Path,
        img_suffix: str,
        mask_suffix: str,
    ) -> list[tuple[Path, Path]]:
        """在目录中查找成对的 (image, mask), 缺失 mask 的跳过"""
        pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(input_path.glob(f"*{img_suffix}")):
            base = img_path.name[: -len(img_suffix)]
            mask_path = input_path / f"{base}{mask_suffix}"
            if not mask_path.exists():
                logger.warning(f"缺少 mask, 跳过: {img_path.name}")
                continue
            pairs.append((img_path, mask_path))
        return pairs