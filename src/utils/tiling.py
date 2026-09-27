"""tiling.py —— 2D 滑窗切片, 纯计算, 不写文件。"""

from __future__ import annotations

from typing import Iterator

from numpy.typing import NDArray


def tile_coords(length: int, crop_size: int, stride: int) -> list[int]:
    """沿某一维度切片的起始坐标, 末尾自动对齐边界。

    长度小于 crop_size 时返回空列表。
    """
    if length < crop_size:
        return []
    coords = list(range(0, length - crop_size + 1, stride))
    if coords[-1] != length - crop_size:
        coords.append(length - crop_size)
    return coords


def iter_tiles(
    img: NDArray,
    mask: NDArray,
    crop_size: int,
    stride: int,
    min_mask_fraction: float = 0.0,
) -> Iterator[tuple[int, int, NDArray, NDArray]]:
    """以相同坐标同时切分 img 和 mask, yield (y, x, img_tile, mask_tile)。

    min_mask_fraction > 0 时, mask 中有效像素比例低于该阈值的 tile 会被跳过。
    """
    H, W = img.shape[-2], img.shape[-1]
    for y in tile_coords(H, crop_size, stride):
        for x in tile_coords(W, crop_size, stride):
            img_tile = img[y : y + crop_size, x : x + crop_size]
            mask_tile = mask[y : y + crop_size, x : x + crop_size]
            if min_mask_fraction > 0 and (mask_tile > 0).mean() < min_mask_fraction:
                continue
            yield y, x, img_tile, mask_tile