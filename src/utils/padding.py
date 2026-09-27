"""padding.py —— 尺寸统一工具, 纯计算, 不写文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from tifffile import TiffFile


def pad_to_size(img: NDArray, target_h: int, target_w: int) -> NDArray:
    """把 2D (H, W) 图左上角对齐 pad 到 (target_h, target_w), 0 填充。

    尺寸已经一致时原样返回。
    """
    H, W = img.shape[-2], img.shape[-1]
    if H == target_h and W == target_w:
        return img
    pad_h = target_h - H
    pad_w = target_w - W
    return np.pad(
        img,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )


def round_up(h: int, w: int, multiple_of: int) -> tuple[int, int]:
    """把 (h, w) 向上取整到 multiple_of 的倍数; multiple_of<=1 时不变。"""
    if not multiple_of or multiple_of <= 1:
        return h, w
    return (
        int(np.ceil(h / multiple_of) * multiple_of),
        int(np.ceil(w / multiple_of) * multiple_of),
    )


def tif_shape(path: Path) -> tuple[int, int]:
    """只读 TIFF 元数据获取 (H, W), 不加载像素。"""
    with TiffFile(path) as tif:
        shape = tif.series[0].shape
    return shape[-2], shape[-1]


def compute_target_size(
    shapes: Iterable[tuple[int, int]],
    multiple_of: int = 16,
) -> tuple[int, int, bool]:
    """根据一组 (H, W) 计算统一目标尺寸。

    Returns
    -------
    target_h, target_w : int
        统一后的高度和宽度。
    needs_pad : bool
        是否至少有一张图的尺寸与目标尺寸不同。
    """
    shapes = list(shapes)
    if not shapes:
        return 0, 0, False

    h_max = max(h for h, _ in shapes)
    w_max = max(w for _, w in shapes)
    h_max, w_max = round_up(h_max, w_max, multiple_of)

    needs_pad = any((h, w) != (h_max, w_max) for h, w in shapes)
    return h_max, w_max, needs_pad