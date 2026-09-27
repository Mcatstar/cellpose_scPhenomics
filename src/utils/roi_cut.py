"""roi.py —— ROI 裁剪与归一化, 纯计算, 不写文件。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def roi_cut(
    img: NDArray,
    masks: NDArray,
    roi: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, tuple[int, int, int, int]]:
    """把 img / masks 裁剪到 roi 的包围盒, 并把 roi 外的像素置 0。

    Parameters
    ----------
    img : np.ndarray
        2D 灰度图 (H, W)。
    masks : np.ndarray
        2D 掩码 (H, W)。
    roi : np.ndarray or None
        2D 区域掩码, roi>0 视为有效区域。None 表示全图有效。

    Returns
    -------
    img_crop, masks_crop, roi_crop : np.ndarray
        裁剪后的图、掩码和区域掩码。
    box : tuple of int
        (y0, y1, x0, x1), 便于逆向还原坐标。
    """
    if roi is None:
        roi_crop = np.ones_like(img, dtype=bool)
        return img.copy(), masks.copy(), roi_crop, (
            0, img.shape[-2], 0, img.shape[-1],
        )

    ys, xs = np.where(roi > 0)
    if len(ys) == 0:
        return (
            np.zeros_like(img),
            np.zeros_like(masks),
            np.zeros_like(img, dtype=bool),
            (0, 0, 0, 0),
        )

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    roi_crop = roi[y0:y1, x0:x1]
    keep = roi_crop > 0

    img_crop = np.where(keep, img[y0:y1, x0:x1], 0)
    masks_crop = np.where(keep, masks[y0:y1, x0:x1], 0)
    return img_crop, masks_crop, roi_crop, (y0, y1, x0, x1)


def normalize_roi(
    img: NDArray,
    roi: NDArray,
    lower: int = 1,
    upper: int = 99,
) -> NDArray:
    """ROI 内按 1-99 百分位归一化到 [0, 1], ROI 外置 0。"""
    img = img.astype(np.float32)
    values = img[roi > 0]
    if values.size == 0:
        return np.zeros_like(img)

    low, high = np.percentile(values, [lower, upper])
    high = max(high, low + 1e-6)
    out = np.clip((img - low) / (high - low), 0.0, 1.0)
    out[roi == 0] = 0
    return out