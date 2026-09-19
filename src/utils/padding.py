"""
utils/padding.py —— 统一尺寸填充器

- 所有图尺寸一致且已对齐时 no-op, 不改任何文件
- 尺寸不一致时,把全部图 pad 到最大尺寸(左上角对齐, 0 填充)
- 支持 2D (H, W) 和多通道 (..., H, W)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from tifffile import TiffFile, imread, imwrite
from tqdm import tqdm


class Padder:
    """统一尺寸填充器"""

    def __init__(self, multiple_of: int = 16):
        """
        :param multiple_of: 目标边长向上取整到该值的倍数 (0 表示不取整)
        """
        self.multiple_of = multiple_of

    # pad
    @staticmethod
    def pad_to_size(img: NDArray, target_h: int, target_w: int) -> NDArray:
        """把(..., H, W)左上角对齐 pad 到 (target_h, target_w), 0 填充"""
        H, W = img.shape[-2], img.shape[-1]
        if H == target_h and W == target_w:
            return img
        pad_h = target_h - H
        pad_w = target_w - W
        if img.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        else:
            pad_width = ((0, 0),) * (img.ndim - 2) + ((0, pad_h), (0, pad_w))
        return np.pad(img, pad_width, mode="constant", constant_values=0)

    def pad_pair(
        self,
        img: NDArray,
        mask: NDArray,
        target_h: int,
        target_w: int,
    ) -> tuple[NDArray, NDArray]:
        """同尺寸 pad 一对(image, mask)"""
        return (
            self.pad_to_size(img, target_h, target_w),
            self.pad_to_size(mask, target_h, target_w),
        )

    # shape
    def round_up(self, h: int, w: int) -> tuple[int, int]:
        """把(h, w)向上取整到 multiple_of 的倍数"""
        m = self.multiple_of
        if not m or m <= 1:
            return h, w
        return int(np.ceil(h / m) * m), int(np.ceil(w / m) * m)

    @staticmethod
    def shape(path: Path) -> tuple[int, int]:
        """只读 TIFF 元数据获取 (H, W), 不加载像素"""
        with TiffFile(path) as tif:
            s = tif.series[0].shape
        return s[-2], s[-1]

    # unify
    def unify(
        self,
        paths: Iterable[Path],
        output_path: Optional[Path] = None,
    ) -> bool:
        """检查一组 tif 的尺寸; 不一致则全部 pad 到最大尺寸

        :param paths: 待检查的 tif 路径集合
        :param output_path: None 表示原地覆盖; 否则写到该目录(同名)
        :return: True 表示发生了 pad; False 表示尺寸已一致且无需改动
        """
        paths = list(paths)
        if not paths:
            logger.warning("unify_dir 收到空路径列表")
            return False

        # ---- 读取元数据(只读头部,很快) ----
        shapes: dict[Path, tuple[int, int]] = {}
        for p in tqdm(paths, desc="读取尺寸", unit=" file", leave=False):
            shapes[p] = self.shape(p)
        uniq = set(shapes.values())

        # ---- 决定目标尺寸 ----
        if len(uniq) == 1:
            (h, w), = uniq
            rh, rw = self.round_up(h, w)
            if (h, w) == (rh, rw) and output_path is None:
                logger.success(f"所有图尺寸一致 ({h}, {w}),无需 padding")
                return False
            h_max, w_max = rh, rw
        else:
            h_max = max(h for h, _ in shapes.values())
            w_max = max(w for _, w in shapes.values())
            h_max, w_max = self.round_up(h_max, w_max)
            logger.info(
                f"发现 {len(uniq)} 种尺寸, 统一 pad 到 ({h_max}, {w_max})"
            )

        # ---- 逐个处理 ----
        n_padded = 0
        for path, (h, w) in tqdm(
            shapes.items(),
            desc="Padding",
            unit=" file",
            total=len(shapes),
        ):
            target_path = path if output_path is None else output_path / path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if (h, w) == (h_max, w_max):
                # 已对齐：只在需要输出到新目录时复制
                if output_path is not None:
                    imwrite(target_path, imread(path))
                continue

            arr = imread(path)
            arr = self.pad_to_size(arr, h_max, w_max)
            imwrite(target_path, arr)
            n_padded += 1

        return n_padded > 0