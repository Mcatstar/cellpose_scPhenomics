"""contact.py —— mito (线粒体) 与 ER (内质网) 接触位点的量化计算。

本模块由第三方 deepcontact 代码重构而来, 相对原实现统一了三处语义:

1. 掩码约定统一为「非 0 即前景」。原代码在 calinter_* 里用 255 表示前景,
   在 calContactDist_range_10pix_min 里却用 255 表示背景, 两种相反约定已废弃。
2. 邻域搜索统一使用欧氏距离。原 calinter / calinter_range 用的是切比雪夫窗口,
   与 calinter_dist 的欧氏距离互相矛盾。
3. 图像边界取自数组实际形状, 不再硬编码为 1024。

叠加图统一按 OpenCV 习惯使用 BGR 通道顺序, 可直接交给 cv2.imwrite。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

# Canny 双阈值, 沿用原实现
_CANNY_LOW = 30
_CANNY_HIGH = 100

# 距离直方图的最大距离, 同时是 contact_map 中「无接触」的哨兵值
NO_CONTACT = 11

# 原 deepcontact/visualise.py 的热力图配色, 索引为接触距离, 已转为 BGR
_HEATMAP_BGR = np.array(
    [
        (255, 0, 0),
        (238, 0, 0),
        (205, 0, 0),
        (139, 0, 0),
        (205, 0, 205),
        (238, 0, 238),
        (255, 0, 255),
        (0, 0, 255),
        (0, 0, 238),
        (0, 0, 205),
        (0, 0, 139),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)[:, ::-1]


@dataclass(frozen=True, slots=True)
class ObjectContactResult:
    """按 mito 对象统计的边界接触结果。"""

    n_mito: int
    n_contact: int
    mito_len: int
    contact_len: int
    contact_mask: NDArray[np.bool_]
    overlay: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ContourContactResult:
    """按轮廓像素统计的接触结果, 同时给出接触距离分布。"""

    n_mito: int
    n_contact: int
    mito_len: int
    contact_len: int
    er_len: int
    er_elongation: float
    distance_hist: NDArray[np.int64]
    contact_map: NDArray[np.int32]
    overlay: NDArray[np.uint8]


def overlay_contact(image: NDArray, mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """在原图 (BGR) 上把接触像素标为纯红, 返回新数组, 不修改入参。"""
    vis = np.array(image, dtype=np.uint8, copy=True)
    vis[mask, 0] = 0
    vis[mask, 1] = 0
    vis[mask, 2] = 255
    return vis


def overlay_heatmap(
    contact_map: NDArray,
    image: NDArray,
    alpha: float = 0.5,
) -> NDArray[np.uint8]:
    """按原 deepcontact 配色把接触距离图叠到原图 (BGR) 上。

    contact_map 中 NO_CONTACT 的像素按配色的最后一档 (白色) 处理。
    """
    colors = _HEATMAP_BGR[np.clip(contact_map, 0, NO_CONTACT)]
    return (alpha * colors + (1.0 - alpha) * image).astype(np.uint8)


def contact_objects(
    mito_mask: NDArray,
    er_mask: NDArray,
    image: NDArray,
) -> ObjectContactResult:
    """按 mito 连通对象统计其边界与 ER 的接触。

    等价于原 calContact, 但去掉了从未使用的 px 参数, 并把 n_mito 修正为真实
    对象数 (原实现返回的是含背景的连通分量数, 恒比对象数多 1)。

    Parameters
    ----------
    mito_mask : np.ndarray
        2D 图, 非 0 视为 mito 前景, 内部按 8 连通拆分对象。
    er_mask : np.ndarray
        2D 图, 非 0 视为 ER 前景。
    image : np.ndarray
        (H, W, 3) BGR 底图。

    Returns
    -------
    ObjectContactResult
        contact_mask 为 mito 边界与 ER 的交集。
    """
    mito = np.asarray(mito_mask) > 0
    er = np.asarray(er_mask) > 0
    n_labels, labels = cv2.connectedComponents(mito.astype(np.uint8))

    contact_mask = np.zeros(mito.shape, dtype=bool)
    mito_len = 0
    n_contact = 0
    # Canny 每次只接受单个对象的掩码, 故按对象循环; 循环次数等于 mito 数量而非像素数
    for label in range(1, n_labels):
        boundary = _contour(labels == label)
        mito_len += int(boundary.sum())
        touched = boundary & er
        if touched.any():
            n_contact += 1
            contact_mask |= touched

    return ObjectContactResult(
        n_mito=n_labels - 1,
        n_contact=n_contact,
        mito_len=mito_len,
        contact_len=int(contact_mask.sum()),
        contact_mask=contact_mask,
        overlay=overlay_contact(image, contact_mask),
    )


def contact_contours(
    mito_mask: NDArray,
    er_mask: NDArray,
    image: NDArray,
    px: int = 3,
    inner_px: int | None = None,
) -> ContourContactResult:
    """统计 mito 轮廓像素周围 px 内是否存在 ER 前景。

    等价于原 calinter_dist / calContactDist; 传入 inner_px 后等价于原
    calinter_range (px=inner_px, py=px), 即要求 ER 落在 (inner_px, px] 内。

    与原实现的差异除模块 docstring 的三处统一外, 还有两点: 原 calinter_dist
    返回的 mito 分量数被误写成了 ER 分量数, 此处修正; 原 calinter_range 在发现
    内圈 ER 时会把整个分量的 contact 标记清掉, 此处改为逐像素判定后由像素反推
    分量。

    Parameters
    ----------
    mito_mask, er_mask : np.ndarray
        2D 图, 非 0 为前景。
    image : np.ndarray
        (H, W, 3) BGR 底图。
    px : int
        判定接触的最大欧氏距离, 需 <= 10 才能被热力图配色覆盖。
    inner_px : int or None
        判定接触的最小欧氏距离 (开区间), None 表示不排除任何距离。

    Returns
    -------
    ContourContactResult
        contact_map 取值为接触距离档位, 无接触处为 NO_CONTACT。
    """
    mito = np.asarray(mito_mask) > 0
    er = np.asarray(er_mask) > 0
    mito_contour = _contour(mito)
    er_contour = _contour(er)

    # 每个像素到最近 ER 前景像素的欧氏距离, 与原逐像素 sqrt(p^2 + q^2) 等价
    dist_er = ndimage.distance_transform_edt(~er)
    if inner_px is None:
        on_contour = mito_contour & (dist_er <= px)
    else:
        on_contour = mito_contour & (dist_er > inner_px) & (dist_er <= px)

    contact_map = _contact_map(dist_er, on_contour)
    n_labels, labels = cv2.connectedComponents(mito_contour.astype(np.uint8))
    return ContourContactResult(
        n_mito=n_labels - 1,
        n_contact=int(np.unique(labels[on_contour]).size),
        mito_len=int(mito_contour.sum()),
        contact_len=int(on_contour.sum()),
        er_len=int(er_contour.sum()),
        er_elongation=_er_elongation(er, er_contour),
        distance_hist=_histogram(contact_map, px),
        contact_map=contact_map,
        overlay=overlay_contact(image, on_contour),
    )


def contact_contours_filtered(
    mito_mask: NDArray,
    er_mask: NDArray,
    image: NDArray,
    px: int = 10,
) -> ContourContactResult:
    """带 overlap 排除与双向最近邻约束的 mito 轮廓接触统计。

    等价于原 calContactDist_range_10pix_min 与 calContactDist_er_elongation,
    后者只是多返回一项 ER 伸长率, 此处统一返回。

    保留的原始语义:

    - overlap 为「mito 前景或 mito 轮廓」与 ER 前景的交集, overlap 像素不参与配对;
    - ER 轮廓像素只与距离不超过自身最近 mito 轮廓距离 +1 的 mito 轮廓像素配对;
    - overlap 且属于 mito 轮廓的像素在 contact_map 中记 0, 并计入 distance_hist[0]。

    同时修正了原实现的两处自相矛盾:

    - 原 distmap[0] 是覆盖写而非累加, 会丢掉「配对命中且距离为 0」的像素, 使
      distmap 与 contact_map 对不上; 此处按 contact_map 生成直方图;
    - 原 contact_flag 只统计配对命中, 不含仅因 overlap 接触的分量, 而 contact_len
      却包含这些像素; 此处 n_contact 与 contact_len 统一按 contact_map 判定。

    Parameters
    ----------
    mito_mask, er_mask : np.ndarray
        2D 图, 非 0 为前景。
    image : np.ndarray
        (H, W, 3) BGR 底图。
    px : int
        配对的最大欧氏距离, 需 <= 10 才能被热力图配色覆盖。

    Returns
    -------
    ContourContactResult
    """
    mito = np.asarray(mito_mask) > 0
    er = np.asarray(er_mask) > 0
    mito_contour = _contour(mito)
    er_contour = _contour(er)
    overlap = (mito | mito_contour) & er

    er_free = er_contour & ~overlap
    mito_free = mito_contour & ~overlap
    dist_mito = ndimage.distance_transform_edt(~mito_free)
    dist_er = ndimage.distance_transform_edt(~er_free)

    # 剪枝: 距离超过 px 的像素不可能参与任何配对, 去掉后显著减小配对规模
    mito_pts = np.argwhere(mito_free & (dist_er <= px))
    er_pts = np.argwhere(er_free & (dist_mito <= px))

    best = np.full(len(mito_pts), np.inf)
    if len(mito_pts) and len(er_pts):
        # 每个 ER 像素只接受距离不超过「自身最近 mito 距离 + 1」的 mito 像素
        er_radius = dist_mito[er_pts[:, 0], er_pts[:, 1]] + 1.0
        pairs = cKDTree(mito_pts).sparse_distance_matrix(
            cKDTree(er_pts), px, output_type="coo_matrix"
        )
        valid = pairs.data <= er_radius[pairs.col]
        np.minimum.at(best, pairs.row[valid], pairs.data[valid])

    found = np.isfinite(best)
    contact_map = np.full(mito.shape, NO_CONTACT, dtype=np.int32)
    contact_map[mito_pts[found, 0], mito_pts[found, 1]] = _step(best[found])
    contact_map[overlap & mito_contour] = 0
    in_contact = contact_map < NO_CONTACT

    n_labels, labels = cv2.connectedComponents(mito_contour.astype(np.uint8))
    return ContourContactResult(
        n_mito=n_labels - 1,
        n_contact=int(np.unique(labels[in_contact]).size),
        mito_len=int(mito_contour.sum()),
        contact_len=int(in_contact.sum()),
        er_len=int(er_contour.sum()),
        er_elongation=_er_elongation(er, er_contour),
        distance_hist=_histogram(contact_map, px),
        contact_map=contact_map,
        overlay=overlay_heatmap(contact_map, image),
    )


def _contour(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Canny 提取二值掩码的 1 像素轮廓。"""
    return cv2.Canny(mask.astype(np.uint8) * 255, _CANNY_LOW, _CANNY_HIGH) > 0


def _step(distances: NDArray) -> NDArray[np.int64]:
    """把欧氏距离映射为直方图档位: 0 保持 0, 正距离向上取整。

    容忍 1e-3 的浮点误差, 与原实现的 int(dist - 0.001) + 1 一致。
    """
    return (np.floor(distances - 1e-3) + 1).astype(np.int64)


def _histogram(contact_map: NDArray, px: int) -> NDArray[np.int64]:
    """按距离档位统计接触像素数, 长度为 px + 1, 与 contact_map 恒一致。"""
    counts = np.bincount(contact_map.ravel(), minlength=NO_CONTACT + 1)
    counts[NO_CONTACT] = 0
    return counts[: px + 1].astype(np.int64)


def _contact_map(
    distances: NDArray,
    in_contact: NDArray[np.bool_],
) -> NDArray[np.int32]:
    """接触像素写入距离档位, 其余写 NO_CONTACT。"""
    return np.where(in_contact, _step(distances), NO_CONTACT).astype(np.int32)


def _er_elongation(er_mask: NDArray[np.bool_], er_contour: NDArray[np.bool_]) -> float:
    """ER 各对象的平均伸长率 perimeter^2 / (4 * pi * area)。

    原实现逐对象调用 Canny 求周长, 这里复用全局轮廓再按标号分箱; 因为连通分量
    之间互不相邻, 全局轮廓就是各对象轮廓的并集。
    """
    n_labels, labels = cv2.connectedComponents(er_mask.astype(np.uint8))
    if n_labels <= 1:
        return 0.0
    perimeter = np.bincount(labels[er_contour].ravel(), minlength=n_labels)[1:]
    area = np.bincount(labels.ravel(), minlength=n_labels)[1:]
    return float((perimeter**2 / (4 * np.pi * area)).mean())


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "用法: python -m deepcontact.contact <mito.npy> <er.npy> <image.npy>"
        )

    mito_pred = np.load(sys.argv[1])
    er_pred = np.load(sys.argv[2])
    base_image = np.load(sys.argv[3])
    result = contact_contours_filtered(mito_pred, er_pred, base_image)
    cv2.imwrite("deepcontact_debug.png", result.overlay)
