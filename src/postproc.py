from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import ndimage as nd
from skimage.measure import regionprops
from tifffile import imread, imwrite
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, PRED_DATA_DIR

app = typer.Typer()


# ---- 通道定义：图像/掩码为 (C, H, W)，第 0 层 mito，第 1 层 lipid ----
MITO_CHANNEL = 0
LIPID_CHANNEL = 1

OBJECT_COLUMNS = [
    "roi_id", "organelle", "area", "perimeter", "eccentricity", "solidity",
    "axis_major_length", "axis_minor_length", "feret_diameter_max",
    "centroid_y", "centroid_x", "mean_intensity",
    "dist_to_other_min", "dist_to_other_max", "dist_to_other_mean",
]
SUMMARY_COLUMNS = [
    "image", "mito_count", "lipid_count", "mito_total_area", "lipid_total_area",
    "overlap_area", "overlap_ratio_mito", "overlap_ratio_lipid",
]


def extract_features(mask: np.ndarray, intensity_img: np.ndarray,
                     organelle: str, cross_edt: np.ndarray | None = None
                     ) -> list[dict]:
    """对一个通道的掩码做 regionprops, 返回对象特征列表。

    cross_edt: 到异类细胞器（mito 看 lipid，lipid 看 mito）的欧几里得距离图。
               传入后会额外计算每个对象到异类细胞器的最小/最大/平均距离。
    """
    rows = [
        {
            "roi_id": p.label,
            "organelle": organelle,
            "area": p.area,
            "perimeter": p.perimeter,
            "eccentricity": p.eccentricity,
            "solidity": p.solidity,
            "axis_major_length": p.axis_major_length,
            "axis_minor_length": p.axis_minor_length,
            "feret_diameter_max": p.feret_diameter_max,
            "centroid_y": p.centroid[0],
            "centroid_x": p.centroid[1],
            "mean_intensity": p.mean_intensity,
        }
        for p in regionprops(mask, intensity_image=intensity_img)
    ]

    if cross_edt is not None:
        for row, p in zip(rows, regionprops(mask, intensity_image=cross_edt)):
            row["dist_to_other_min"] = p.intensity_min
            row["dist_to_other_max"] = p.intensity_max
            row["dist_to_other_mean"] = p.intensity_mean

    return rows


def filter_objects(df: pd.DataFrame, min_area: int = 0,
                   min_intensity: float = 0.0) -> pd.DataFrame:
    """按面积和平均强度过滤对象。"""
    return df[(df["area"] >= min_area) & (df["mean_intensity"] >= min_intensity)]


def analyze_image(mask_path: Path, output_path: Path):
    """处理一张 _masks_pred.tif, 返回 (summary, mito_feats, lipid_feats)。"""
    masks = imread(mask_path)
    mito_mask, lipid_mask = masks[MITO_CHANNEL], masks[LIPID_CHANNEL]

    img_path = mask_path.with_name(
        mask_path.name.replace("_masks_pred.tif", "_img.tif"))
    if not img_path.exists():
        raise FileNotFoundError(f"原图未找到: {img_path}")
    img = imread(img_path)
    mito_img, lipid_img = img[MITO_CHANNEL], img[LIPID_CHANNEL]

    # 交叉距离图：mito_edt 到最近 mito 的距离，用于衡量 lipid 与 mito 的邻近程度
    mito_edt = np.asarray(nd.distance_transform_edt(mito_mask == 0), dtype=np.float32)
    lipid_edt = np.asarray(nd.distance_transform_edt(lipid_mask == 0), dtype=np.float32)

    # mito 对象算它到最近 lipid 的距离，反之亦然
    mito_feats = extract_features(mito_mask, mito_img, "mito", cross_edt=lipid_edt)
    lipid_feats = extract_features(lipid_mask, lipid_img, "lipid", cross_edt=mito_edt)

    mito_binary, lipid_binary = mito_mask > 0, lipid_mask > 0
    overlap = int((mito_binary & lipid_binary).sum())
    mito_area, lipid_area = int(mito_binary.sum()), int(lipid_binary.sum())

    summary = {
        "image": mask_path.stem,
        "mito_count": len(mito_feats),
        "lipid_count": len(lipid_feats),
        "mito_total_area": mito_area,
        "lipid_total_area": lipid_area,
        "overlap_area": overlap,
        "overlap_ratio_mito": overlap / mito_area if mito_area else 0.0,
        "overlap_ratio_lipid": overlap / lipid_area if lipid_area else 0.0,
    }

    # 保存两个通道的欧几里得距离变换图（对象内 0，越远越大）
    for name, edt in (("mito", mito_edt), ("lipid", lipid_edt)):
        imwrite(output_path / f"{mask_path.stem}_{name}_edt.tif", edt)

    return summary, mito_feats, lipid_feats


@app.command()
def main(
    pred_path: Path = PRED_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    min_area: int = 0,
    min_intensity: float = 0.0,
) -> None:
    mask_paths = sorted(pred_path.glob("*_masks_pred.tif"))
    if not mask_paths:
        logger.warning(f"在 {pred_path} 下未找到 *_masks_pred.tif")
        return

    summaries, mito_all, lipid_all = [], [], []
    for mask_path in tqdm(mask_paths, desc="Analysis"):
        try:
            summary, mito_feats, lipid_feats = analyze_image(mask_path, output_path)
        except Exception as e:
            logger.error(f"{mask_path.name} 处理失败: {e}")
            continue

        # 过滤掉噪声对象后再汇总
        mito_feats = filter_objects(
            pd.DataFrame(mito_feats), min_area, min_intensity).to_dict("records")
        lipid_feats = filter_objects(
            pd.DataFrame(lipid_feats), min_area, min_intensity).to_dict("records")

        summary["mito_count"] = len(mito_feats)
        summary["lipid_count"] = len(lipid_feats)

        summaries.append(summary)
        mito_all.extend(mito_feats)
        lipid_all.extend(lipid_feats)

    pd.DataFrame(mito_all, columns=OBJECT_COLUMNS).to_csv(
        output_path / "mito_objects.csv", index=False)
    pd.DataFrame(lipid_all, columns=OBJECT_COLUMNS).to_csv(
        output_path / "lipid_objects.csv", index=False)
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(
        output_path / "summary.csv", index=False)
    logger.success(f"分析结果已保存至 {output_path}")


if __name__ == "__main__":
    app()