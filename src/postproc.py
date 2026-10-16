"""postproc.py —— 推理后处理与 Mito-LD 互作指标分析。

读取 PRED_DATA_DIR 下 EmpanadaPredict.py 的输出:
    <name>_mito_masks_pred.tif
    <name>_ld_masks_pred.tif
mask 是连续整数标号图（0 = 背景，1~n = 实例），Cellpose 风格。

像素分辨率从 INTERIM_DATA_DIR/metadata.csv 按图名查询:
    metadata 的 img_name 例如 "C-1-2.5-1_img.tif" → 键 "C-1-2.5-1"
    每张图的 nm_per_px 可能不同，必须逐图换算。

按三篇文献的指标体系输出 CSV 到 REPORTS_DIR 根目录:
    global_metrics.csv           每张图一行的全局汇总
    mito_morphology.csv          每个线粒体实例的形态 + PDM/CM + 单实例接触比例
    ld_morphology.csv            每个脂滴实例的形态 + 大小分类 + 单实例接触比例
    contact_pairs.csv            每对 (mito, ld) 的接触长度与宽度 bin
    contact_width_bins.csv       每张图在各宽度 bin 上的接触长度
    contact_patterns.csv         每个 mito 的接触模式分类

指标来源:
    DeepContact            接触长度、宽度 10 nm bin、接触比例
    T2DM-MASLD (TEM)       PDM/CM 分类、LD 大小分类、接触模式
                            单实例接触长度/周长比值（截图核心指标）
    Human liver (Empanada) 面积 / 周长 / 圆度 / 长宽比 / 实度

单实例级接触指标:
    mito_morphology.contact_length_um   该线粒体与所有 LD 的接触长度之和
    mito_morphology.contact_ratio       接触长度 / 该线粒体周长
    ld_morphology.contact_length_um     该脂滴与所有 Mito 的接触长度之和
    ld_morphology.contact_ratio         接触长度 / 该脂滴周长
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from numpy.typing import NDArray
from scipy import ndimage
from skimage import measure
from tifffile import imread
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR, PRED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

# 30 nm 是文献里 Mito-LD 接触的常用临界距离
DEFAULT_CONTACT_THRESHOLD_NM = 30.0
DEFAULT_MAX_DISTANCE_NM = 100.0
DEFAULT_BIN_WIDTH_NM = 10.0
# metadata 缺失时的兜底分辨率，仅用于警告，不建议依赖
FALLBACK_NM_PER_PX = 0.6153846153846154


# ==================== metadata ====================

def load_metadata(metadata_path: Path) -> dict[str, float]:
    """读取 metadata.csv，返回 {图名 stem: nm_per_px}。

    metadata 的 img_name 例如 "C-1-2.5-1_img.tif"，去掉 "_img" 后缀后
    得到键 "C-1-2.5-1"，与预测结果的 stem 一致。
    """
    if not metadata_path.is_file():
        logger.warning(f"metadata 不存在: {metadata_path}")
        return {}

    df = pd.read_csv(metadata_path)
    if "img_name" not in df.columns or "nm_per_px" not in df.columns:
        logger.error("metadata 缺少 img_name 或 nm_per_px 列")
        return {}

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        img_name = str(row["img_name"])
        stem = Path(img_name).stem.removesuffix("_img")
        result[stem] = float(row["nm_per_px"])
    logger.info(f"metadata 载入 {len(result)} 张图的分辨率")
    return result


# ==================== 工具函数 ====================

def _bin_edges(max_distance_nm: float, bin_width_nm: float) -> NDArray:
    """生成宽度 bin 边界，例如 max=100, width=10 → [0, 10, ..., 100]。"""
    n = int(round(max_distance_nm / bin_width_nm))
    return np.linspace(0.0, n * bin_width_nm, n + 1)


def _extract_boundary(binary: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """单像素边界：mask 减去其腐蚀结果。"""
    eroded = ndimage.binary_erosion(binary, iterations=1, border_value=0)
    return binary & ~eroded


# ==================== 形态学 ====================

def morphology_table(
    mask: NDArray,
    pixel_size_nm: float,
    kind: str,
    image_name: str,
) -> pd.DataFrame:
    """对一张 mask 的每个实例计算形态参数。"""
    relabeled = measure.label(mask > 0, connectivity=2)
    px_to_um = pixel_size_nm / 1000.0

    rows = []
    for prop in measure.regionprops(relabeled):
        area_px = float(prop.area)
        perim_px = float(prop.perimeter)
        if area_px <= 0 or perim_px <= 0:
            continue

        area_um2 = area_px * px_to_um ** 2
        perim_um = perim_px * px_to_um

        elongation = (perim_px ** 2) / (4 * np.pi * area_px)
        circularity = 4 * np.pi * area_px / (perim_px ** 2)

        major_px = float(prop.axis_major_length)
        minor_px = float(prop.axis_minor_length)
        aspect_ratio = major_px / minor_px if minor_px > 0 else np.nan
        roundness = (
            4 * area_px / (np.pi * major_px ** 2) if major_px > 0 else np.nan
        )
        solidity = float(prop.solidity) if prop.solidity is not None else np.nan
        equiv_diam_um = 2.0 * np.sqrt(area_um2 / np.pi)

        rows.append({
            "image": image_name,
            "kind": kind,
            "label": int(prop.label),
            "nm_per_px": pixel_size_nm,
            "area_um2": area_um2,
            "perimeter_um": perim_um,
            "circularity": circularity,
            "elongation": elongation,
            "aspect_ratio": aspect_ratio,
            "roundness": roundness,
            "solidity": solidity,
            "equivalent_diameter_um": equiv_diam_um,
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
        })

    return pd.DataFrame(rows)


# ==================== 接触分析 ====================

def contact_table(
    mito_mask: NDArray,
    ld_mask: NDArray,
    pixel_size_nm: float,
    image_name: str,
    contact_threshold_nm: float,
    max_distance_nm: float,
    bin_width_nm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """向量化计算 mito-ld 接触。"""
    mito_bin = mito_mask > 0
    ld_bin = ld_mask > 0
    if not mito_bin.any() or not ld_bin.any():
        return pd.DataFrame(), pd.DataFrame()

    mito_rl = measure.label(mito_bin, connectivity=2)
    ld_rl = measure.label(ld_bin, connectivity=2)

    # LD 边界 + 距离变换（向量化，比双重 for 快几百倍）
    ld_boundary = _extract_boundary(ld_bin)
    ld_dist_nm = ndimage.distance_transform_edt(~ld_boundary) * pixel_size_nm

    # 每个像素最近的非零 LD 标号
    _, indices = ndimage.distance_transform_edt(
        ld_rl == 0, return_indices=True
    )
    nearest_ld = ld_rl[indices[0], indices[1]]

    # Mito 边界像素坐标
    mito_boundary = _extract_boundary(mito_bin)
    ys, xs = np.where(mito_boundary)
    if ys.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    dist_v = ld_dist_nm[ys, xs]
    mito_v = mito_rl[ys, xs]
    ld_v = nearest_ld[ys, xs]

    valid = (dist_v <= max_distance_nm) & (mito_v > 0) & (ld_v > 0)
    if not valid.any():
        return pd.DataFrame(), pd.DataFrame()

    dist_v = dist_v[valid]
    mito_v = mito_v[valid]
    ld_v = ld_v[valid]

    edges = _bin_edges(max_distance_nm, bin_width_nm)
    n_bins = len(edges) - 1
    px_to_um = pixel_size_nm / 1000.0

    df = pd.DataFrame({"mito": mito_v, "ld": ld_v, "dist": dist_v})

    pair_rows = []
    for (m, l), g in df.groupby(["mito", "ld"], sort=False):
        d = g["dist"].to_numpy()
        min_d = float(d.min())
        contact_mask = d <= contact_threshold_nm
        contact_len = float(contact_mask.sum()) * px_to_um

        row = {
            "image": image_name,
            "nm_per_px": pixel_size_nm,
            "mito_label": int(m),
            "ld_label": int(l),
            "min_dist_nm": min_d,
            "is_contact": bool(min_d <= contact_threshold_nm),
            "contact_length_um": contact_len,
            "n_contact_pixels": int(contact_mask.sum()),
        }
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            n_pix = int(((d > lo) & (d <= hi)).sum())
            row[f"bin_{int(lo)}_{int(hi)}nm_um"] = n_pix * px_to_um
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)

    image_bin_rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        n_pix = int(((dist_v > lo) & (dist_v <= hi)).sum())
        image_bin_rows.append({
            "image": image_name,
            "nm_per_px": pixel_size_nm,
            "bin_low_nm": float(lo),
            "bin_high_nm": float(hi),
            "contact_length_um": n_pix * px_to_um,
            "contact_pixel_count": n_pix,
        })
    image_bin_df = pd.DataFrame(image_bin_rows)

    return pair_df, image_bin_df


# ==================== PDM/CM 分类 ====================

def classify_pdm_cm(
    mito_mask: NDArray,
    pair_df: pd.DataFrame,
    image_name: str,
) -> pd.DataFrame:
    """把 mito 分为 PDM（接触 LD）和 CM（不接触 LD）。"""
    mito_rl = measure.label(mito_mask > 0, connectivity=2)
    labels = np.unique(mito_rl)
    labels = labels[labels > 0]
    if labels.size == 0:
        return pd.DataFrame()

    contact_labels: set[int] = set()
    if not pair_df.empty:
        contact_labels = set(
            pair_df.loc[pair_df["is_contact"], "mito_label"].astype(int).tolist()
        )

    rows = [
        {
            "image": image_name,
            "mito_label": int(m),
            "class": "PDM" if int(m) in contact_labels else "CM",
        }
        for m in labels
    ]
    return pd.DataFrame(rows)


def per_mito_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    """按 mito 汇总接触：接触 LD 数量、总接触长度。"""
    if pair_df.empty:
        return pd.DataFrame()
    contact = pair_df[pair_df["is_contact"]]
    if contact.empty:
        return pd.DataFrame()

    rows = []
    for (image, m), g in contact.groupby(["image", "mito_label"], sort=False):
        ld_ids = sorted(int(v) for v in g["ld_label"].unique())
        rows.append({
            "image": image,
            "mito_label": int(m),
            "n_contact_ld": len(ld_ids),
            "ld_labels": ";".join(map(str, ld_ids)),
            "contact_length_um": float(g["contact_length_um"].sum()),
        })
    return pd.DataFrame(rows)


def detect_contact_patterns(per_mito_df: pd.DataFrame) -> pd.DataFrame:
    """接触模式分类：单接触 vs 多接触。"""
    if per_mito_df.empty:
        return pd.DataFrame()

    df = per_mito_df.copy()
    df["pattern"] = np.where(
        df["n_contact_ld"] == 1, "single_mito_single_ld", "multi_contact"
    )
    return df[["image", "mito_label", "n_contact_ld", "pattern",
               "contact_length_um"]]


# ==================== 单实例接触长度与比例 ====================

def attach_contact_to_morphology(
    morph_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    label_col: str,
    label_in_pair: str,
) -> pd.DataFrame:
    """把每个实例的总接触长度 join 到形态表，并算接触比例。

    Args:
        morph_df: 形态表，含 'label' 列和 'perimeter_um' 列
        pair_df: 接触表，含 label_in_pair 列和 'is_contact'、'contact_length_um'
        label_col: morph_df 里的实例标号列名（固定为 'label'）
        label_in_pair: pair_df 里对应的标号列名（'mito_label' 或 'ld_label'）

    返回: 加了两列 contact_length_um / contact_ratio 的新 DataFrame
    """
    if morph_df.empty:
        return morph_df

    out = morph_df.copy()
    out["contact_length_um"] = 0.0

    if not pair_df.empty:
        contact_only = pair_df[pair_df["is_contact"]]
        if not contact_only.empty:
            # 同一实例可能有多对接触，要求和
            sums = contact_only.groupby(label_in_pair)["contact_length_um"].sum()
            out["contact_length_um"] = (
                out[label_col].map(sums).fillna(0.0).astype(float)
            )

    # 单实例接触比例 = 该实例的接触长度 / 该实例的周长
    perim = out["perimeter_um"].to_numpy()
    contact = out["contact_length_um"].to_numpy()
    ratio = np.zeros_like(contact)
    mask = perim > 0
    ratio[mask] = contact[mask] / perim[mask]
    out["contact_ratio"] = ratio

    return out


# ==================== 全局汇总 ====================

def build_global_row(
    image_name: str,
    nm_per_px: float,
    mito_morph: pd.DataFrame,
    ld_morph: pd.DataFrame,
    per_mito: pd.DataFrame,
    pdm_cm: pd.DataFrame,
    n_single: int,
    n_multi: int,
) -> dict:
    n_mito = len(mito_morph)
    n_ld = len(ld_morph)

    n_ld_small = (
        int((ld_morph["equivalent_diameter_um"] < 2.0).sum())
        if n_ld > 0 else 0
    )
    n_ld_large = n_ld - n_ld_small

    n_pdm = int((pdm_cm["class"] == "PDM").sum()) if not pdm_cm.empty else 0
    n_cm = n_mito - n_pdm

    n_mito_contact = len(per_mito) if not per_mito.empty else 0
    if not per_mito.empty:
        ld_ids = set()
        for s in per_mito["ld_labels"]:
            ld_ids.update(int(x) for x in str(s).split(";") if x)
        n_ld_contact = len(ld_ids)
    else:
        n_ld_contact = 0

    mito_perim_total = (
        float(mito_morph["perimeter_um"].sum()) if n_mito > 0 else 0.0
    )
    ld_perim_total = (
        float(ld_morph["perimeter_um"].sum()) if n_ld > 0 else 0.0
    )
    total_contact_len = (
        float(per_mito["contact_length_um"].sum()) if not per_mito.empty else 0.0
    )

    return {
        "image": image_name,
        "nm_per_px": nm_per_px,
        "n_mito": n_mito,
        "n_ld": n_ld,
        "n_ld_small": n_ld_small,
        "n_ld_large": n_ld_large,
        "n_pdm": n_pdm,
        "n_cm": n_cm,
        "pdm_ratio": n_pdm / n_mito if n_mito > 0 else 0.0,
        "n_mito_contact": n_mito_contact,
        "n_ld_contact": n_ld_contact,
        "n_pattern_single": n_single,
        "n_pattern_multi": n_multi,
        "total_contact_length_um": total_contact_len,
        "contact_ratio_mito": (
            total_contact_len / mito_perim_total if mito_perim_total > 0 else 0.0
        ),
        "contact_ratio_ld": (
            total_contact_len / ld_perim_total if ld_perim_total > 0 else 0.0
        ),
        "mito_total_area_um2": (
            float(mito_morph["area_um2"].sum()) if n_mito > 0 else 0.0
        ),
        "mito_mean_area_um2": (
            float(mito_morph["area_um2"].mean()) if n_mito > 0 else 0.0
        ),
        "mito_total_perimeter_um": mito_perim_total,
        "mito_mean_circularity": (
            float(mito_morph["circularity"].mean()) if n_mito > 0 else 0.0
        ),
        "mito_mean_aspect_ratio": (
            float(mito_morph["aspect_ratio"].mean()) if n_mito > 0 else 0.0
        ),
        "mito_mean_elongation": (
            float(mito_morph["elongation"].mean()) if n_mito > 0 else 0.0
        ),
        "ld_total_area_um2": (
            float(ld_morph["area_um2"].sum()) if n_ld > 0 else 0.0
        ),
        "ld_mean_area_um2": (
            float(ld_morph["area_um2"].mean()) if n_ld > 0 else 0.0
        ),
        "ld_total_perimeter_um": ld_perim_total,
    }


# ==================== 主流程 ====================

def find_pairs(pred_dir: Path) -> list[tuple[str, Path, Path]]:
    """找 pred_dir 下成对的 mito/ld mask。"""
    pairs: list[tuple[str, Path, Path]] = []
    for mito_path in sorted(pred_dir.glob("*_mito_masks_pred.tif")):
        stem = mito_path.stem.removesuffix("_mito_masks_pred")
        ld_path = pred_dir / f"{stem}_ld_masks_pred.tif"
        if not ld_path.is_file():
            logger.warning(f"缺少 ld mask, 跳过: {stem}")
            continue
        pairs.append((stem, mito_path, ld_path))
    return pairs


@app.command()
def main(
    pred_dir: Path = PRED_DATA_DIR,
    output_dir: Path = REPORTS_DIR,
    metadata_path: Path = INTERIM_DATA_DIR / "metadata.csv",
    contact_threshold_nm: float = DEFAULT_CONTACT_THRESHOLD_NM,
    max_distance_nm: float = DEFAULT_MAX_DISTANCE_NM,
    bin_width_nm: float = DEFAULT_BIN_WIDTH_NM,
) -> None:
    """读取推理结果，按每张图的 nm_per_px 换算物理单位，输出全套 CSV。

    参数:
        pred_dir: 推理结果目录
        output_dir: CSV 输出目录
        metadata_path: 含 img_name 和 nm_per_px 的 metadata.csv
        contact_threshold_nm: 接触判定阈值（nm），默认 30
        max_distance_nm: 宽度 bin 分析的最大距离（nm），默认 100
        bin_width_nm: 宽度 bin 步长（nm），默认 10
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_to_nm = load_metadata(metadata_path)
    if not stem_to_nm:
        logger.warning(
            f"metadata 未载入任何分辨率，所有图将用兜底值 {FALLBACK_NM_PER_PX} nm/px"
        )

    pairs = find_pairs(pred_dir)
    if not pairs:
        logger.error(f"{pred_dir} 下没有找到成对的 mito/ld mask")
        raise typer.Exit(code=1)
    logger.info(f"找到 {len(pairs)} 对 mask，开始分析")
    logger.info(f"接触阈值 = {contact_threshold_nm} nm")

    global_rows: list[dict] = []
    mito_morphs: list[pd.DataFrame] = []
    ld_morphs: list[pd.DataFrame] = []
    pair_dfs: list[pd.DataFrame] = []
    bin_dfs: list[pd.DataFrame] = []
    pattern_dfs: list[pd.DataFrame] = []

    for name, mito_path, ld_path in tqdm(pairs, desc="Analyzing"):
        nm_per_px = stem_to_nm.get(name)
        if nm_per_px is None:
            logger.warning(
                f"{name} 不在 metadata，使用兜底 {FALLBACK_NM_PER_PX} nm/px"
            )
            nm_per_px = FALLBACK_NM_PER_PX

        mito_mask = imread(mito_path)
        ld_mask = imread(ld_path)
        if mito_mask.shape != ld_mask.shape:
            logger.warning(f"{name} mito/ld 尺寸不一致，跳过")
            continue

        # ---- 形态学 ----
        mito_morph = morphology_table(mito_mask, nm_per_px, "mito", name)
        ld_morph = morphology_table(ld_mask, nm_per_px, "ld", name)
        if not ld_morph.empty:
            ld_morph["size_class"] = np.where(
                ld_morph["equivalent_diameter_um"] < 2.0, "small", "large"
            )

        # ---- 接触 ----
        pair_df, image_bin_df = contact_table(
            mito_mask, ld_mask, nm_per_px, name,
            contact_threshold_nm=contact_threshold_nm,
            max_distance_nm=max_distance_nm,
            bin_width_nm=bin_width_nm,
        )

        # ---- PDM/CM 分类 ----
        pdm_cm = classify_pdm_cm(mito_mask, pair_df, name)
        if not mito_morph.empty and not pdm_cm.empty:
            mito_morph = mito_morph.merge(
                pdm_cm[["mito_label", "class"]],
                left_on="label", right_on="mito_label", how="left",
            ).drop(columns=["mito_label"])

        # ---- 单实例接触长度与比例（截图核心指标）----
        mito_morph = attach_contact_to_morphology(
            mito_morph, pair_df, label_col="label", label_in_pair="mito_label"
        )
        ld_morph = attach_contact_to_morphology(
            ld_morph, pair_df, label_col="label", label_in_pair="ld_label"
        )

        # ---- per-mito 汇总 + 模式 ----
        per_mito = per_mito_summary(pair_df)
        pattern_df = detect_contact_patterns(per_mito)

        n_single = int((pattern_df["pattern"] == "single_mito_single_ld").sum()) \
            if not pattern_df.empty else 0
        n_multi = int((pattern_df["pattern"] == "multi_contact").sum()) \
            if not pattern_df.empty else 0

        global_rows.append(build_global_row(
            name, nm_per_px, mito_morph, ld_morph, per_mito, pdm_cm,
            n_single, n_multi,
        ))

        if not mito_morph.empty:
            mito_morphs.append(mito_morph)
        if not ld_morph.empty:
            ld_morphs.append(ld_morph)
        if not pair_df.empty:
            pair_dfs.append(pair_df)
        if not image_bin_df.empty:
            bin_dfs.append(image_bin_df)
        if not pattern_df.empty:
            pattern_dfs.append(pattern_df)

    # ---- 写 CSV ----
    pd.DataFrame(global_rows).to_csv(
        output_dir / "global_metrics.csv", index=False
    )
    if mito_morphs:
        pd.concat(mito_morphs, ignore_index=True).to_csv(
            output_dir / "mito_morphology.csv", index=False
        )
    if ld_morphs:
        pd.concat(ld_morphs, ignore_index=True).to_csv(
            output_dir / "ld_morphology.csv", index=False
        )
    if pair_dfs:
        pd.concat(pair_dfs, ignore_index=True).to_csv(
            output_dir / "contact_pairs.csv", index=False
        )
    if bin_dfs:
        pd.concat(bin_dfs, ignore_index=True).to_csv(
            output_dir / "contact_width_bins.csv", index=False
        )
    if pattern_dfs:
        pd.concat(pattern_dfs, ignore_index=True).to_csv(
            output_dir / "contact_patterns.csv", index=False
        )

    logger.success(f"完成，CSV 输出到 {output_dir}")


if __name__ == "__main__":
    app()