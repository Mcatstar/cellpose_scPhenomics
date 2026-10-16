"""visualize_contact.py —— 可视化 Mito-LD 互作区域。

对每张图:
    1. 读原图 + mito mask + ld mask
    2. 计算 mito 边界到 ld 边界的距离
    3. 距离 ≤ 30 nm 的像素标为接触区
    4. 参与接触的线粒体（PDM）用低饱和蓝标出
    5. LD 用低饱和橙标出
    6. 接触区用醒目红标出

输出:
    <figure_dir>/<name>_contact_overlay.png    单张重叠图
    <figure_dir>/<name>_contact_overlay_sidebyside.png   原图 + overlay 并排

配色（低饱和）:
    原图        灰度
    LD          淡橙 (250, 220, 180)
    PDM         淡蓝 (170, 200, 235)
    Contact     醒目红 (215, 85, 85)  ← 30 nm 内的接触区

用法:
    uv run python -m src.visualize_contact
    uv run python -m src.visualize_contact --only-with-contact
    uv run python -m src.visualize_contact --contact-threshold-nm 30
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy import ndimage
from skimage import measure
from tifffile import imread
from tqdm import tqdm

from src.config import (
    INTERIM_DATA_DIR,
    PRED_DATA_DIR,
    REPORTS_DIR,
)

app = typer.Typer()

# ---------- 配色（低饱和） ----------
LD_COLOR = (250, 220, 180)          # 淡橙
PDM_COLOR = (170, 200, 235)         # 淡蓝
CONTACT_COLOR = (215, 85, 85)       # 醒目红

LD_ALPHA = 0.35
PDM_ALPHA = 0.45
CONTACT_ALPHA = 0.85

FALLBACK_NM_PER_PX = 0.6153846153846154


# ---------- 工具 ----------

def to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """把任意位深的图线性映射到 uint8 灰度。"""
    if img.dtype == np.uint8 and img.ndim == 2:
        return img
    x = img.astype(np.float32)
    if x.ndim == 3:
        x = x[:, :, 0]
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def blend(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    """把 mask 区域用 color 半透明叠加到 rgb (float32 HxWx3) 上，原地修改。"""
    for c in range(3):
        rgb[..., c] = np.where(
            mask,
            rgb[..., c] * (1.0 - alpha) + color[c] * alpha,
            rgb[..., c],
        )


def load_metadata(metadata_path: Path) -> dict[str, float]:
    """从 metadata.csv 读 {图名 stem: nm_per_px}。"""
    if not metadata_path.is_file():
        logger.warning(f"metadata 不存在: {metadata_path}")
        return {}
    df = pd.read_csv(metadata_path)
    if "img_name" not in df.columns or "nm_per_px" not in df.columns:
        logger.error("metadata 缺少 img_name 或 nm_per_px 列")
        return {}
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        stem = Path(str(row["img_name"])).stem.removesuffix("_img")
        result[stem] = float(row["nm_per_px"])
    logger.info(f"metadata 载入 {len(result)} 张图的分辨率")
    return result


# ---------- 单张可视化 ----------

def compute_overlay(
    img_path: Path,
    mito_path: Path,
    ld_path: Path,
    nm_per_px: float,
    contact_threshold_nm: float,
) -> tuple[np.ndarray, dict]:
    """生成 overlay 图并返回统计信息。"""
    img = imread(img_path)
    mito = imread(mito_path)
    ld = imread(ld_path)

    if mito.shape != ld.shape:
        raise ValueError(f"mask 尺寸不一致: {mito.shape} vs {ld.shape}")

    # 原图转灰度
    img_gray = to_uint8_gray(img)
    if img_gray.shape != mito.shape:
        # 有些原始图有 black bar 被裁了，这里以 mask 尺寸为准
        h = min(img_gray.shape[0], mito.shape[0])
        w = min(img_gray.shape[1], mito.shape[1])
        img_gray = img_gray[:h, :w]
        mito = mito[:h, :w]
        ld = ld[:h, :w]

    mito_bin = mito > 0
    ld_bin = ld > 0

    # LD 边界 → 距离变换
    ld_boundary = ld_bin & ~ndimage.binary_erosion(
        ld_bin, iterations=1, border_value=0
    )
    dist_nm = ndimage.distance_transform_edt(~ld_boundary) * nm_per_px

    # 接触区：距离 ≤ 阈值的 mito 像素
    contact_zone = (dist_nm <= contact_threshold_nm) & mito_bin

    # 找接触的 mito 实例
    mito_rl = measure.label(mito_bin, connectivity=2)
    contact_labels = set(int(v) for v in np.unique(mito_rl[contact_zone]) if v > 0)
    pdm_mask = np.isin(mito_rl, list(contact_labels)) if contact_labels else np.zeros_like(mito_bin)

    # 组装 RGB
    rgb = np.stack([img_gray] * 3, axis=-1).astype(np.float32)
    blend(rgb, ld_bin, LD_COLOR, LD_ALPHA)
    blend(rgb, pdm_mask, PDM_COLOR, PDM_ALPHA)
    blend(rgb, contact_zone, CONTACT_COLOR, CONTACT_ALPHA)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    n_mito_total = int(mito_rl.max())
    n_pdm = len(contact_labels)
    n_contact_pixels = int(contact_zone.sum())
    contact_area_um2 = n_contact_pixels * (nm_per_px / 1000.0) ** 2

    stats = {
        "n_mito_total": n_mito_total,
        "n_pdm": n_pdm,
        "pdm_ratio": n_pdm / n_mito_total if n_mito_total > 0 else 0.0,
        "n_contact_pixels": n_contact_pixels,
        "contact_length_um": n_contact_pixels * nm_per_px / 1000.0,
        "contact_area_um2": contact_area_um2,
        "nm_per_px": nm_per_px,
    }
    return rgb, img_gray, stats


def save_overlay(
    img_gray: np.ndarray,
    rgb: np.ndarray,
    stats: dict,
    name: str,
    out_dir: Path,
    contact_threshold_nm: float,
) -> None:
    """保存单张 overlay 和并排对比图。"""
    # 单张 overlay
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(
        f"{name}  |  PDM {stats['n_pdm']}/{stats['n_mito_total']} "
        f"({stats['pdm_ratio']*100:.1f}%)  |  "
        f"contact {stats['contact_length_um']:.2f} μm  |  "
        f"≤{contact_threshold_nm:.0f} nm",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_contact_overlay.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 并排
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(rgb)
    axes[1].set_title(
        f"LD (orange)  |  PDM (blue)  |  contact ≤{contact_threshold_nm:.0f} nm (red)"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_contact_overlay_sidebyside.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------- 主流程 ----------

def find_triples(
    pred_dir: Path, image_dir: Path
) -> list[tuple[str, Path, Path, Path]]:
    """在 pred_dir 里找成对的 mask，再在 image_dir 里找对应的原图。"""
    triples: list[tuple[str, Path, Path, Path]] = []
    for mito_path in sorted(pred_dir.glob("*_mito_masks_pred.tif")):
        stem = mito_path.stem.removesuffix("_mito_masks_pred")
        ld_path = pred_dir / f"{stem}_ld_masks_pred.tif"
        img_path = image_dir / f"{stem}_img.tif"
        if not ld_path.is_file():
            logger.warning(f"缺少 ld mask: {stem}")
            continue
        if not img_path.is_file():
            logger.warning(f"缺少原图: {stem}")
            continue
        triples.append((stem, img_path, mito_path, ld_path))
    return triples


@app.command()
def main(
    pred_dir: Path = PRED_DATA_DIR,
    image_dir: Path = INTERIM_DATA_DIR,
    output_dir: Path = REPORTS_DIR / "contact_overlays",
    metadata_path: Path = INTERIM_DATA_DIR / "metadata.csv",
    contact_threshold_nm: float = 30.0,
    only_with_contact: bool = False,
) -> None:
    """批量生成 Mito-LD 互作可视化图。

    参数:
        pred_dir: 推理结果目录
        image_dir: 原图目录（INTERIM_DATA_DIR 里有 <name>_img.tif）
        output_dir: 输出 PNG 目录
        metadata_path: metadata.csv 路径
        contact_threshold_nm: 接触判定阈值（nm），默认 30
        only_with_contact: 只输出有接触的图
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_to_nm = load_metadata(metadata_path)
    triples = find_triples(pred_dir, image_dir)
    if not triples:
        logger.error("没有找到成对的 (原图, mito, ld)")
        raise typer.Exit(code=1)

    logger.info(f"找到 {len(triples)} 组，开始可视化")
    summary_rows = []

    for name, img_path, mito_path, ld_path in tqdm(triples, desc="Visualizing"):
        nm_per_px = stem_to_nm.get(name)
        if nm_per_px is None:
            logger.warning(f"{name} 不在 metadata，用兜底 {FALLBACK_NM_PER_PX}")
            nm_per_px = FALLBACK_NM_PER_PX

        try:
            rgb, img_gray, stats = compute_overlay(
                img_path, mito_path, ld_path,
                nm_per_px, contact_threshold_nm,
            )
        except Exception as e:
            logger.warning(f"{name} 处理失败: {e}")
            continue

        if only_with_contact and stats["n_pdm"] == 0:
            continue

        save_overlay(img_gray, rgb, stats, name, output_dir, contact_threshold_nm)
        summary_rows.append({"image": name, **stats})

    # 写一份汇总 CSV
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "overlay_summary.csv", index=False
        )

    logger.success(f"完成，输出到 {output_dir}")


if __name__ == "__main__":
    app()