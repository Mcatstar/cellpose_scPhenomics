"""Quantify and filter Cellpose predictions for mitochondria and lipid droplets.

Reads predicted masks (two channels: mito at index 0, lipid at index 1),
extracts per-object morphometric and intensity features using
scikit-image's regionprops, computes cross-organelle distance statistics,
and writes the results as CSV files.

量化并筛选 Cellpose 对线粒体和脂质滴的预测结果。

读取预测的掩膜（两个通道：线粒体在索引 0, 脂质在索引 1) ,
使用 scikit-image 的 regionprops 提取每个对象的形态和强度特征，计算跨细胞器的距离统计，
并将结果写入 CSV 文件。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy import ndimage as nd
from skimage.measure import regionprops
from tifffile import imread, imwrite
from tqdm import tqdm

from src.config import PRED_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# %% ------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Channel layout of the (C, H, W) TIF files produced by the prediction step.
MITO_CHANNEL = 0
LIPID_CHANNEL = 1

# Column order for the per-object CSVs.  Kept explicit so that an empty
# result still produces a CSV with the correct header.
OBJECT_COLUMNS = [
    "roi_id",
    "organelle",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "axis_major_length",
    "axis_minor_length",
    "feret_diameter_max",
    "centroid_y",
    "centroid_x",
    "intensity_mean",
    "dist_to_other_min",
    "dist_to_other_max",
    "dist_to_other_mean",
]

SUMMARY_COLUMNS = [
    "image",
    "mito_count",
    "lipid_count",
    "mito_total_area",
    "lipid_total_area",
    "overlap_area",
    "overlap_ratio_mito",
    "overlap_ratio_lipid",
]


# %% ------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    mask: np.ndarray,
    intensity_image: np.ndarray,
    organelle: str,
    cross_distance_map: np.ndarray | None = None,
) -> list[dict]:
    """Extract morphometric and intensity features for every object in *mask*.

    Parameters
    ----------
    mask : np.ndarray
        2D label image where 0 is background and each positive integer is a
        distinct object.
    intensity_image : np.ndarray
        2D grayscale image used to measure mean intensity per object.  Must
        have the same spatial shape as *mask*.
    organelle : str
        Either ``"mito"`` or ``"lipid"``.  Stored as a column so that mito and
        lipid rows can be concatenated and later distinguished.
    cross_distance_map : np.ndarray or None, optional
        Euclidian distance transform of the *other* organelle.  For mito
        objects this is the distance to the nearest lipid droplet, and vice
        versa.  When provided, three additional columns are appended:
        ``dist_to_other_min``, ``dist_to_other_max`` and ``dist_to_other_mean``.

    Returns
    -------
    list of dict
        One dict per object, ready to be passed to ``pd.DataFrame``.
    """
    features = [
        {
            "roi_id": region.label,
            "organelle": organelle,
            "area": region.area,
            "perimeter": region.perimeter,
            "eccentricity": region.eccentricity,
            "solidity": region.solidity,
            "axis_major_length": region.axis_major_length,
            "axis_minor_length": region.axis_minor_length,
            "feret_diameter_max": region.feret_diameter_max,
            "centroid_y": region.centroid[0],
            "centroid_x": region.centroid[1],
            "intensity_mean": region.intensity_mean,
        }
        for region in regionprops(mask, intensity_image=intensity_image)
    ]

    if cross_distance_map is not None:
        cross_regions = regionprops(mask, intensity_image=cross_distance_map)
        for row, region in zip(features, cross_regions):
            row["dist_to_other_min"] = region.intensity_min
            row["dist_to_other_max"] = region.intensity_max
            row["dist_to_other_mean"] = region.intensity_mean

    return features


# %% ------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_objects(
    features: pd.DataFrame,
    min_area: int = 0,
    min_intensity: float = 0.0,
) -> pd.DataFrame:
    """Drop objects that fall below the area or mean-intensity thresholds.

    Parameters
    ----------
    features : pd.DataFrame
        Per-object table produced by :func:`extract_features`.
    min_area : int, optional
        Objects with an area smaller than this value are removed.  Defaults
        to 0, which keeps every object.
    min_intensity : float, optional
        Objects whose mean intensity is below this value are removed.
        Defaults to 0.0, which keeps every object.

    Returns
    -------
    pd.DataFrame
        Filtered table with the same columns as *features*.
    """
    keep = (features["area"] >= min_area) & (features["intensity_mean"] >= min_intensity)
    return features[keep]


# %% ------------------------------------------------------------------------
# Single-image analysis
# ---------------------------------------------------------------------------


def analyze_image(mask_path: Path, output_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Analyze one prediction file and write its distance maps to disk.

    Parameters
    ----------
    mask_path : Path
        Path to a ``*_masks_pred.tif`` file containing a stacked (2, H, W)
        array: channel 0 is the mito mask, channel 1 is the lipid mask.
    output_path : Path
        Directory where the two distance-transform TIFs are written.

    Returns
    -------
    summary : dict
        Image-level counts, total areas and overlap statistics.
    mito_features : list of dict
        Per-object features for the mitochondria channel.
    lipid_features : list of dict
        Per-object features for the lipid channel.
    """
    masks = imread(mask_path)
    mito_mask = masks[MITO_CHANNEL]
    lipid_mask = masks[LIPID_CHANNEL]

    image_path = mask_path.with_name(mask_path.name.replace("_masks_pred.tif", "_img.tif"))
    if not image_path.exists():
        raise FileNotFoundError(f"Original image not found: {image_path}")

    image = imread(image_path)
    mito_image = image[MITO_CHANNEL]
    lipid_image = image[LIPID_CHANNEL]

    # Distance transform: zero inside an object, increasing towards the
    # nearest background pixel.  Used as a cross-organelle proximity measure.
    mito_distance = np.asarray(nd.distance_transform_edt(mito_mask == 0), dtype=np.float32)
    lipid_distance = np.asarray(nd.distance_transform_edt(lipid_mask == 0), dtype=np.float32)

    # Each organelle is quantified against the distance map of the *other*
    # organelle, so that "dist_to_other_*" answers "how close is this mito to
    # the nearest lipid droplet?" and vice versa.
    mito_features = extract_features(
        mito_mask, mito_image, "mito", cross_distance_map=lipid_distance
    )
    lipid_features = extract_features(
        lipid_mask, lipid_image, "lipid", cross_distance_map=mito_distance
    )

    mito_binary = mito_mask > 0
    lipid_binary = lipid_mask > 0
    overlap_area = int((mito_binary & lipid_binary).sum())
    mito_area = int(mito_binary.sum())
    lipid_area = int(lipid_binary.sum())

    summary = {
        "image": mask_path.stem,
        "mito_count": len(mito_features),
        "lipid_count": len(lipid_features),
        "mito_total_area": mito_area,
        "lipid_total_area": lipid_area,
        "overlap_area": overlap_area,
        "overlap_ratio_mito": overlap_area / mito_area if mito_area else 0.0,
        "overlap_ratio_lipid": overlap_area / lipid_area if lipid_area else 0.0,
    }

    # Persist the distance maps so they can be inspected or reused downstream.
    imwrite(output_path / f"{mask_path.stem}_mito_edt.tif", mito_distance)
    imwrite(output_path / f"{mask_path.stem}_lipid_edt.tif", lipid_distance)

    return summary, mito_features, lipid_features


# %% ------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@app.command()
def main(
    pred_path: Path = PRED_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    min_area: int = 0,
    min_intensity: float = 0.0,
) -> None:
    """Run per-object quantification over a directory of predictions.

    Parameters
    ----------
    pred_path : Path
        Directory containing ``*_masks_pred.tif`` files.
    output_path : Path
        Directory where the CSV summaries and distance-map TIFs are written.
    min_area : int
        Passed through to :func:`filter_objects`.
    min_intensity : float
        Passed through to :func:`filter_objects`.
    """
    mask_paths = sorted(pred_path.glob("*_masks_pred.tif"))
    if not mask_paths:
        logger.warning(f"No *_masks_pred.tif files found in {pred_path}")
        return

    summaries: list[dict] = []
    mito_rows: list[dict] = []
    lipid_rows: list[dict] = []

    for mask_path in tqdm(mask_paths, desc="Analysis"):
        try:
            summary, mito_features, lipid_features = analyze_image(mask_path, output_path)
        except Exception as exc:
            logger.error(f"{mask_path.name} failed: {exc}")
            continue

        # Filter before aggregation so that counts and CSVs stay consistent.
        mito_features = filter_objects(
            pd.DataFrame(mito_features), min_area, min_intensity
        ).to_dict("records")
        lipid_features = filter_objects(
            pd.DataFrame(lipid_features), min_area, min_intensity
        ).to_dict("records")

        summary["mito_count"] = len(mito_features)
        summary["lipid_count"] = len(lipid_features)

        summaries.append(summary)
        mito_rows.extend(mito_features)
        lipid_rows.extend(lipid_features)

    pd.DataFrame(mito_rows, columns=OBJECT_COLUMNS).to_csv(
        output_path / "mito_objects.csv", index=False
    )
    pd.DataFrame(lipid_rows, columns=OBJECT_COLUMNS).to_csv(
        output_path / "lipid_objects.csv", index=False
    )
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(
        output_path / "summary.csv", index=False
    )

    logger.success(f"Analysis results saved to {output_path}")


if __name__ == "__main__":
    app()
