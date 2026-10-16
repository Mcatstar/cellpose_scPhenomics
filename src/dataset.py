from pathlib import Path
import re
import time, random

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import ndimage as nd
import cv2
from rapidocr import EngineType, RapidOCR, LangDet, LangRec
from sklearn.model_selection import train_test_split
import shutil
from tifffile import imread, imwrite, TiffFile
from loguru import logger
from tqdm import tqdm
import typer

from src.config import EXTERNAL_DATA_DIR, IMAGES_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = IMAGES_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    # Load image-------------------------------------
    img_list: list[Path] = list(Path(input_path).rglob("*.tif"))

    logger.info("images found:\n" + "\n".join(map(lambda x: str(x.relative_to(input_path)), img_list)))
    logger.info(f"Number of images found: {len(img_list)}")
    # img_index = int(input('Select image: '))
    metadata = []
    metadata_path = output_path / "metadata.csv"
    OCRengine = RapidOCR(params={
        "Det.engine_type": EngineType.TORCH,
        "Det.lang_type": LangDet.EN,
        "Cls.engine_type": EngineType.TORCH,
        "Cls.lang_type": LangDet.CH,
        "Rec.engine_type": EngineType.TORCH,
        "Rec.lang_type": LangRec.EN,
        # "EngineConfig.torch.use_cuda": True,  # 使用 torch GPU 版推理
        # "EngineConfig.torch.cuda_ep_cfg.device_id": 0,  # 指定GPU id
    })
    mito_labeled, ld_labeled, inference_only = [], [], []
    for img_index, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Dataset Generation", unit="file"):
        # Select image and load it
        logger.info(f"Selected image: {img_path.name}")
        logger.info(f"Image path: {img_path}")
        # process image and extract metadata
        with TiffFile(img_path) as tif:
            img = tif.asarray()
            page = tif.pages.first

            page = tif.pages.first
            desc = page.description
            Magnification = re.search(r'Magnification=([\d\.]+)', desc).group(1) # pyright: ignore[reportOptionalMemberAccess]
    
            width_px  = page.imagewidth
            height_px = page.imagelength
            origin_shape = page.shape# (H, W)
            dtype = page.dtype
            row_medians = np.median(img, axis=1)
            black_bar_height = len(row_medians) - 1 - np.max(np.where(row_medians > 5)[0], initial=-1)
            effective_img = img[:img.shape[0] - black_bar_height, :]
            black_bar = img[img.shape[0] - black_bar_height:, :]
            bh = black_bar.shape[0]
            bottom_8u = cv2.normalize(
                src=black_bar, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            ).astype(np.uint8) # type: ignore
            _, binary = cv2.threshold(bottom_8u, 200, 255, cv2.THRESH_BINARY)

            # --- OCR 同时拿 bbox 和文字 ---
            ocrres = OCRengine(binary, use_det=True, use_cls=False, use_rec=True)

            # 1) 从所有识别结果里找比例尺文本（数字 + nm/um/μm/µm）
            scale_pattern = re.compile(r'(\d+\.?\d*)\s*(nm|um|μm|µm)', re.IGNORECASE)
            scale_text = None
            for t in ocrres.txts:
                if scale_pattern.search(t):
                    scale_text = t
                    break

            if scale_text is None:
                logger.warning(f"未找到比例尺文本，跳过 {img_path.name}：{ocrres.txts}")
                continue

            # 2) 所有文字 bbox 生成 mask，从二值图扣掉
            text_mask = np.zeros_like(binary)
            for box in ocrres.boxes:
                pts = np.asarray(box, dtype=np.int32)
                cv2.fillPoly(text_mask, [pts], 255)
            text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)
            binary_no_text = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))

            # 3) 在无文字区域检测刻度线
            col_white = np.sum(binary_no_text > 0, axis=0)
            min_white = max(3, int(bh * 0.20))
            tick_cols = np.where(col_white > min_white)[0]

            if len(tick_cols) < 2:
                logger.warning(f"刻度线列不足，跳过 {img_path.name}")
                continue

            groups, current = [], [tick_cols[0]]
            for c in tick_cols[1:]:
                if c - current[-1] <= 3:
                    current.append(c)
                else:
                    groups.append(current)
                    current = [c]
            groups.append(current)
            
            tick_centers = [int(np.mean(g)) for g in groups]
            if len(tick_centers) < 2:
                logger.warning(f"有效刻度线组不足 2，跳过 {img_path.name}")
                continue

            left_x, right_x = tick_centers[0], tick_centers[-1]
            pixel_distance = right_x - left_x
            if pixel_distance <= 0:
                logger.warning(f"刻度线像素距离异常：{tick_centers}")
                continue
            # 4) 解析物理长度
            m = scale_pattern.search(scale_text)
            bar_value = float(m.group(1))
            unit = m.group(2).lower()
            physical_nm = bar_value if unit == 'nm' else bar_value * 1000.0
            nm_per_px = physical_nm / pixel_distance

            # 调试日志，跑第一张图时确认
            logger.debug(
                f"[{img_path.name}] scale_text={scale_text!r}, "
                f"tick_centers={tick_centers}, pixel_distance={pixel_distance}, "
                f"physical_nm={physical_nm}, nm_per_px={nm_per_px:.6f}"
            )

            metadata.append(
                {
                    "origin_path": str(img_path.relative_to(input_path)),
                    "origin_name": img_path.name,
                    "img_name": f"{img_path.parent.name}-{img_path.stem}_img.tif",
                    "origin_shape": origin_shape,
                    "shape": effective_img.shape,
                    "width_px": width_px,
                    "height_px": height_px - black_bar_height,
                    "Magnification": float(Magnification) / 1000,
                    "nm_per_px": nm_per_px,
                    "dtype": str(dtype)
                }
            )

        imwrite(
            output_path / f"{img_path.parent.name}-{img_path.stem}_img.tif",
            np.uint16(effective_img),
        )
        logger.info(
            f"Successfully processing the No.{img_index + 1} image in total {len(img_list)}"
        )

        # processing labels and generate masks
        mito_labels = EXTERNAL_DATA_DIR / f"{img_path.parent.name}-{img_path.stem}_mito_labels.tif"
        ld_labels = EXTERNAL_DATA_DIR / f"{img_path.parent.name}-{img_path.stem}_ld_labels.tif"
        if mito_labels.exists():
            mito_mask = imread(mito_labels)[: effective_img.shape[0], : effective_img.shape[1]]
            imwrite(
                output_path / f"{img_path.parent.name}-{img_path.stem}_mito_masks.tif",
                mito_mask,
            )
            mito_labeled.append(f"{img_path.parent.name}-{img_path.stem}")
        if ld_labels.exists():
            ld_mask = imread(ld_labels)[: effective_img.shape[0], : effective_img.shape[1]]
            imwrite(
                output_path / f"{img_path.parent.name}-{img_path.stem}_ld_masks.tif",
                ld_mask,
            )
            ld_labeled.append(f"{img_path.parent.name}-{img_path.stem}")
        if (not mito_labels.exists()) or (not ld_labels.exists()):
            inference_only.append(f"{img_path.parent.name}-{img_path.stem}")

    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)

    logger.debug(f"mito_labeled: {mito_labeled}")
    logger.debug(f"ld_labeled: {ld_labeled}")
    logger.debug(f"inference_only: {inference_only}")

     # splite into train/test sets
    train_ratio = 0.8
    mito_train, mito_test = train_test_split(mito_labeled, train_size=train_ratio, random_state=42)
    ld_train, ld_test = train_test_split(ld_labeled, train_size=train_ratio, random_state=42)
    rows = (
        [{"split": "mito_train", "name": n} for n in mito_train] +
        [{"split": "mito_test",  "name": n} for n in mito_test] +
        [{"split": "ld_train",   "name": n} for n in ld_train] +
        [{"split": "ld_test",    "name": n} for n in ld_test] +
        [{"split": "infer",      "name": n} for n in inference_only]
    )
    pd.DataFrame(rows).to_csv(output_path / "train_test_split.csv", index=False)
    # save train/test sets
    config = {
        ("mito", "train"): mito_train,
        ("mito", "test"):  mito_test,
        ("ld",   "train"): ld_train,
        ("ld",   "test"):  ld_test,
    }

    for (dtype, split), names in config.items():
        dst = PROCESSED_DATA_DIR / split / dtype
        dst.mkdir(parents=True, exist_ok=True)
        for name in names:
            for src_suffix, dst_suffix in ((f"{dtype}_masks", f"{dtype}_masks"),):
                shutil.copy2(output_path / f"{name}_img.tif", dst / f"{name}_img.tif")
                shutil.copy2(output_path / f"{name}_{src_suffix}.tif", dst / f"{name}_{dst_suffix}.tif")

    # inference 单独处理（只复制 img）
    infer_dst = PROCESSED_DATA_DIR / "infer"
    infer_dst.mkdir(parents=True, exist_ok=True)
    for name in inference_only:
        shutil.copy2(output_path / f"{name}_img.tif", infer_dst / f"{name}_img.tif")
    

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
