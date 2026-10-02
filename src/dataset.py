from pathlib import Path
import re
import time, random

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import ndimage as nd
import cv2
from rapidocr import EngineType, RapidOCR, LangDet, LangRec
from tifffile import imread, imwrite, TiffFile
from loguru import logger
from tqdm import tqdm
import typer

from src.config import EXTERNAL_DATA_DIR, IMAGES_DATA_DIR, INTERIM_DATA_DIR

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
        "EngineConfig.torch.use_cuda": True,  # 使用 torch GPU 版推理
        "EngineConfig.torch.cuda_ep_cfg.device_id": 0,  # 指定GPU id
    })
    for img_index, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Dataset Generation", unit="file"):
        # Select image and load it
        logger.info(f"Selected image: {img_path.name}")
        logger.info(f"Image path: {img_path}")
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
            bottom_8u = cv2.normalize(src=black_bar, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8) # pyright: ignore[reportArgumentType, reportCallIssue]
            _, binary = cv2.threshold(bottom_8u, 200, 255, cv2.THRESH_BINARY)
            ocrres = OCRengine(binary, use_det=False, use_cls=False, use_rec=True)
            # ocrres.vis("vis_result.jpg")
            text = ocrres.txts[0]
            logger.debug(text)
            pattern = re.compile(r'\d{1,2}:\d{1,2}:(\d{1,2})\s*(\d*\.?\d+)\s*([uμ]m|nm)', re.IGNORECASE)
            scale_match = pattern.search(text)
            bar_value = float(scale_match.group(2)) # type: ignore
            unit = scale_match.group(3).lower() # type: ignore
            if 'nm' in unit:
                physical_nm = bar_value
            else:  # µm / um
                physical_nm = bar_value * 1000.0
            col_white = np.sum(binary > 0, axis=0)
            min_white = int(bh * 0.30)  # 刻度线至少占黑条高度的 30%
            tick_cols = np.where(col_white > min_white)[0]
            groups, current = [], [tick_cols[0]]
            for c in tick_cols[1:]:
                if c - current[-1] <= 3:
                    current.append(c)
                else:
                    groups.append(current)
                    current = [c]
            groups.append(current)

            tick_centers = [int(np.mean(g)) for g in groups]
            left_x, right_x = tick_centers[0], tick_centers[-1]
            pixel_distance = right_x - left_x
            nm_per_px = physical_nm / pixel_distance
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
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
