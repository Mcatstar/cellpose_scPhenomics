from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
import logging; logging.basicConfig(level=logging.DEBUG)
from skimage import io
from tqdm import tqdm

from empanada.config_loaders import load_config
from empanada.inference import engines
from src.config import MODELS_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROJ_ROOT

app = typer.Typer()

MODELS_CONFIGS_DIR = PROJ_ROOT / "src" / "modeling" / "configs"


class EmpanadaPredictor:
    """加载 empanada TorchScript 模型做推理。"""

    name: str = "empanada"

    def __init__(self, model_path: Path, config_path: Path, gpu: bool = True):
        self.gpu = gpu and torch.cuda.is_available()
        self.device = "cuda" if self.gpu else "cpu"

        # 加载配置
        self.cfg = load_config(str(config_path))
        self.norms = self.cfg["norms"]
        self.padding_factor = self.cfg["padding_factor"]
        ep = self.cfg["FINETUNE"]["engine_params"]

        # 保存推理后处理所需参数
        self.label_divisor = ep["label_divisor"]
        self.thing_list = ep["thing_list"]
        self.stuff_area = ep["stuff_area"]
        self.void_label = ep["void_label"]

        # 加载 TorchScript 模型
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

        # 创建推理引擎
        self.engine = engines.PanopticDeepLabEngine(
            model=self.model,
            confidence_thr=ep["confidence_thr"],
            label_divisor=ep["label_divisor"],
            nms_kernel=ep["nms_kernel"],
            nms_threshold=ep["nms_threshold"],
            stuff_area=ep["stuff_area"],
            thing_list=ep["thing_list"],
            void_label=ep["void_label"],
        )

        logger.info(f"[{self.name}] loaded model from {model_path}")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """单通道灰度图 → 归一化 + padding → tensor (1, 1, H, W)。"""
        if image.ndim == 3:
            image = image[:, :, 0]

        image = image.astype(np.float32)
        # 根据 dtype 缩放到 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype in (np.float32, np.float64):
            image = image.astype(np.float32)
            # float 类型且 max > 1，说明还没归一化，按最大值归一
            if image.max() > 1.0:
                image = image / image.max()
        else:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / image.max()
        image = (image - self.norms["mean"]) / self.norms["std"]

        h, w = image.shape
        pf = self.padding_factor
        pad_h = (pf - h % pf) % pf
        pad_w = (pf - w % pf) % pf
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")

        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return tensor

    def _decode_pan_seg(self, pan_seg: np.ndarray) -> np.ndarray:
        """将 panoptic 编码标签解码为连续实例标签（1, 2, 3, ...）。

        pan_seg 的编码方式是: label = class_id * label_divisor + instance_id
        例如 label_divisor=1000, class_id=1 时:
            第 1 个实例 = 1001
            第 2 个实例 = 1002
        """
        instance_map = np.zeros_like(pan_seg, dtype=np.uint16)
        counter = 1
        for class_id in self.thing_list:
            mask = (pan_seg // self.label_divisor) == class_id
            for inst_label in np.unique(pan_seg[mask]):
                if inst_label == 0:
                    continue
                instance_map[pan_seg == inst_label] = counter
                counter += 1
        return instance_map

    def predict(self, image: np.ndarray) -> np.ndarray:
        """推理并返回实例标签图（uint16, H×W）。"""
        h, w = image.shape[:2]

        tensor = self._preprocess(image)

        with torch.no_grad():
            # 用 engine(tensor) 而不是 engine.infer(tensor)
            # __call__ 内部会做完 harden_seg + postprocess，返回 pan_seg
            pan_seg = self.engine(tensor)

        if isinstance(pan_seg, torch.Tensor):
            pan_seg = pan_seg.cpu().numpy()

        # 打印一次形状，便于确认
        if not hasattr(self, "_printed_shape"):
            logger.info(f"[{self.name}] engine() 输出形状: {pan_seg.shape}, dtype: {pan_seg.dtype}")
            self._printed_shape = True

        # 压缩 batch / channel 维度
        while pan_seg.ndim > 2:
            pan_seg = pan_seg[0]

        # 裁掉 padding 部分，恢复原始尺寸
        pan_seg = pan_seg[:h, :w]

        # 解码为连续实例标签
        instance_map = self._decode_pan_seg(pan_seg)

        return instance_map


class MitoPredictor(EmpanadaPredictor):
    name = "mito"

    def __init__(self, gpu: bool = True):
        super().__init__(
            model_path=MODELS_DIR / "MitoNet_v1_mini.pth",
            config_path=MODELS_CONFIGS_DIR / "MitoNet_v1_mini.yaml",
            gpu=gpu,
        )


class LipidPredictor(EmpanadaPredictor):
    name = "lipid"

    def __init__(self, gpu: bool = True):
        super().__init__(
            model_path=MODELS_DIR / "DropNet_base_v1.pth",
            config_path=MODELS_CONFIGS_DIR / "DropNet_base_v1.yaml",
            gpu=gpu,
        )


@app.command()
def main(
    infer_path: Path = INTERIM_DATA_DIR,
    output_path: Path = EXTERNAL_DATA_DIR,
    gpu: bool = False,
):
    output_path.mkdir(parents=True, exist_ok=True)

    mito_predictor = MitoPredictor(gpu=gpu)
    lipid_predictor = LipidPredictor(gpu=gpu)

    img_paths = sorted(infer_path.glob("*_img.tif"))
    logger.info(f"Number of images found: {len(img_paths)}")

    for img_path in tqdm(img_paths, desc="Inference"):
        img = io.imread(img_path)

        masks_mito = mito_predictor.predict(img)
        stem = img_path.stem.replace("_img", "")
        io.imsave(
            output_path / f"{stem}_mito_masks_pred.tif",
            masks_mito,
            check_contrast=False,
        )
        masks_lipid = lipid_predictor.predict(img)
        io.imsave(
            output_path / f"{stem}_ld_masks_pred.tif",
            masks_lipid,
            check_contrast=False,
        )
        logger.info(f"saved masks for {img_path.name}")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()