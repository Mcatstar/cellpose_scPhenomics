import os
import time
import yaml
import platform
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

from empanada import losses
from empanada import data
from empanada import metrics
from empanada import models
from empanada.inference import engines
from empanada.config_loaders import load_config
from empanada.data.utils.transforms import FactorPad
from empanada.models import quantization as quant_models

from empanada_core.utils import get_configs, abspath, add_new_model

schedules = sorted(
    name
    for name in lr_scheduler.__dict__
    if callable(lr_scheduler.__dict__[name])
    and not name.startswith("__")
    and name[0].isupper()
)

optimizers = sorted(
    name
    for name in optim.__dict__
    if callable(optim.__dict__[name])
    and not name.startswith("__")
    and name[0].isupper()
)

augmentations = sorted(
    name
    for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith("__") and name[0].isupper()
)

datasets = sorted(name for name in data.__dict__ if callable(data.__dict__[name]))

engine_names = sorted(
    name for name in engines.__dict__ if callable(engines.__dict__[name])
)

loss_names = sorted(name for name in losses.__dict__ if callable(losses.__dict__[name]))


def main(config):
    # create model directory if None
    if not os.path.isdir(config["TRAIN"]["model_dir"]):
        os.mkdir(config["TRAIN"]["model_dir"])

    # validate parameters
    assert config["TRAIN"]["lr_schedule"] in schedules
    assert config["TRAIN"]["optimizer"] in optimizers
    assert config["TRAIN"]["criterion"] in loss_names
    assert config["EVAL"]["engine"] in engine_names

    return main_worker(config)


def main_worker(config):
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if str(config["device"]) == "cpu":
        print(f"Using CPU for training.")
    else:
        print(f"Using GPU for training.")

    if platform.system() == "Darwin":
        try:
            # force=True: empanada_core/__init__.py already sets this at import
            # time, but re-assert it here in case something else locked in a
            # different context first.
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # setup the model and pick dataset class
    model_arch = config["MODEL"]["arch"]
    model = models.__dict__[model_arch](**config["MODEL"])
    dataset_class_name = config["TRAIN"]["dataset_class"]
    data_cls = data.__dict__[dataset_class_name]

    # load pre-trained weights, if using
    if config["TRAIN"]["encoder_pretraining"] is not None:
        state = torch.hub.load_state_dict_from_url(
            config["TRAIN"]["encoder_pretraining"], map_location="cpu"
        )
        state_dict = state["state_dict"]

        # add the prefix 'encoder' to all of the keys
        for k in list(state_dict.keys()):
            if not k.startswith("fc"):
                state_dict["encoder." + k] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state["state_dict"], strict=False)
        norms = {}
        norms["mean"] = state["norms"][0]
        norms["std"] = state["norms"][1]
    else:
        norms = config["DATASET"]["norms"]

    if norms is None:
        print("Calculating dataset norms...")
        impaths = glob(os.path.join(config["TRAIN"]["train_dir"], "**/images/*"))
        means = []
        stds = []
        for imp in tqdm(impaths):
            img = io.imread(imp)
            assert img.dtype == np.uint8
            means.append(img.mean())
            stds.append(img.std())

        norms = {"mean": np.mean(means).item() / 255, "std": np.mean(stds).item() / 255}
        print("Norms:", norms)

    finetune_layer = config["TRAIN"]["finetune_layer"]
    # start by freezing all encoder parameters
    for pname, param in model.named_parameters():
        if "encoder" in pname:
            param.requires_grad = False

    # freeze encoder layers
    if finetune_layer == "none":
        # leave all encoder layers frozen
        pass
    elif finetune_layer == "all":
        # unfreeze all encoder parameters
        for pname, param in model.named_parameters():
            if "encoder" in pname:
                param.requires_grad = True
    else:
        valid_layers = ["stage1", "stage2", "stage3", "stage4"]
        assert finetune_layer in valid_layers
        # unfreeze all layers from finetune_layer onward
        for layer_name in valid_layers[valid_layers.index(finetune_layer) :]:
            # freeze all encoder parameters
            for pname, param in model.named_parameters():
                if f"encoder.{layer_name}" in pname:
                    param.requires_grad = True

    num_trainable = sum(
        p[1].numel() for p in model.named_parameters() if p[1].requires_grad
    )
    print(f"Model with {num_trainable} trainable parameters.")

    model = model.to(config["device"])
    cudnn.benchmark = True

    # set the training image augmentations
    config["aug_string"] = []
    dataset_augs = []
    for aug_params in config["TRAIN"]["augmentations"]:
        aug_name = aug_params["aug"]

        assert aug_name in augmentations, (
            f"{aug_name} is not a valid albumentations augmentation!"
        )

        config["aug_string"].append(aug_params["aug"])
        del aug_params["aug"]
        dataset_augs.append(A.__dict__[aug_name](**aug_params))

    tfs = A.Compose([*dataset_augs, A.Normalize(**norms), ToTensorV2()])

    # create training dataset and loader
    train_dataset = data_cls(
        config["TRAIN"]["train_dir"],
        transforms=tfs,
        **config["TRAIN"]["dataset_params"],
    )
    if config["TRAIN"]["additional_train_dirs"] is not None:
        for train_dir in config["TRAIN"]["additional_train_dirs"]:
            add_dataset = data_cls(
                train_dir, transforms=tfs, **config["TRAIN"]["dataset_params"]
            )
            train_dataset = train_dataset + add_dataset

    if config["TRAIN"]["dataset_params"]["weight_gamma"] is not None:
        train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    else:
        train_sampler = None

    # num workers always less than number of batches in train dataset
    num_workers = min(
        config["TRAIN"]["workers"], len(train_dataset) // config["TRAIN"]["batch_size"]
    )
    if platform.system() == "Darwin":
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=config["TRAIN"]["workers"],
        pin_memory=torch.cuda.is_available(),
        sampler=train_sampler,
        drop_last=True,
    )

    if config["EVAL"]["eval_dir"] is not None:
        eval_tfs = A.Compose(
            [
                FactorPad(128),  # pad image to be divisible by 128
                A.Normalize(**norms),
                ToTensorV2(),
            ]
        )
        eval_dataset = data_cls(
            config["EVAL"]["eval_dir"],
            transforms=eval_tfs,
            **config["TRAIN"]["dataset_params"],
        )
        # evaluation runs on a single gpu
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=config["TRAIN"]["workers"],
        )
    else:
        eval_loader = None

    # set criterion
    criterion_name = config["TRAIN"]["criterion"]
    criterion = losses.__dict__[criterion_name](
        **config["TRAIN"]["criterion_params"]
    ).to(config["device"])

    # set optimizer and lr scheduler
    opt_name = config["TRAIN"]["optimizer"]
    opt_params = config["TRAIN"]["optimizer_params"]
    optimizer = configure_optimizer(model, opt_name, **opt_params)

    schedule_name = config["TRAIN"]["lr_schedule"]
    schedule_params = config["TRAIN"]["schedule_params"]

    if "steps_per_epoch" in schedule_params:
        n_steps = schedule_params["steps_per_epoch"]
        if n_steps != len(train_loader):
            schedule_params["steps_per_epoch"] = len(train_loader)
            print(f"Steps per epoch adjusted from {n_steps} to {len(train_loader)}")

    scheduler = lr_scheduler.__dict__[schedule_name](optimizer, **schedule_params)
    scaler = GradScaler() if config["TRAIN"]["amp"] else None

    # training and evaluation loop
    if "epochs" in config["TRAIN"]["schedule_params"]:
        epochs = config["TRAIN"]["schedule_params"]["epochs"]
    elif "epochs" in config["TRAIN"]:
        epochs = config["TRAIN"]["epochs"]
    else:
        raise Exception("Number of training epochs not defined!")

    config["TRAIN"]["epochs"] = epochs

    for epoch in range(epochs):
        # train for one epoch
        train(
            train_loader, model, criterion, optimizer, scheduler, scaler, epoch, config
        )

        # evaluate on validation set
        is_val_epoch = (epoch + 1) % config["EVAL"]["epochs_per_eval"] == 0
        is_last_epoch = (epoch + 1) % epochs == 0
        if eval_loader is not None and (is_val_epoch or is_last_epoch):
            validate(eval_loader, model, criterion, epoch, config)

        save_now = (epoch + 1) % config["TRAIN"]["save_freq"] == 0
        if save_now:
            torch.save(
                {
                    "arch": config["MODEL"]["arch"],
                    "state_dict": model.state_dict(),
                    "norms": norms,
                },
                os.path.join(
                    config["TRAIN"]["model_dir"],
                    f"{config['model_name']}_checkpoint.pth.tar",
                ),
            )

    return config


def configure_optimizer(model, opt_name, **opt_params):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.

    Only BatchNorm parameters and biases are excluded.
    """

    # easy if there's no weight_decay
    if "weight_decay" not in opt_params:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)
    elif opt_params["weight_decay"] == 0:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)

    # otherwise separate parameters into two groups
    decay = set()
    no_decay = set()

    blacklist = (torch.nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = "%s.%s" % (mn, pn) if mn else pn

            if full_name.endswith("bias"):
                no_decay.add(full_name)
            elif full_name.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "Overlapping decay and no decay"
    assert len(param_dict.keys() - union_params) == 0, "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    param_groups = [
        {"params": decay_params, **opt_params},
        {"params": no_decay_params, **opt_params},
    ]
    param_groups[1]["weight_decay"] = 0  # overwrite default to 0 for no_decay group

    return optim.__dict__[opt_name](param_groups, **opt_params)


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, config):
    # generic progress
    batch_time = ProgressAverageMeter("Time", ":6.3f")
    data_time = ProgressAverageMeter("Data", ":6.3f")
    loss_meters = None

    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time], prefix="Epoch: [{}]".format(epoch)
    )

    # end of epoch metrics
    class_names = config["DATASET"]["class_names"]
    metric_dict = {}
    for metric_params in config["TRAIN"]["metrics"]:
        reg_name = metric_params["name"]
        metric_name = metric_params["metric"]
        metric_params = {
            k: v for k, v in metric_params.items() if k not in ["name", "metric"]
        }
        metric_dict[reg_name] = metrics.__dict__[metric_name](
            metrics.EMAMeter, **metric_params
        )

    meters = metrics.ComposeMetrics(metric_dict, class_names)

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = batch["image"]
        target = {k: v for k, v in batch.items() if k not in ["image", "fname"]}

        images = images.to(config["device"], non_blocking=True)
        target = {
            k: tensor.to(config["device"], non_blocking=True)
            for k, tensor in target.items()
        }

        # zero grad before running
        optimizer.zero_grad()

        # compute output
        if scaler is not None:
            with autocast():
                output = model(images)
                loss, aux_loss = criterion(
                    output, target
                )  # output and target are both dicts

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)
            loss, aux_loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # update the LR
        scheduler.step()

        # record losses
        if loss_meters is None:
            loss_meters = {}
            for k, v in aux_loss.items():
                loss_meters[k] = ProgressEMAMeter(k, ":.4e")
                loss_meters[k].update(v)
                # add to progress
                progress.meters.append(loss_meters[k])
        else:
            for k, v in aux_loss.items():
                loss_meters[k].update(v)

        # calculate human-readable per epoch metrics
        with torch.no_grad():
            meters.evaluate(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["TRAIN"]["print_freq"] == 0:
            progress.display(i)

    # end of epoch print evaluation metrics
    print("\n")
    print(f"Epoch {epoch} training metrics:")
    meters.display()


def validate(eval_loader, model, criterion, epoch, config):
    # validation metrics to track
    class_names = config["DATASET"]["class_names"]
    metric_dict = {}
    for metric_params in config["EVAL"]["metrics"]:
        reg_name = metric_params["name"]
        metric_name = metric_params["metric"]
        metric_params = {
            k: v for k, v in metric_params.items() if k not in ["name", "metric"]
        }
        metric_dict[reg_name] = metrics.__dict__[metric_name](
            metrics.AverageMeter, **metric_params
        )

    meters = metrics.ComposeMetrics(metric_dict, class_names)

    # validation tracking
    batch_time = ProgressAverageMeter("Time", ":6.3f")
    loss_meters = None

    progress = ProgressMeter(len(eval_loader), [batch_time], prefix="Validation: ")

    # create the Inference Engine
    engine_name = config["EVAL"]["engine"]
    engine = engines.__dict__[engine_name](model, **config["EVAL"]["engine_params"])

    for i, batch in enumerate(eval_loader):
        end = time.time()
        images = batch["image"]
        target = {k: v for k, v in batch.items() if k not in ["image", "fname"]}

        images = images.to(config["device"], non_blocking=True)
        target = {
            k: tensor.to(config["device"], non_blocking=True)
            for k, tensor in target.items()
        }

        # compute panoptic segmentations
        # from prediction and ground truth
        output = engine.infer(images)
        semantic = engine._harden_seg(output["sem"])
        output["pan_seg"] = engine.postprocess(
            semantic, output["ctr_hmp"], output["offsets"]
        )
        target["pan_seg"] = engine.postprocess(
            target["sem"].unsqueeze(1), target["ctr_hmp"], target["offsets"]
        )

        loss, aux_loss = criterion(output, target)

        # record losses
        if loss_meters is None:
            loss_meters = {}
            for k, v in aux_loss.items():
                loss_meters[k] = ProgressAverageMeter(k, ":.4e")
                loss_meters[k].update(v)
                # add to progress
                progress.meters.append(loss_meters[k])
        else:
            for k, v in aux_loss.items():
                loss_meters[k].update(v)

        # compute metrics
        with torch.no_grad():
            meters.evaluate(output, target)

        batch_time.update(time.time() - end)

        if i % config["TRAIN"]["print_freq"] == 0:
            progress.display(i)

    # end of epoch print evaluation metrics
    print("\n")
    print(f"Validation results:")
    meters.display()


class ProgressAverageMeter(metrics.AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        super().__init__()

    def __str__(self):
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressEMAMeter(metrics.EMAMeter):
    """Computes and stores the exponential moving average and current value"""

    def __init__(self, name, fmt=":f", momentum=0.98):
        self.name = name
        self.fmt = fmt
        super().__init__(momentum)

    def __str__(self):
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


MAIN_CONFIG = abspath(__file__, "training/train_config.yaml")
CEM_WEIGHTS = "https://zenodo.org/record/6453160/files/cem1.5m_swav_resnet50_200ep_balanced.pth.tar?download=1"
MODEL_CONFIGS = {
    "PanopticDeepLab": abspath(
        __file__ if "__file__" in globals() else ".", "training/pdl_model.yaml"
    ),
    "PanopticBiFPN": abspath(
        __file__ if "__file__" in globals() else ".", "training/bifpn_model.yaml"
    ),
}


def build_train_config(
    model_name,
    train_dir,
    eval_dir,
    model_dir,
    label_text,
    label_divisor,
    model_arch,
    use_cem,
    finetune_layer,
    iterations,
    patch_size,
    custom_config,
    description,
):
    """Assemble a training config dict from the same parameters the napari
    widget collected. Returns the config passed to main().
    """
    train_dir = str(train_dir)
    model_dir = str(model_dir)

    if str(eval_dir) == ".":
        eval_dir = None

    assert os.path.isdir(train_dir)
    if model_arch == "PanopticBiFPN":
        assert patch_size % 128 == 0, (
            "Patch size must be divisible by 128 to use PanopticBiFPN!"
        )

    # extract class_names, labels, and thing list
    class_names = {}
    thing_list = []
    for seg_class in label_text.split("\n"):
        class_id, class_name, class_kind = seg_class.split(",")
        class_id = class_id.strip()
        class_name = class_name.strip()
        class_kind = class_kind.strip()

        assert class_kind in ["semantic", "instance"], (
            "Class kind must be 'semantic' or 'instance'"
        )
        class_names[int(class_id)] = class_name
        if class_kind == "instance":
            thing_list.append(int(class_id))

    custom_config = str(custom_config)
    if custom_config != "default config":
        assert os.path.isfile(custom_config)
        config = load_config(custom_config)
    else:
        config = load_config(MAIN_CONFIG)

    # load the model config if needed
    if "MODEL" not in config:
        config["MODEL"] = load_config(MODEL_CONFIGS[model_arch])

    n_classes = len(class_names)
    config["MODEL"]["num_classes"] = n_classes + 1 if n_classes > 1 else 1

    config["DATASET"]["class_names"] = class_names
    config["DATASET"]["labels"] = list(class_names.keys())
    config["DATASET"]["thing_list"] = thing_list
    config["EVAL"]["engine_params"]["thing_list"] = thing_list

    config["description"] = description

    # training on mac breaks with more than 1 data worker
    if platform.system() == "Darwin":
        config["TRAIN"]["workers"] = 0

    config["model_name"] = model_name
    config["TRAIN"]["train_dir"] = train_dir
    config["TRAIN"]["model_dir"] = model_dir
    config["TRAIN"]["finetune_layer"] = finetune_layer if use_cem else "all"
    config["TRAIN"]["encoder_pretraining"] = CEM_WEIGHTS if use_cem else None

    # turn off the instance decoder if all semantic channels
    if not thing_list:
        config["MODEL"]["ins_decoder"] = False

    if n_classes == 1:
        config["TRAIN"]["dataset_class"] = "SingleClassInstanceDataset"
    else:
        config["TRAIN"]["dataset_class"] = "PanopticDataset"
        config["TRAIN"]["dataset_params"]["labels"] = config["DATASET"]["labels"]
        config["TRAIN"]["dataset_params"]["thing_list"] = config["DATASET"][
            "thing_list"
        ]
        config["TRAIN"]["dataset_params"]["label_divisor"] = int(label_divisor)

    # get number of images in train_dir
    n_imgs = len(glob(os.path.join(train_dir, "**/images/*")))
    bsz = config["TRAIN"]["batch_size"]
    if not n_imgs:
        raise Exception(f"No images found in {os.path.join(train_dir, '**/images/*')}")
    elif n_imgs < bsz:
        raise Exception(f"Need {bsz} images for batch size {bsz}, got {n_imgs}.")
    else:
        epochs = int(iterations // (n_imgs // bsz))

    print(f"Found {n_imgs} images for training. Training for {epochs} epochs.")

    # update the patch size in augmentations if parameters are null
    for aug in config["TRAIN"]["augmentations"]:
        for k in aug.keys():
            aug[k] = (
                patch_size
                if ("height" in k or "width" in k) and aug.get(k) is None
                else aug[k]
            )

    config["TRAIN"]["save_freq"] = max(1, epochs // 5)
    config["EVAL"]["eval_dir"] = eval_dir
    config["EVAL"]["epochs_per_eval"] = max(1, epochs // 5)

    if "epochs" in config["TRAIN"]["schedule_params"]:
        config["TRAIN"]["schedule_params"]["epochs"] = epochs
    else:
        config["TRAIN"]["epochs"] = epochs

    # fill labels to track for metrics
    for metric in config["TRAIN"]["metrics"]:
        if metric["metric"] in ["IoU", "PQ"]:
            metric["labels"] = config["DATASET"]["labels"]
        elif metric["metric"] in ["F1"]:
            metric["labels"] = config["DATASET"]["thing_list"]
        else:
            raise Exception("Unsupported metric", metric["metric"])

    for metric in config["EVAL"]["metrics"]:
        if metric["metric"] in ["IoU", "PQ"]:
            metric["labels"] = config["DATASET"]["labels"]
        elif metric["metric"] in ["F1"]:
            metric["labels"] = config["DATASET"]["thing_list"]
        else:
            raise Exception("Unsupported metric", metric["metric"])

    return config


def run_training(config):
    """Train from a config built by build_train_config, then script and export
    the model (TorchScript .pth + describing .yaml). Returns the yaml path.
    """
    config = main(config)

    outpath = os.path.join(
        config["TRAIN"]["model_dir"], config["model_name"] + "_checkpoint.pth.tar"
    )
    print(f"Finished model training. Model weights saved to {outpath}")

    print(f"Scripting and exporting model...")
    model_arch = config["MODEL"]["arch"]
    quant_arch = "Quantizable" + model_arch

    # load the state
    state = torch.load(outpath, map_location="cpu")
    norms = state["norms"]
    state_dict = state["state_dict"]

    # remove module. prefix from state_dict keys, if needed
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[len("module.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model = quant_models.__dict__[quant_arch](**config["MODEL"], quantize=False)

    # prep the model
    model.load_state_dict(state_dict)
    model.eval()
    model.fuse_model()
    if torch.cuda.is_available():
        model.cuda()

    model = torch.jit.script(model)

    print("Model scripted successfully.")

    model_out = os.path.join(
        config["TRAIN"]["model_dir"], f"{model_arch}_{config['model_name']}.pth"
    )
    torch.jit.save(model, model_out)
    print("Exported model successfully.")

    # export a yaml file describing the models
    finetune_params = {
        "dataset_class": config["TRAIN"]["dataset_class"],
        "dataset_params": config["TRAIN"]["dataset_params"],
        "criterion": config["TRAIN"]["criterion"],
        "criterion_params": config["TRAIN"]["criterion_params"],
        "engine": config["EVAL"]["engine"],
        "engine_params": config["EVAL"]["engine_params"],
    }
    desc = {
        "model": model_out,
        "model_quantized": None,
        "norms": {"mean": norms["mean"], "std": norms["std"]},
        "padding_factor": 128,
        "thing_list": config["DATASET"]["thing_list"],
        "labels": config["DATASET"]["labels"],
        "class_names": config["DATASET"]["class_names"],
        "description": config["description"],
        "FINETUNE": finetune_params,
    }

    export_config_path = os.path.join(
        config["TRAIN"]["model_dir"], f"{model_arch}_{config['model_name']}.yaml"
    )
    with open(export_config_path, mode="w") as f:
        yaml.dump(desc, f)

    return export_config_path


def train_and_register(config):
    """run_training followed by registering the exported model as a new model."""
    export_config_path = run_training(config)
    add_new_model(config["model_name"], export_config_path)
    print(f"Registered new model {config['model_name']}")

    return export_config_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m empanada_core.train <config.yaml>")

    main(load_config(sys.argv[1]))
