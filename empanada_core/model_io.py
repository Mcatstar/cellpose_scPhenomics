import glob
import os
import shutil
from urllib.parse import urlparse
import yaml
from empanada.config_loaders import load_config
import zipfile
import urllib.request
from empanada_core.utils import get_configs, add_new_model
import requests
import warnings


def export_model(model_name, export_path):
    """Exports a model config and its weights to a .empanada zip archive."""
    model_configs = get_configs()
    warnings.filterwarnings("ignore")
    config_yaml = model_configs[model_name]
    config = load_config(config_yaml)

    model_path = config["model"]
    quantized_path = config.get("model_quantized")
    parsed = urlparse(model_path)
    if quantized_path:
        quantized_parsed = urlparse(quantized_path)
        if quantized_parsed.scheme and quantized_parsed.netloc:
            print(f"Downloading quantized model from {quantized_path}")
            loc = quantized_path
            quantized_path = os.path.join(export_path, model_name + "_quantized.pth")

            with open(quantized_path, "wb") as f:
                resp = requests.get(loc, verify=False)
                f.write(resp.content)
        else:
            shutil.copy(quantized_path, export_path)

        new_quantized_path = os.path.basename(quantized_path)

        config["model_quantized"] = new_quantized_path

    if parsed.scheme and parsed.netloc:
        print(f"Downloading model from {model_path}")
        loc = model_path
        model_path = os.path.join(export_path, model_name + ".pth")

        with open(model_path, "wb") as f:
            resp = requests.get(loc, verify=False)
            f.write(resp.content)
    else:
        shutil.copy(model_path, export_path)

    out_path = os.path.join(export_path, model_name + ".empanada")

    new_model_path = os.path.basename(model_path)

    config["model"] = new_model_path
    new_yaml = os.path.join(export_path, model_name + ".yaml")

    with open(new_yaml, "w") as f:
        f.write(yaml.dump(config))

    with zipfile.ZipFile(str(out_path), "w") as f:
        f.write(new_yaml, os.path.basename(new_yaml))
        f.write(os.path.join(export_path, new_model_path), new_model_path)
        if quantized_path:
            f.write(os.path.join(export_path, new_quantized_path), new_quantized_path)

    os.remove(new_yaml)
    os.remove(os.path.join(export_path, new_model_path))

    if quantized_path:
        os.remove(os.path.join(export_path, new_quantized_path))

    print(f"Model exported to {out_path}")
    warnings.filterwarnings("default")


def import_model(model_name, import_path):
    """Imports a .empanada archive into ~/.empanada/models and registers its config."""
    warnings.filterwarnings("ignore")
    tmp_folder = os.path.join(os.path.dirname(import_path), "tmp")

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    with zipfile.ZipFile(import_path, "r") as zip_ref:
        zip_ref.extractall(tmp_folder)

    new_yaml = os.path.join(
        tmp_folder, os.path.basename(import_path).replace(".empanada", ".yaml")
    )
    new_model = os.path.join(
        tmp_folder, os.path.basename(import_path).replace(".empanada", ".pth")
    )
    target_models_folder = os.path.join(os.path.expanduser("~"), ".empanada/models")

    quantized_target_model_path = None
    new_yaml_c = yaml.load(open(new_yaml, "r"), Loader=yaml.FullLoader)
    if new_yaml_c.get("model_quantized"):
        new_model_q = new_yaml_c.get("model_quantized")
        new_model_q_path = os.path.join(tmp_folder, new_model_q)
        shutil.copy(
            new_model_q_path,
            os.path.join(target_models_folder, model_name + "_quantized.pth"),
        )
        quantized_target_model_path = os.path.join(
            target_models_folder, model_name + "_quantized.pth"
        )

    if not os.path.isfile(new_model):
        new_model = glob.glob(os.path.join(tmp_folder, "*.p*"))[0]

    os.makedirs(target_models_folder, exist_ok=True)
    shutil.copy(new_model, os.path.join(target_models_folder, model_name + ".pth"))
    target_models_name = os.path.join(target_models_folder, model_name + ".pth")

    add_new_model(
        model_name,
        new_yaml,
        target_models_name,
        quantized_target_model_path if quantized_target_model_path else False,
    )
    shutil.rmtree(tmp_folder)
    print(f"Model imported to {target_models_name}")
    warnings.filterwarnings("default")


def archive_model(model_name):
    """Moves a model's weights, quantized weights and config into ~/.empanada/archived."""
    model_configs = get_configs()

    empanada_internal_dir = os.path.join(os.path.expanduser("~"), ".empanada")
    configs_dir = os.path.join(empanada_internal_dir, "configs")
    models_dir = os.path.join(empanada_internal_dir, "models")

    archived_folder = os.path.join(empanada_internal_dir, "archived")
    os.makedirs(archived_folder, exist_ok=True)

    # get model paths
    config_yaml = model_configs[model_name]
    config = load_config(config_yaml)
    model_path = config["model"]
    quantized_path = config.get("model_quantized")

    # get model name
    if os.path.exists(model_path):
        shutil.move(
            model_path, os.path.join(archived_folder, os.path.basename(model_path))
        )
    if quantized_path:
        if os.path.exists(quantized_path):
            shutil.move(
                quantized_path,
                os.path.join(archived_folder, os.path.basename(quantized_path)),
            )

    # move config file
    if os.path.exists(config_yaml):
        shutil.move(
            config_yaml, os.path.join(archived_folder, os.path.basename(config_yaml))
        )

    print("Model {} archived, reload to see the changes.".format(model_name))
