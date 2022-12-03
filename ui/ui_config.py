from omegaconf import OmegaConf
from pathlib import Path
from os import walk
from gradio import skip

def save_ui_config(**kwargs):
    config = OmegaConf.create(kwargs)
    dir_path = Path("./configs")
    file_name = "last_run.yaml"
    if not dir_path.exists():
        dir_path.mkdir()
    dest = (dir_path/file_name)
    OmegaConf.save(config=config, f=dest)
    return config

def load_ui_config(model_dropdown):
    dir_path = Path("./configs")
    file_name = "last_run.yaml"
    dest = (dir_path/file_name)

    if not dest.exists():
        return {model_dropdown: skip()}

    config = OmegaConf.load(dest)
    return list(config.values())

def enumerate_models():
    dir_path = Path("./content")
    if not dir_path.exists():
        return None

    dirs = next(walk(dir_path))[1]
    return dirs