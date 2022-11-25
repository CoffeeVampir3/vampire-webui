from omegaconf import OmegaConf
from pathlib import Path

def save_ui_config(**kwargs):
    config = OmegaConf.create(kwargs)
    dir_path = Path("./configs")
    file_name = "last_run.yaml"
    if not dir_path.exists():
        dir_path.mkdir()
    dest = (dir_path/file_name)
    OmegaConf.save(config=config, f=dest)

def load_ui_config():
    dir_path = Path("./configs")
    file_name = "last_run.yaml"
    dest = (dir_path/file_name)

    if not dest.exists():
        return ("banana", "", 1, 1, 1, 768, 768, 20, 7)

    config = OmegaConf.load(dest)
    return list(config.values())