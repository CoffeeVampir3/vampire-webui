from pathlib import Path
from git import Repo

dir_path = Path("./content")
if not dir_path.exists():
    dir_path.mkdir()

print("Cloning stable diffusion 2.0 base")
Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-2-base", dir_path.joinpath("stable-diffusion-2-base"))

print("Cloning stable diffusion 2.0 v-predict")
Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-2", dir_path.joinpath("stable-diffusion-2-base"))