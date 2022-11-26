# vampire-webui

In development web-ui for running SD 2.0 locally, using diffusers.
![alt text](https://github.com/CoffeeVampir3/vampire-webui/blob/96733e8763bcb02b975687a050337482dcd5fb2c/image.png)

# Usage

In the main directory for windows:

```python
conda activate ldm
python webui.py
```

Linux:

```shell
. launch.sh
```

# Installation

Installation is still being worked on. If you're encountering errors please open an issue.

Create a conda enviroment using enviroment.yaml:
```
conda env create -f environment.yaml
conda activate ldm
```

Install torch and dependencies
```
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch
```

Install transformers and dependencies
```
pip install transformers==4.19.2
pip install -e .
```

Run initial_setup.py, this will clone the huggingface diffusers for SD 2.0 into a new folder called content
```
python initial_setup.py
```

Recommended to follow these repos installation instructions for xformers <https://github.com/Stability-AI/stablediffusion>
And if you have problems installing diffusers, see <https://github.com/huggingface/diffusers>

This is still a work in progress so please report any problems.
