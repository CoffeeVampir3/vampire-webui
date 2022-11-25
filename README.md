# vampire-webui

In development web-ui for running SD 2.0 locally, using diffusers.
![alt text](https://github.com/CoffeeVampir3/vampire-webui/blob/a438783f0f90ec40563e7747938a11e832792d8a/image.png)

# Installation

Installation is still being worked on. If you're encountering errors please open an issue.

Create a conda enviroment using enviroment.yaml:
```
conda env create -f environment.yaml
conda activate ldm
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
