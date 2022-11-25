import torch
import tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import ui_config as conf
import sys
import random

global pipe

def run_pipeline(model_id, prompt, neg_prompt, seed, generate_x_in_parallel, batches, width, height, num_steps, cfg):
    global pipe
    if pipe is None:
        print('Wait for the model to load or select a model to load if none are selected.')
        return []

    nseed = random.randint(0, (sys.maxsize/64)) if seed == -1 else seed
    generator = torch.Generator("cuda").manual_seed(nseed)

    multi_prompt = [prompt] * generate_x_in_parallel
    multi_negative_prompt = [neg_prompt] * generate_x_in_parallel

    images = []
    with autocast("cuda"):
        for i in range(batches):
            generator = torch.Generator("cuda").manual_seed(nseed + i)
            out = pipe(prompt=multi_prompt, negative_prompt=multi_negative_prompt, height=height, width=width, num_inference_steps=num_steps, guidance_scale=cfg,generator=generator)
            images.extend(out.images)
            del out
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    conf.save_ui_config(
        model_id=model_id,
        prompt=prompt, 
        neg_prompt=neg_prompt, 
        seed=seed, 
        generate_x_in_parallel=generate_x_in_parallel, 
        batches=batches, 
        width=width, 
        height=height, 
        num_steps=num_steps, 
        cfg=cfg)
    return images

def load_pipeline(model_id):
    global pipe
    pipe = None
    if model_id is None:
        print("Cannot load empty model!")
        return
    model_path = f"./content/{model_id}"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, scheduler=scheduler)
    pipe = pipe.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()