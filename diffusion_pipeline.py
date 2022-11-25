import torch
import tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import ui_config as conf

global pipe

def run_pipeline(prompt, neg_prompt, seed, generate_x_in_parallel, batches, width, height, num_steps, cfg):
    conf.save_ui_config(
        prompt=prompt, 
        neg_prompt=neg_prompt, 
        seed=seed, 
        generate_x_in_parallel=generate_x_in_parallel, 
        batches=batches, 
        width=width, 
        height=height, 
        num_steps=num_steps, 
        cfg=cfg)
    return []
    global pipe
    generator = torch.Generator("cuda").manual_seed(seed)

    multi_prompt = [prompt] * generate_x_in_parallel
    multi_negative_prompt = [neg_prompt] * generate_x_in_parallel

    images = []
    with autocast("cuda"):
        for i in range(batches):
            generator = torch.Generator("cuda").manual_seed(seed + i)
            out = pipe(prompt=multi_prompt, negative_prompt=multi_negative_prompt, height=height, width=width, num_inference_steps=num_steps, guidance_scale=cfg,generator=generator)
            images.extend(out.images)
            del out
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return images

def load_pipeline():
    model_id = "stable-diffusion-2"
    model_path = f"./content/{model_id}"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, scheduler=scheduler)
    pipe = pipe.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()