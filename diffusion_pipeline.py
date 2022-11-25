import torch
import tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

global pipe

def testpipeline(prompt, neg_prompt, seed, generate_x_in_parallel, batches, width, height, num_steps, cfg):
    global pipe
    generator = torch.Generator("cuda").manual_seed(seed)
    print(f'Prompt: {prompt}, negatives: {neg_prompt}, seed: {seed}, nimages: {generate_x_in_parallel}, w: {width}, h: {height}, steps: {num_steps}, cfg: {cfg}')
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