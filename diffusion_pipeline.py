import torch
import tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

def testpipeline(pipe, prompt, neg_prompt, seed, num_images, width, height, num_steps, cfg):
    print(pipe)
    generator = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.empty_cache()

    print(f'Prompt: {prompt}, negatives: {neg_prompt}, seed: {seed}, nimages: {num_images}, w: {width}, h: {height}, steps: {num_steps}, cfg: {cfg}')
    multi_prompt = [prompt] * num_images
    multi_negative_prompt = [neg_prompt] * num_images
    with autocast("cuda"):
        out = pipe(prompt=multi_prompt, negative_prompt=multi_negative_prompt, height=height, width=width, num_inference_steps=num_steps, guidance_scale=cfg,generator=generator)
        images = out.images
        del out
    
    return images, pipe

def load_pipeline():
    model_id = "stable-diffusion-2"
    model_path = f"./content/{model_id}"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, scheduler=scheduler)
    pipe = pipe.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return pipe
