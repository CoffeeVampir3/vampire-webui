import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

def no_nsfw_filter(images, clip_input):
    return images, False

def testpipeline(prompt, num_images, width, height, num_steps, cfg):
    pipe = StableDiffusionPipeline.from_pretrained("./content/stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    pipe.safety_checker = no_nsfw_filter

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f'Prompt: {prompt}, nimages: {num_images}, w: {width}, h: {height}, steps: {num_steps}, cfg: {cfg}')

    multi_prompt = [prompt] * num_images
    with autocast("cuda"):
        out = pipe(multi_prompt, height=height, width=width, num_inference_steps=num_steps, guidance_scale=cfg)
        images = out.images
        del out
    
    return images