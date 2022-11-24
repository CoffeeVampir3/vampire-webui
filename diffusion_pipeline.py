import torch
import tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline



def no_nsfw_filter(images, clip_input):
    return images, False

def testpipeline(prompt, neg_prompt, num_images, width, height, num_steps, cfg):
    pipe = StableDiffusionPipeline.from_pretrained("./content/stable-diffusion-2-base", safety_checker=None)
    print(pipe)
    pipe = pipe.to("cuda")

    print(pipe)

    generator = torch.Generator("cuda").manual_seed(1024)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f'Prompt: {prompt}, negatives: {neg_prompt}, nimages: {num_images}, w: {width}, h: {height}, steps: {num_steps}, cfg: {cfg}')
    multi_prompt = [prompt] * num_images
    multi_negative_prompt = [neg_prompt] * num_images
    with autocast("cuda"):
        out = pipe(prompt=multi_prompt, negative_prompt=multi_negative_prompt, height=height, width=width, num_inference_steps=num_steps, guidance_scale=cfg,generator=generator)
        images = out.images
        del out
    
    return images
