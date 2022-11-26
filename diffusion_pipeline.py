import torch
import tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
import ui_config as conf
import sys
import random

global pipe
global current_model_path
global sampler

def run_pipeline(model_id, sampler_id, prompt, neg_prompt, seed, generate_x_in_parallel, batches, width, height, num_steps, cfg):
    global pipe
    global sampler
    global current_model_path
    if pipe is None:
        print('Wait for the model to load or select a model to load if none are selected.')
        return []

    try:
        pipe.scheduler = sampler.from_pretrained(current_model_path, subfolder="scheduler")
    except:
        pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(current_model_path, subfolder="scheduler")

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
        sampler_id=sampler_id,
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
    global current_model_path
    pipe = None
    if model_id is None:
        print("Cannot load empty model!")
        return
    model_path = f"./content/{model_id}"
    current_model_path = model_path

    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None)
    pipe = pipe.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def switch_sampler(new_sampler):
    global sampler
    all_samplers = get_sampling_strategies()
    out = all_samplers[new_sampler]
    sampler = out

def get_sampling_strategies():
    samplers = {
        "DDIM": DDIMScheduler,
        "Euler": EulerDiscreteScheduler,
        "DPM": DPMSolverMultistepScheduler
    }
    return samplers

def enumerate_samplers():
    samplers = list(get_sampling_strategies().keys())
    return samplers