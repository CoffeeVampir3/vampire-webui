import torch
from torch import autocast
from pipeline.modified_stable_diffusion import ModifiedDiffusionPipeline
from pipeline.modified_stable_diffusion_img2img import ModifiedDiffusionImg2ImgPipeline
import ui.ui_config as conf
import sys
import random
from PIL import Image
from omegaconf import OmegaConf
from utils.image_utils import encode_and_save_data

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    HeunDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)

global pipe
global current_model_path
global sampler

#Working but considering UI implementation.
"""
#TODO @Z:: Hardcoded path.
from pathlib import Path
import os
def enumerate_embeddings():
    dir_path = Path("./embeddings")
    if not dir_path.exists():
        return None

    files = os.listdir(dir_path)
    return files

def load_embeddings(pipe):
    [test_load_pt_embedding(file, pipe) for (file) in enumerate_embeddings()]

def test_load_pt_embedding(embedding_file, pipe):
    filename = "./embeddings/" + embedding_file
    embedding_name = embedding_file.split('.')[0]
    loaded_embeds = torch.load(filename, map_location="cpu")
    print(loaded_embeds)

    #trick to get the * token out of the embedding
    string_to_token = loaded_embeds['string_to_token']
    string_to_param = loaded_embeds['string_to_param']
    token = list(string_to_token.keys())[0]

    #we cast the embeds to the correct type
    embeds = string_to_param[token]
    dtype = pipe.text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    
    #try to form the embedding token if it's not already existant in the token data
    token = f'<{embedding_name}>'
    repeated_token = [token]
    num_added_tokens = pipe.tokenizer.add_tokens(repeated_token)
    if num_added_tokens == 0:
        return None

    #add the embedding token into the tokenizer and add the embed weights to the text encoder
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(repeated_token)
    embeds = embeds.cuda()
    print(pipe.text_encoder.get_input_embeddings().weight.data.shape)
    pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embeds
"""

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

    config = conf.save_ui_config(
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

    images = []
    with autocast("cuda"):
        for i in range(batches):
            current_seed = nseed+i
            generator.manual_seed(current_seed)
            out = pipe(prompt=multi_prompt, negative_prompt=multi_negative_prompt, height=height, width=width, num_inference_steps=num_steps, guidance_scale=cfg,generator=generator)
            
            images.extend([encode_and_save_data(x, current_seed, index, i, config) for index, x in enumerate(out.images)])
            yield images
            del out
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
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

    pipe = ModifiedDiffusionPipeline.from_pretrained(model_path, safety_checker=None)
    pipe = pipe.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def load_img2img_pipeline_temp(model_id):
    global pipe
    global current_model_path
    pipe = None
    if model_id is None:
        print("Cannot load empty model!")
        return
    model_path = f"./content/{model_id}"
    current_model_path = model_path

    pipe = ModifiedDiffusionPipeline.from_pretrained(model_path, safety_checker=None)
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
        "Euler-A": EulerAncestralDiscreteScheduler,
        "LMS": LMSDiscreteScheduler,
        "PNDM": PNDMScheduler,
        "DPM-Multi": DPMSolverMultistepScheduler,
        "Heun":    HeunDiscreteScheduler,
        "DPM-Single":    DPMSolverSinglestepScheduler,
        "KDPM-2":    KDPM2DiscreteScheduler,
        "KDPM-2A":    KDPM2AncestralDiscreteScheduler
    }
    return samplers

def enumerate_samplers():
    samplers = list(get_sampling_strategies().keys())
    return samplers