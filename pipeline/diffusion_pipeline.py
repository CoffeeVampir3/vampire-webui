import torch
from torch import autocast
from pipeline.modified_stable_diffusion import ModifiedDiffusionPipeline
import ui.ui_config as conf
import sys
import random
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

global pipe
global current_model_path
global sampler

#Working but considering UI implementation.
"""
def test_load_pt_embedding(embedding_name, pipe):
    #Load an embedding
    embedding_path = embedding_name + ".pt"
    loaded_embeds = torch.load(embedding_path, map_location="cpu")
    print(loaded_embeds)
    print(loaded_embeds.keys())

    #trick to get the * token out of the embedding
    string_to_token = loaded_embeds['string_to_token']
    string_to_param = loaded_embeds['string_to_param']
    token = list(string_to_token.keys())[0]
    print(f'got key{token}')

    #we cast the embeds to the correct type
    embeds = string_to_param[token]
    dtype = pipe.text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    
    #try to form the embedding token if it's not already existant in the token data
    token = '<art by {embedding_name}>'
    num_added_tokens = pipe.tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        return None

    #add the embedding token into the tokenizer and add the embed weights to the text encoder
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
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
            
            images.extend([encode_exif_data(x, current_seed, index, i, config) for index, x in enumerate(out.images)])
            yield images
            del out
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return images

def encode_exif_data(image, seed, image_num, batch_number, config):
    dir_path = Path(f"./outputs/{config.prompt}")
    file_name = f"b{batch_number}-s{seed+image_num}.png"
    if not dir_path.exists():
        dir_path.mkdir()
    dest = (dir_path/file_name)

    #unicode
    prefix = bytes.fromhex('554E49434F444500')
    content = ""
    #hacky but whatever
    config.seed = seed
    config.generate_x_in_parallel = image_num
    config.batches = batch_number 
    for cf, cv in zip(list(config), list(config.values())):
        content += cf + ": " + str(cv) + "\n"

    exif_data = image.getexif()
    user_comment_tiff = 0x9286
    exif_data[user_comment_tiff] = prefix + content.encode('utf-16le')
    image.save(dest, exif=exif_data)
    return image

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
        "DPM": DPMSolverMultistepScheduler
    }
    return samplers


def enumerate_samplers():
    samplers = list(get_sampling_strategies().keys())
    return samplers