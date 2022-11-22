import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

def no_nsfw_filter(images, clip_input):
    return images, False

pipe = StableDiffusionPipeline.from_pretrained("./content/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

pipe.safety_checker = no_nsfw_filter

num_images = 1
prompt = ["Woman riding a horse, moonlight, oil painting on canvas"] * num_images

torch.cuda.empty_cache()
torch.cuda.synchronize()
height = 512
width = 512
num_inference_steps = 30
guidance_scale = 7.5
with autocast("cuda"):
    out = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    images = out.images
    del out
    
for i, img in enumerate(images):
	img.save("test"+str(i)+".png")
