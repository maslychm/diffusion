# huggingface
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm.auto import trange

text_to_im_model_id = "stabilityai/stable-diffusion-2"

scheduler = EulerDiscreteScheduler.from_pretrained(text_to_im_model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(text_to_im_model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

if text_to_im_model_id=="stability/stable-diffusion-2":
    pipe.enable_attention_slicing()

# image gen params
temperature = 0.7
max_length = 42
guidance_scale = 7.5
height = 768
width = 768
steps = 100
num_loops = 1

def makeimage(pipe, prompt):
    prompt = "<s>Prompt: " + prompt + ","
    for _ in trange(num_loops):
        image = pipe(prompt, 
                    num_inference_steps=steps, 
                    height=height, width=width,
                    guidance_scale=guidance_scale).images[0]  
        image.save("output.png")

    return image

prompt = 'A man in space'

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
    with record_function("model_inference"):
        image = makeimage(pipe, prompt)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))