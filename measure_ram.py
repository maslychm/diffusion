# huggingface
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm.auto import trange

import time
import pandas as pd

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
num_loops = 1

def makeimage(pipe, prompt, height=768, width=768, steps=100):
    prompt = "<s>Prompt: " + prompt + ","
    for _ in trange(num_loops):
        image = pipe(prompt, 
                    num_inference_steps=steps, 
                    height=height, width=width,
                    guidance_scale=guidance_scale).images[0]  
        image.save("output.png")

    return image

def run_profiler(pipe, prompt, height=768, width=768, steps=100):
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
            with record_function("model_inference"):
                start_time = time.perf_counter()
                image = makeimage(pipe, prompt, height=height, width=width, steps=steps)
                final_time = time.perf_counter()

    total_time = final_time - start_time
    profiler_results = prof.key_averages()

    print(profiler_results.table(sort_by="cpu_time_total", row_limit=10))

    return profiler_results, total_time, image

def extract_timings(profiler_results):
    rows = []
    for summary in profiler_results:
        if summary.key != "model_inference":
            continue
        row = {
            "name": summary.key,
            "cpu_time_total": summary.cpu_time_total,
            "cuda_time_total": summary.cuda_time_total,
            "self_cpu_time_total": summary.self_cpu_time_total,
            "self_cuda_time_total": summary.self_cuda_time_total,
            "cpu_memory_usage": summary.cpu_memory_usage,
            "cuda_memory_usage": summary.cuda_memory_usage,
            "self_cpu_memory_usage": summary.self_cpu_memory_usage,
            "self_cuda_memory_usage": summary.self_cuda_memory_usage,
            "cpu_time": summary.cpu_time,
            "cuda_time": summary.cuda_time,
            "count": summary.count
        }
        rows.append(row)
    return pd.DataFrame(rows)

def collect_profiler_info(pipe, prompt, height=768, width=768, steps=100):
    profiler_results, total_time, image = run_profiler(pipe, prompt, height=height, width=width, steps=steps)
    timings_df = extract_timings(profiler_results)
    timings_df["total_time"] = total_time
    return timings_df, image


hws = [256, 512, 768]
stepss = [10, 50, 100, 200]
repeats = 3

dfs = []
for r in range(repeats):
    for hw in hws:
        for steps in stepss:
            print(f"Running for height/width: {hw}, steps: {steps}")
            df, image = collect_profiler_info(pipe, "A street spray painted art of a fox riding a rocket to the moon.", height=hw, width=hw, steps=steps)
            df["size"] = hw
            df["steps"] = steps
            df.to_csv(f"results/timings_{hw}_{steps}_{r}.csv", index=False)
            dfs.append(df)

df = pd.concat(dfs)
df.to_csv("results/timings.csv", index=False)

# =============================================================================
# Plots
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("results/timings.csv")
df["Image size"] = df["size"].astype(str) + "x" + df["size"].astype(str)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.barplot(data=df, x="steps", y="total_time", hue="Image size", ax=ax)
ax.set_title("Diffusion time VS num. steps and image size")
ax.set_xlabel("Num. steps")
ax.set_ylabel("Diffusion time (s)")
fig.tight_layout()
fig.savefig("results/diffusion_time_vs_steps_and_size.png", dpi=300)
