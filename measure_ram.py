import subprocess
import threading
import argparse

# huggingface
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm.auto import trange

import time
import pandas as pd

stop_thread = True

class GPUUsageThread(threading.Thread):
    def __init__(self, process_name):
        super().__init__()
        self.process_name = process_name
        self.process = None
        self.peak_ram = 0
    
    def run(self):
        # Continuously read the output of the subprocess and update peak RAM usage
        while True:
            try:
                output = subprocess.check_output("nvidia-smi --query-compute-apps=process_name,pid,used_memory --format=csv | grep python", shell=True)
                output_str = output.decode("utf-8")
                used_memory = int(output_str.split(",")[2].strip().split()[0])
            except:
                used_memory = 0
                
            # Update the peak RAM usage
            if used_memory > self.peak_ram:
                self.peak_ram = used_memory

            # Sleep for 0.1 seconds
            time.sleep(0.1)

            global stop_thread
            if stop_thread:
                break
    
    def finalize(self):
        print("Finalizing GPU RAM measuring thread")
        return self.peak_ram

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

# parse args and run main
argparser = argparse.ArgumentParser()
argparser.add_argument("--prompt", type=str, default="A street spray painted art of a fox riding a rocket to the moon.")
argparser.add_argument("--size", type=int, default=768)
argparser.add_argument("--steps", type=int, default=100)
argparser.add_argument("--repetition", type=int, default=0)

args = argparser.parse_args()

stop_thread = False
gpu_usage_thread = GPUUsageThread("python")
gpu_usage_thread.start()
df, image = collect_profiler_info(pipe, args.prompt, height=args.size, width=args.size, steps=args.steps)
peak_gpu_ram = gpu_usage_thread.finalize()
print("here")
stop_thread = True
gpu_usage_thread.join()
print("here2")
print(f"Peak GPU RAM usage: {peak_gpu_ram} MB")

df["size"] = args.size
df["steps"] = args.steps
df["peak_gpu_ram"] = peak_gpu_ram
df.to_csv(f"results/timings_{args.size}_{args.steps}_{args.repetition}.csv", index=False)