import os
import subprocess
import threading
import time
import pandas as pd

def main():

    hws = [256, 512, 768]
    stepss = [10, 50, 100, 200]
    repeats = 3

    dfs = []
    for r in range(repeats):
        for hw in hws:
            for steps in stepss:
                print(f"Running for height/width: {hw}, steps: {steps}")

                # run this script with the following command:
                # python measure_ram.py --prompt "A street spray painted art of a fox riding a rocket to the moon." --size 768 --steps 100 --repetition 0
                cmd = f"python measure_ram.py --prompt \"A street spray painted art of a fox riding a rocket to the moon.\" --size {hw} --steps {steps} --repetition {r}"
                subprocess.run(cmd, shell=True)

                df = pd.read_csv(f"results/timings_{hw}_{steps}_{r}.csv")
                dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv("results/timings.csv", index=False)

if __name__ == "__main__":
    main()