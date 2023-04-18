"""
This script will iterate over combinations of parameters and save results to a CSV.
To run: `python start_fox.py`
"""

import subprocess
import pandas as pd

def main():

    # Params
    hws = [256, 512, 768, 2014]
    stepss = [10, 50, 100, 200]
    repeats = 3

    print("Total number of runs: ", runs:=len(hws) * len(stepss) * repeats)
    run = 0
    dfs = []
    for r in range(repeats):
        for hw in hws:
            for steps in stepss:
                run += 1
                print("\n=====================================")
                print(f"Run {run}/{runs}", f"height/width: {hw}, steps: {steps}")

                # run this script with the following command:
                # python measure_ram.py --prompt "A street spray painted art of a fox riding a rocket to the moon." --size 768 --steps 100 --repetition 0
                cmd = f"python measure_ram.py --prompt \"A street spray painted art of a fox riding a rocket to the moon.\" --size {hw} --steps {steps} --repetition {r}"
                subprocess.run(cmd, shell=True)

                df = pd.read_csv(f"results/timings_{hw}_{steps}_{r}.csv")
                dfs.append(df)

    df = pd.concat(dfs)
    # collect all the results in a single csv file
    df.to_csv("results/timings.csv", index=False)

if __name__ == "__main__":
    main()