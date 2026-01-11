#!/usr/bin/env python

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

HOUSE_COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <dataset.csv>")
        sys.exit(1)

    try:
        df = pd.read_csv(sys.argv[1])
    except:
        print("Error: Cannot read dataset")
        return
    print("Dataset loaded successfully")

    nums = [c for c in df.select_dtypes(include="number").columns if c != "Index"]
    n = len(nums)

    colors = df["Hogwarts House"].map(lambda x: HOUSE_COLORS.get(x, "black"))
    _, axes = plt.subplots(n, n, figsize=(n, n))

    for i in range(n):
        for j in range(n):
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")
            axes[i, j].set_yticks([])
            axes[i, j].set_xticks([])

            if j > i:  # skip the upper scatter plots
                axes[i, j].set_frame_on(False)
                continue
            if i == j:
                axes[i, j].hist(df[nums[i]].dropna(), bins=20, alpha=0.6)
            else:
                axes[i, j].scatter(df[nums[j]], df[nums[i]], c=colors, s=20, alpha=0.6)
            if i == n - 1:
                axes[i, j].set_xlabel(nums[j], rotation=0, ha="center")
            if j == 0:
                axes[i, j].set_ylabel(nums[i], rotation=0, ha="right", va="center")
    plt.show()


if __name__ == "__main__":
    main()
