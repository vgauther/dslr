#!/usr/bin/env python

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

FEATURES = ["Astronomy", "Defense Against the Dark Arts"]

HOUSE_COLORS = {
    "Ravenclaw": "blue",
    "Slytherin": "green",
    "Gryffindor": "red",
    "Hufflepuff": "yellow",
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

    colors = df["Hogwarts House"].map(lambda x: HOUSE_COLORS.get(x, "gray"))
    f1, f2 = FEATURES
    plt.figure()
    plt.scatter(df[f1], df[f2], alpha=0.6, c=colors)
    plt.title(f"{f1} vs {f2}")
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()


if __name__ == "__main__":
    main()
