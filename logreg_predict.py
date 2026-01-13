#!/usr/bin/env python

import json
import os
import sys
from pathlib import Path

import pandas as pd

PATH_WEIGHT = Path("save.json")


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

    try:
        with open(PATH_WEIGHT, "r") as file:
            weights = json.load(file)
    except:
        print("Error: Cannot read weights")
        return
    print("Weights loaded successfully")

    df = df.select_dtypes(include="number").drop(columns="Index")


if __name__ == "__main__":
    main()
