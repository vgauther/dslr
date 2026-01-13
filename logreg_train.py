#!/usr/bin/env python


import json
import os
import sys

import numpy as np
import pandas as pd

from dslr import (
    EPOCHS,
    EXPECTED_COLUMNS,
    HOUSE_TO_LABEL,
    IGNORED_COLUMNS,
    LEARNING_RATE,
    SUBJECTS,
)


def replace_nan_by_mean(df):
    for col in df:
        values = df[col].tolist()

        total = 0
        count = 0

        i = 0
        while i < len(values):
            v = values[i]
            if v == v:  # pas NaN
                total += v
                count += 1
            i += 1

        mean = total / count

        i = 0
        while i < len(values):
            if values[i] != values[i]:  # NaN
                values[i] = mean
            i += 1

        df[col] = values

    return df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_weights(num_features):
    weights = []
    i = 0

    while i < num_features + 1:  # +1 pour le bias
        weights.append(0.0)
        i += 1

    return weights


def somme_pondere(weights, notes):
    bias = weights[0]
    i = 0
    sp = bias
    while i < len(notes):
        sp = sp + (notes[i] * weights[i + 1])
        i = i + 1
    return sp


def update_weights(weights, x, is_good_house, learning_rate):
    sp = somme_pondere(weights, x)
    prediction = sigmoid(sp)

    error = prediction - is_good_house
    weights[0] -= learning_rate * error

    i = 0
    while i < len(x):
        weights[i + 1] = weights[i + 1] - learning_rate * error * x[i]
        i += 1

    return weights


def is_good_house(current, real):
    return 1 if current == real else 0


def save_weights(all_weights, filename):
    data = {}

    for house in all_weights:
        data[house] = {"bias": all_weights[house][0], "weights": all_weights[house][1:]}

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def normalize_dataframe(df, subjects):
    for col in subjects:
        values = df[col].tolist()

        # Calcul de la moyenne
        total = 0
        count = 0
        for v in values:
            total += v
            count += 1
        mean = total / count

        # Calcul de l'écart-type
        var = 0
        for v in values:
            var += (v - mean) ** 2
        std = (var / count) ** 0.5

        # Normalisation
        i = 0
        while i < len(values):
            values[i] = (values[i] - mean) / std
            i += 1

        df[col] = values

    return df


def main():
    if len(sys.argv) < 2:
        filename = os.path.basename(__file__)
        print(f"Usage: python3 {filename} <dataset.csv> [learning_rate] [epochs]")
        sys.exit(1)

    file_path = sys.argv[1]
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else LEARNING_RATE
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS

    try:
        df = pd.read_csv(file_path)
    except:
        print("Error: Cannot read dataset")
        return
    print("Dataset loaded successfully")

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing or extra:
        print("Error: Dataset columns do not match expected format.")
        if missing:
            print("Missing columns:", missing)
        if extra:
            print("Unexpected columns:", extra)
        return
    
    houses = df["Hogwarts House"]
    features_df = df.drop(columns=IGNORED_COLUMNS)
    features_df = replace_nan_by_mean(features_df)
    features_df = normalize_dataframe(features_df, SUBJECTS)

    houses = houses.values.tolist()
    students_scores = features_df.values.tolist()
    final_weights = {}
    for h in HOUSE_TO_LABEL:
        weights = initialize_weights(len(SUBJECTS))
        for _ in range(epochs):
            for stud, house in zip(students_scores, houses):
                igh = is_good_house(h, house)
                weights = update_weights(weights, stud, igh, rate)
        final_weights[h] = weights

    save_weights(final_weights, "save.json")


if __name__ == "__main__":
    main()
