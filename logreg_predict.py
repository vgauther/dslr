#!/usr/bin/env python

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOUSE_TO_LABEL = {
    "Gryffindor": 0,
    "Slytherin": 1,
    "Ravenclaw": 2,
    "Hufflepuff": 3
}

PATH_WEIGHT = Path("save.json")

EXPECTED_COLUMNS = [
    "Index", "Hogwarts House", "First Name", "Last Name",
    "Birthday", "Best Hand",
    "Arithmancy", "Astronomy", "Herbology",
    "Defense Against the Dark Arts", "Divination",
    "Muggle Studies", "Ancient Runes", "History of Magic",
    "Transfiguration", "Potions",
    "Care of Magical Creatures", "Charms", "Flying"
]

IGNORED_COLUMNS = [
    "Index", "Hogwarts House","First Name",
    "Last Name", "Birthday", "Best Hand"
]

SUBJECTS = [c for c in EXPECTED_COLUMNS 
            if c not in IGNORED_COLUMNS and c != "Hogwarts House"]

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

def load_weights_as_arrays(data):

    all_weights = {}

    for house in data:
        bias = data[house]["bias"]
        weights = data[house]["weights"]

        # Tableau final : [bias, w1, w2, ...]
        all_weights[house] = [bias] + weights

    return all_weights

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def somme_pondere(weights, notes):
    bias = weights[0]
    i = 0
    sp = bias
    while i < len(notes):
        sp = sp + (notes[i] * weights[i+1])
        i = i + 1
    return sp

def predict(weights, notes):
    sp = somme_pondere(weights, notes)
    ret = sigmoid(sp)
    print(ret)
    return ret

def create_dict_array(size):
    arr = []
    i = 0

    while i < size:
        arr.append({"Gryffindor": 0, "Slytherin": 0, "Ravenclaw": 0, "Hufflepuff": 0})
        i += 1

    return arr

def get_best_house(d):
    best_house = None
    best_value = -1

    for house in d:
        if d[house] > best_value:
            best_value = d[house]
            best_house = house

    return best_house

def write_houses_csv(results, filename):
    f = open(filename, "w")

    f.write("Index,Hogwarts House\n")

    i = 0
    while i < len(results):
        best_house = get_best_house(results[i])
        f.write(str(i) + "," + best_house + "\n")
        i += 1

    f.close()

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

    # df = df.select_dtypes(include="number").drop(columns="Index")
    features_df = df.drop(columns=IGNORED_COLUMNS)
    features_df = replace_nan_by_mean(features_df)
    features_df = normalize_dataframe(features_df, SUBJECTS)
    weights = load_weights_as_arrays(weights)
    students_scores = features_df.values.tolist()
    results = create_dict_array(len(students_scores))

    for h in HOUSE_TO_LABEL:
        i = 0
        for stud in students_scores:
             results[i][h] = predict(weights[h], stud)
             i = i + 1 

    print(results)
    write_houses_csv(results, "houses.csv")


if __name__ == "__main__":
    main()
