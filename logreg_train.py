import sys
import pandas as pd
import numpy as np
import json

HOUSE_TO_LABEL = {
    "Gryffindor": 0,
    "Slytherin": 1,
    "Ravenclaw": 2,
    "Hufflepuff": 3
}

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
    "Index", "First Name",
    "Last Name", "Birthday", "Best Hand"
]

SUBJECTS = [c for c in EXPECTED_COLUMNS 
            if c not in IGNORED_COLUMNS and c != "Hogwarts House"]

EPOCHS = 100

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
    i = 1
    sp = bias
    while i < len(notes):
        sp = sp + (notes[i] * weights[i])
        i = i + 1
    return sp

def update_weights(weights, x, is_good_house):
    sp = somme_pondere(weights, x)
    prediction = sigmoid(sp)

    error = prediction - is_good_house
    weights[0] += error



    return weights

def is_good_house(current, real):
    return current == real

def save_weights(all_weights, filename):
    data = {}

    for house in all_weights:
        data[house] = {
            "bias": all_weights[house][0],
            "weights": all_weights[house][1:]
        }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def usage():
    print("Usage: python3 logreg_train.py <dataset.csv>")
    sys.exit(1)

def main():

    if len(sys.argv) != 2:
        usage()

    file_path = sys.argv[1]

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
    
    features_df = df.drop(columns=IGNORED_COLUMNS)
    features_df = features_df.dropna()
    students_scores = features_df.values.tolist()
    final_weights = {}
    for h in HOUSE_TO_LABEL:
        weights = initialize_weights(len(SUBJECTS))
        for i in range(EPOCHS):
            for stud in students_scores:
                weights = update_weights(weights, stud, is_good_house(h, stud[0]))
        final_weights[h] = weights
    
    print(final_weights)
    save_weights(final_weights, "save.json")



if __name__ == "__main__":
    main()