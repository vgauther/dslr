import sys
import pandas as pd

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
    "Index", "Hogwarts House", "First Name",
    "Last Name", "Birthday", "Best Hand"
]

STAT_NAMES = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

def calcul_percentile(values, p, n):
    sorted_values = sorted(values)
    index = int(p * (n - 1))
    return sorted_values[index]

def calcul_std(values, mean, n):
    var_sum = 0
    for v in values:
        var_sum += (v - mean) ** 2
    std = (var_sum / n) ** 0.5
    return std

def clean_values(values):
    cleaned = []
    for v in values:
        if v == v:   # NaN != NaN
            cleaned.append(v)
    return cleaned

def calcul_min_max(values):
    min_val = values[0]
    max_val = values[0]
    for v in values:
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v
    return min_val, max_val

def calcul_mean(values):
    n = len(values)
    total = 0
    i = 0
    for v in values:
        i = i + 1
        if v != v:  # détecte NaN
            print("Error: NaN value detected in dataset")
            print(v)
            print(i)
            sys.exit(1)
        total += v
    return total / n

def compute_data(values):
    count = len(values)
    mean = calcul_mean(values)
    min, max = calcul_min_max(values)
    std = calcul_std(values, mean, count)
    p25 = calcul_percentile(values, 0.25, count)
    p50 = calcul_percentile(values, 0.50, count)
    p75 = calcul_percentile(values, 0.75, count)
    return count, mean, std, min, p25, p50, p75, max

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset.csv>")
        return

    file_path = sys.argv[1]

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
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

    features = df.drop(columns=IGNORED_COLUMNS)

    results = {}

    for col in features.columns:
        values = features[col].tolist()
        values = clean_values(values)
        results[col] = compute_data(values)

    print("".ljust(30), end="")

    feature_names = list(features.columns)
    for name in feature_names:
        print(name.ljust(30), end="")
    print()

    i = 0
    while i < len(STAT_NAMES):
        print(STAT_NAMES[i].ljust(30), end="")

        j = 0
        while j < len(feature_names):
            col = feature_names[j]
            value = results[col][i]

            if type(value) == float:
                print(("{:.6f}".format(value)).ljust(30), end="")
            else:
                print(str(value).ljust(30), end="")
            j += 1

        print()
        i += 1

if __name__ == "__main__":
    main()
