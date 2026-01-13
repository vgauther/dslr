import sys
import pandas as pd

def usage():
    print("Usage: python3 evaluate.py <dataset.csv> <houses.csv>")
    sys.exit(1)

def main():
    if len(sys.argv) != 3:
        usage()

    dataset_path = sys.argv[1]
    houses_path = sys.argv[2]

    try:
        df_real = pd.read_csv(dataset_path)
        df_pred = pd.read_csv(houses_path)
    except:
        print("Error: Cannot read files")
        return

    # Vérification des colonnes
    if "Hogwarts House" not in df_real.columns:
        print("Error: Dataset must contain 'Hogwarts House'")
        return

    if "Index" not in df_pred.columns or "Hogwarts House" not in df_pred.columns:
        print("Error: houses.csv must contain 'Index' and 'Hogwarts House'")
        return

    total = 0
    correct = 0

    i = 0
    while i < len(df_pred):
        index = df_pred["Index"][i]
        predicted = df_pred["Hogwarts House"][i]
        real = df_real["Hogwarts House"][index]

        if predicted == real:
            correct += 1

        total += 1
        i += 1

    accuracy = (correct / total) * 100

    print("Correct predictions:", correct)
    print("Total:", total)
    print("Accuracy: {:.2f}%".format(accuracy))

if __name__ == "__main__":
    main()
