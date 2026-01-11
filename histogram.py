import sys
import pandas as pd
import matplotlib.pyplot as plt

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
    "Index", "First Name", "Last Name", "Birthday", "Best Hand"
]

HOUSE_COLORS = {
    "Ravenclaw": "blue",
    "Slytherin": "green",
    "Gryffindor": "red",
    "Hufflepuff": "yellow"
}

def usage():
    print("Usage: python3 histogram.py <dataset.csv> [Subject]")
    sys.exit(1)


def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        usage()

    file_path = sys.argv[1]
    subject_filter = None

    if len(sys.argv) == 3:
        subject_filter = sys.argv[2]

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
    
    if subject_filter:
        if subject_filter not in EXPECTED_COLUMNS:
            print("Error: Unknown subject")
            return
        
    subjects = [c for c in EXPECTED_COLUMNS if c not in IGNORED_COLUMNS and c != "Hogwarts House"]

    if subject_filter:
        subjects = [subject_filter]
    
    for subject in subjects:
            plt.figure()

            for house in HOUSE_COLORS:
                values = []

                for i in range(len(df)):
                    if df["Hogwarts House"][i] == house:
                        v = df[subject][i]
                        if v == v:  # pas NaN
                            values.append(v)

                plt.hist(
                    values,
                    bins=20,
                    alpha=0.5,
                    color=HOUSE_COLORS[house],
                    label=house
                )

            plt.title(subject)
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    main()
