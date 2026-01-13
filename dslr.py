from pathlib import Path

EPOCHS = 50
LEARNING_RATE = float(0.00001)

STAT_NAMES = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
HOUSE_TO_LABEL = {"Gryffindor": 0, "Slytherin": 1, "Ravenclaw": 2, "Hufflepuff": 3}
PATH_WEIGHT = Path("save.json")
HOUSE_COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}
EXPECTED_COLUMNS = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]
IGNORED_COLUMNS = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
]
SUBJECTS = [
    c for c in EXPECTED_COLUMNS if c not in IGNORED_COLUMNS and c != "Hogwarts House"
]
