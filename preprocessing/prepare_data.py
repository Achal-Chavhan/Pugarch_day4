"""Clean the raw facility hygiene dataset and save a model-ready CSV."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "facility_hygiene.csv"
OUTPUT = ROOT / "preprocessing" / "cleaned_facility_hygiene.csv"

FEATURES = [
    "cleanliness_score", "odor_score", "waste_level",
    "complaints", "footfall", "hours_since_cleaning"
]
TARGET = "hygiene_risk"

def main():
    df = pd.read_csv(SOURCE)
    print("Raw rows:", len(df))
    print("Missing values before cleaning:\n", df.isna().sum())
    print("Duplicate rows before cleaning:", df.duplicated().sum())

    df = df.drop_duplicates().copy()
    # Scores are expected to be between 0 and 10.
    for col in ["cleanliness_score", "odor_score", "waste_level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~df[col].between(0, 10), col] = float("nan")
        df[col] = df[col].fillna(df[col].median()).round(1)

    # Counts cannot be negative; footfall and hours are constrained to valid ranges.
    for col in ["complaints", "footfall", "hours_since_cleaning"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["complaints"] < 0, "complaints"] = float("nan")
    df.loc[df["footfall"] < 0, "footfall"] = float("nan")
    df.loc[~df["hours_since_cleaning"].between(0, 24), "hours_since_cleaning"] = float("nan")

    for col in ["complaints", "footfall", "hours_since_cleaning"]:
        df[col] = df[col].fillna(df[col].median())
    df["complaints"] = df["complaints"].round().astype(int)
    df["footfall"] = df["footfall"].round().astype(int)
    df["hours_since_cleaning"] = df["hours_since_cleaning"].round(1)

    df = df.dropna(subset=[TARGET])
    df.to_csv(OUTPUT, index=False)
    print("\nCleaned rows:", len(df))
    print("Missing values after cleaning:\n", df.isna().sum())
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
