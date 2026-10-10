"""Basic exploratory data analysis: summary statistics and charts."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "facility_hygiene.csv"
OUT = ROOT / "evaluation"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    print("Shape:", df.shape)
    print("\nSummary statistics:\n", df.describe(include="all"))
    print("\nMissing values:\n", df.isna().sum())
    print("\nRisk label counts:\n", df["hygiene_risk"].value_counts(dropna=False))

    numeric = df.select_dtypes(include="number")
    numeric.hist(figsize=(10, 7), bins=20)
    plt.tight_layout()
    plt.savefig(OUT / "eda_histograms.png", dpi=150)
    print("\nSaved EDA histogram:", OUT / "eda_histograms.png")

if __name__ == "__main__":
    main()
