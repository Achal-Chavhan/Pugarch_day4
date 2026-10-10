"""Create evaluation plots after training."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "evaluation" / "model_comparison.csv"
OUT = ROOT / "evaluation" / "model_comparison.png"

def main():
    df = pd.read_csv(METRICS)
    ax = df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(
        kind="bar", ylim=(0, 1), figsize=(9, 5), rot=0
    )
    ax.set_title("Hygiene Risk Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Algorithm")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
