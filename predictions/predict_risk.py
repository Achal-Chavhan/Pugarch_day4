"""Predict hygiene risk for new facility records using the selected model."""
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
INPUT = ROOT / "predictions" / "new_facilities.csv"
OUTPUT = ROOT / "predictions" / "hygiene_risk_predictions.csv"
FEATURES = ["cleanliness_score", "odor_score", "waste_level", "complaints", "footfall", "hours_since_cleaning"]

def main():
    selected = (MODEL_DIR / "selected_model.txt").read_text(encoding="utf-8")
    model_name = selected.split(":", 1)[1].strip().splitlines()[0]
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    data = pd.read_csv(INPUT)
    data["predicted_hygiene_risk"] = model.predict(data[FEATURES])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data[FEATURES])
        data["prediction_confidence"] = probabilities.max(axis=1).round(3)
    data.to_csv(OUTPUT, index=False)
    print(data.to_string(index=False))
    print("\nSaved predictions:", OUTPUT)

if __name__ == "__main__":
    main()
