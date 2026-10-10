"""Train and compare Logistic Regression and Random Forest classifiers."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "preprocessing" / "cleaned_facility_hygiene.csv"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
FEATURES = ["cleanliness_score", "odor_score", "waste_level", "complaints", "footfall", "hours_since_cleaning"]
TARGET = "hygiene_risk"

def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
            min_samples_leaf=2
        )
    }

    results, reports = [], {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0)
        })
        reports[name] = {
            "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True),
            "confusion_matrix_labels": sorted(y.unique().tolist()),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique().tolist())).tolist()
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    metrics = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    metrics.to_csv(EVAL_DIR / "model_comparison.csv", index=False)
    with open(EVAL_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    best_name = metrics.iloc[0]["model"]
    (MODEL_DIR / "selected_model.txt").write_text(
        f"Selected model: {best_name}\nSelection rule: highest macro F1-score on the held-out test set.\n",
        encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("Saved model files in:", MODEL_DIR)
    print("Saved evaluation files in:", EVAL_DIR)

if __name__ == "__main__":
    main()
