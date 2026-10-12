"""Load the three saved models in a fresh process and score a few rows."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models"
    meta = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(root / "data" / "processed" / "customer_features.csv").head(5)

    extra = [c for c in ["churn_risk_score", "cluster", "cluster_name"] if c in df.columns]
    sample = df.drop(columns=extra)

    churn = joblib.load(models_dir / "churn_model.pkl")
    seg = joblib.load(models_dir / "segmentation_model.pkl")
    clv = joblib.load(models_dir / "clv_model.pkl")

    churn_proba = churn.predict_proba(sample[meta["churn_input_columns"]])[:, 1]
    clusters = seg.predict(sample[meta["cluster_columns"]])
    clv_pred = clv.predict(sample[meta["clv_input_columns"]])

    print("reload ok")
    print("churn_risk_score:", [round(float(x), 3) for x in churn_proba])
    print("cluster:", [int(x) for x in clusters])
    print("clv_pred:", [round(float(x), 2) for x in clv_pred])


if __name__ == "__main__":
    main()
