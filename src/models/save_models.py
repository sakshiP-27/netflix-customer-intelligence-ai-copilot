from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.config import (
    CLUSTER_COLS,
    CLUSTER_N,
    CLUSTER_NAMES,
    CLV_DROP_COLS,
    CHURN_DROP_COLS,
    CHURN_THRESHOLD,
    RANDOM_STATE,
    project_root,
)


def _existing_drop(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols


def _preprocess(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def fit_churn(df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    y = df["churned"].astype(int)
    X = df.drop(columns=_existing_drop(df, CHURN_DROP_COLS))
    cat_cols, num_cols = _split_cols(X)
    pipe = Pipeline(
        [
            ("prep", _preprocess(cat_cols, num_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe, list(X.columns)


def fit_segmentation(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=CLUSTER_N, n_init=10, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(df[CLUSTER_COLS])
    return pipe


def fit_clv(df: pd.DataFrame) -> tuple[TransformedTargetRegressor, list[str]]:
    y = df["customer_lifetime_value"]
    X = df.drop(columns=_existing_drop(df, CLV_DROP_COLS))
    cat_cols, num_cols = _split_cols(X)
    inner = Pipeline(
        [
            ("prep", _preprocess(cat_cols, num_cols)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=inner, func=np.log1p, inverse_func=np.expm1
    )
    model.fit(X, y)
    return model, list(X.columns)


def save_all(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else project_root()
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(root / "data" / "processed" / "customer_features.csv")
    churn_pipe, churn_cols = fit_churn(df)
    seg_pipe = fit_segmentation(df)
    clv_pipe, clv_cols = fit_clv(df)

    joblib.dump(churn_pipe, models_dir / "churn_model.pkl")
    joblib.dump(seg_pipe, models_dir / "segmentation_model.pkl")
    joblib.dump(clv_pipe, models_dir / "clv_model.pkl")

    X_churn = df.drop(columns=_existing_drop(df, CHURN_DROP_COLS))
    df["churn_risk_score"] = churn_pipe.predict_proba(X_churn)[:, 1]
    df["cluster"] = seg_pipe.predict(df[CLUSTER_COLS])
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

    features_path = root / "data" / "processed" / "customer_features.csv"
    df.to_csv(features_path, index=False)

    metadata = {
        "churn_model": "logreg_balanced",
        "churn_threshold": CHURN_THRESHOLD,
        "churn_input_columns": churn_cols,
        "segmentation_k": CLUSTER_N,
        "cluster_columns": CLUSTER_COLS,
        "cluster_names": {str(k): v for k, v in CLUSTER_NAMES.items()},
        "clv_model": "random_forest_log1p",
        "clv_input_columns": clv_cols,
    }
    meta_path = models_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "n_customers": int(len(df)),
        "models_dir": str(models_dir),
        "features_path": str(features_path),
        "metadata_path": str(meta_path),
        "churn_risk_mean": float(df["churn_risk_score"].mean()),
    }
