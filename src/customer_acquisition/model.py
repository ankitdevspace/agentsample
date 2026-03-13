"""Model training and evaluation utilities."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( 
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_pipeline() -> Pipeline:
    numeric_features = ["age", "annual_income", "credit_score", "visited_website", "past_purchases"]
    categorical_features = ["marketing_channel"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
        ]
    )
    return pipeline


def train_model(df: pd.DataFrame, target: str = "acquired") -> Pipeline:
    """Train a classifier to predict customer acquisition."""

    X = df.drop(columns=[target])
    y = df[target]

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    return pipeline


def evaluate_model(model: Pipeline, df: pd.DataFrame, target: str = "acquired") -> dict:
    """Evaluate model performance and return key metrics."""

    X = df.drop(columns=[target])
    y_true = df[target]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_model(model: Pipeline, dest: str | pathlib.Path) -> None:
    """Persist trained model to disk."""

    import joblib

    dest_path = pathlib.Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest_path)
