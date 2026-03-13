"""Run end-to-end customer acquisition modeling pipeline."""

from __future__ import annotations

import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.customer_acquisition.data import generate_synthetic_data, save_data
from src.customer_acquisition.model import evaluate_model, save_model, train_model


def main() -> None:
    workspace = pathlib.Path(__file__).resolve().parents[1]
    data_path = workspace / "data" / "customer_acquisition.csv"
    model_path = workspace / "models" / "customer_acquisition_model.joblib"
    report_path = workspace / "models" / "evaluation_report.csv"

    print("Generating synthetic dataset...")
    df = generate_synthetic_data(n_samples=2000)
    save_data(df, data_path)

    print("Splitting data into train/test...")
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["acquired"])

    print("Training model...")
    model = train_model(train)

    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test)
    report_df = pd.DataFrame([metrics])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    print("Saving model...")
    save_model(model, model_path)

    print("✅ Pipeline complete.")
    print(f"- Dataset: {data_path}")
    print(f"- Model: {model_path}")
    print(f"- Metrics: {report_path}")
    print(f"- Test metrics: {metrics}")


if __name__ == "__main__":
    main()
