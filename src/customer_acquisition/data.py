"""Data generation and loading utilities for customer acquisition."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic customer acquisition dataset.

    The dataset is intentionally simple for demonstration purposes.

    Returns
    -------
    pd.DataFrame
        Columns:
        - age
        - annual_income
        - credit_score
        - marketing_channel
        - visited_website
        - past_purchases
        - acquired (target)
    """

    rng = np.random.default_rng(random_state)
    ages = rng.integers(18, 75, size=n_samples)
    annual_income = np.round(rng.normal(65_000, 25_000, size=n_samples)).astype(int)
    credit_score = rng.integers(300, 850, size=n_samples)

    marketing_channel = rng.choice(
        ["email", "social", "search", "referral"], size=n_samples, p=[0.35, 0.25, 0.25, 0.15]
    )

    visited_website = rng.choice([0, 1], size=n_samples, p=[0.3, 0.7])

    past_purchases = rng.poisson(lam=1.2, size=n_samples)

    # Simulated acquisition probability
    score = (
        (annual_income - 40_000) / 40_000
        + (credit_score - 600) / 250
        + (visited_website * 0.8)
        + (past_purchases * 0.25)
        + np.where(marketing_channel == "referral", 0.6, 0.0)
    )
    probs = 1 / (1 + np.exp(-score))

    acquired = rng.binomial(1, probs)

    df = pd.DataFrame(
        {
            "age": ages,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "marketing_channel": marketing_channel,
            "visited_website": visited_website,
            "past_purchases": past_purchases,
            "acquired": acquired,
        }
    )

    return df


def save_data(df: pd.DataFrame, dest: str | pathlib.Path) -> None:
    """Save dataset to CSV."""

    dest_path = pathlib.Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)


def load_data(source: str | pathlib.Path) -> pd.DataFrame:
    """Load dataset from CSV."""

    return pd.read_csv(source)
