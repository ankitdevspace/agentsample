"""Customer acquisition modeling package."""

from .data import generate_synthetic_data, load_data, save_data
from .model import train_model, evaluate_model, save_model

__all__ = [
    "generate_synthetic_data",
    "load_data",
    "save_data",
    "train_model",
    "evaluate_model",
    "save_model",
]
