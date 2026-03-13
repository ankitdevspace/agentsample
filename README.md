# Customer Acquisition Data Science Project

A minimal Python data science project that demonstrates customer acquisition analysis and modeling using synthetic data.

## 🚀 What’s included

- Synthetic customer acquisition dataset generation (`src/customer_acquisition/data.py`)
- A simple classification model to predict customer acquisition (`src/customer_acquisition/model.py`)
- A runnable pipeline/script (`scripts/run_analysis.py`) that trains and evaluates the model
- A Jupyter notebook for exploratory data analysis (`notebooks/Customer_Acquisition_Analysis.ipynb`)

## 🧰 Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the analysis

Generate data, train, and evaluate the model:

```bash
python scripts/run_analysis.py
```

## 🧪 Running the notebook

Start Jupyter and open the notebook:

```bash
jupyter notebook
```

Then open `notebooks/Customer_Acquisition_Analysis.ipynb`.

---

## 📦 Project structure

- `src/` - core analysis and modeling code
- `scripts/` - runnable scripts (pipeline, data generation)
- `notebooks/` - exploratory analysis
