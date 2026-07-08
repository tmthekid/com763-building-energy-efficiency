# Building Energy Efficiency Predictor

A machine learning system that predicts the heating and cooling load of a
residential building from eight design parameters. Built for COM763 Advanced
Machine Learning.

**Live app:** https://com763-building-energy-efficiency.streamlit.app

## Problem

Early-stage building design decisions (shape, glazing, orientation) have a large
effect on how much energy a building needs for heating and cooling. Estimating
that impact normally requires running building-energy simulation software. This
project trains regression models on simulation data so the two loads can be
predicted instantly from the design parameters, giving a fast decision-support
tool during design.

## Data

The [UCI Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
(768 samples). Each row describes a building via eight features and records its
heating and cooling load. The data is a complete factorial simulation over twelve
base building shapes crossed with glazing area, glazing distribution, and
orientation, so it is clean and has no missing values.

| Feature | Meaning |
|---|---|
| X1 | Relative compactness |
| X2 | Surface area |
| X3 | Wall area |
| X4 | Roof area |
| X5 | Overall height |
| X6 | Orientation |
| X7 | Glazing area |
| X8 | Glazing area distribution |

Targets: heating load and cooling load.

## Model

Two Gradient Boosting regression pipelines, one per target, saved together in
`model.pkl`. The full development process (exploration, model comparison,
evaluation, and selection) is documented in the accompanying notebook.

## Running locally

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Python 3.11–3.13 is recommended, matching the pinned `scikit-learn` version that
the model was trained with. The app opens in your browser, where you select the
design parameters and get the predicted heating and cooling loads.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `model.pkl` | Trained heating and cooling models |
| `requirements.txt` | Python dependencies |
| `building_energy_efficiency.ipynb` | Development, evaluation, and model selection |
