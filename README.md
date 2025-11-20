# FeatureBeats: Hit Song Prediction

This repository provides a week-by-week notebook workflow for predicting hit songs using Spotify audio features. The notebooks prioritize class imbalance handling, reproducibility, and interpretability via SHAP.

## Contents
- `00_Setup_and_Installation.ipynb` — bootstrap directories, verify environment, and smoke-test dependencies.
- `01_Week1_Data_Setup_EDA.ipynb` — load raw datasets, label hits (ID → exact match → fuzzy), explore imbalance, and export processed data.
- `02_Week2_Baseline_Modeling.ipynb` — dummy baseline and class-balanced logistic regression with comprehensive metrics and coefficient insights.
- `03_Week3_XGBoost_SHAP.ipynb` — XGBoost with imbalance-aware training, optional tuning, evaluation, and SHAP interpretability.

## Quickstart
1. **Create directories and install deps**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Place data**
   - `data/raw/tracks.csv` — full Spotify tracks dataset.
   - `data/raw/top100_tracks.csv` — Top 100 (or chart) reference list.
3. **Run notebooks in order**
   ```bash
   jupyter notebook
   ```
   Execute notebooks `00` → `03`. Outputs (processed CSVs, figures, models) are saved under `data/processed`, `figures`, and `models`.

## Notes
- Class imbalance is addressed with `class_weight='balanced'` (logreg) and `scale_pos_weight` (XGBoost).
- Visualizations are saved at 300 DPI for presentation use.
- Update `AUDIO_FEATURES` and column names in the Week 1 notebook if your dataset schema differs.
