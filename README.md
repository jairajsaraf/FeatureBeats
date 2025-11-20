# FeatureBeats Hit Prediction Starter

This repository provides a week-by-week, notebook-driven workflow for predicting hit songs using Spotify audio features and Top 100 chart data. The flow emphasizes reproducibility, class-imbalance handling, and interpretability with SHAP.

## Project structure
- `notebooks/00_Setup_and_Installation.ipynb` — create folders and verify dependencies
- `notebooks/01_Week1_Data_Setup_EDA.ipynb` — data loading, fuzzy matching, labeling, and exploratory analysis
- `notebooks/02_Week2_Baseline_Modeling.ipynb` — dummy baseline and class-balanced logistic regression with evaluation plots
- `notebooks/03_Week3_XGBoost_SHAP.ipynb` — imbalance-aware XGBoost, optional tuning, SHAP explanations, and metric comparison
- `data/` — `raw/` for inputs, `processed/` for derived datasets
- `figures/` — exported plots (confusion matrices, ROC/PR curves, SHAP summaries)
- `models/` — saved model artifacts and metric JSON files

## Getting started
1. Create the expected folders and check your environment by running the setup notebook.
2. Download `tracks.csv` and `top100_tracks.csv` into `data/raw/`.
3. Run the Week 1 notebook to produce `data/processed/hits_dataset.csv` plus class balance and EDA visuals.
4. Run the Week 2 notebook to train/save the baseline logistic regression and generate evaluation plots.
5. Run the Week 3 notebook to train/save the XGBoost model, produce SHAP explanations, and compare against the baseline.

## Notes
- Both models address severe class imbalance (`class_weight="balanced"` and `scale_pos_weight`).
- SHAP plots sample up to 1,000 rows for efficiency; adjust `sample_size` if resources allow.
- Figures are saved at 300 DPI for presentation-ready quality.
- If your dataset columns differ, adjust the feature lists in the notebooks before running.
