#!/usr/bin/env python3
"""
Train models on engineered features dataset and compare with original features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*80)
print("TRAINING MODELS ON ENGINEERED FEATURES DATASET")
print("="*80)

# Setup paths
project_root = Path.cwd()
data_dir = project_root / 'data' / 'processed'
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

# Load both datasets
print("\n1. Loading datasets...")
df_original = pd.read_csv(data_dir / 'hits_dataset.csv')
df_engineered = pd.read_csv(data_dir / 'hits_dataset_engineered.csv')

print(f"   Original: {df_original.shape[0]:,} rows × {df_original.shape[1]} columns")
print(f"   Engineered: {df_engineered.shape[0]:,} rows × {df_engineered.shape[1]} columns")
print(f"   New features added: {df_engineered.shape[1] - df_original.shape[1]}")

# Prepare original features
print("\n2. Preparing original features...")
exclude_cols = ['is_hit', 'year', 'track_name', 'artists']
original_features = [col for col in df_original.columns
                     if col not in exclude_cols and df_original[col].dtype in ['float64', 'int64']]
print(f"   Features: {original_features}")

X_orig = df_original[original_features].values
y = df_original['is_hit'].values

# Prepare engineered features
print("\n3. Preparing engineered features...")
engineered_features = [col for col in df_engineered.columns
                       if col not in exclude_cols and df_engineered[col].dtype in ['float64', 'int64']]
print(f"   Features: {engineered_features}")

X_eng = df_engineered[engineered_features].values

# Create same train/test split for fair comparison
print("\n4. Creating train/test splits...")
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

X_orig_train = X_orig[train_idx]
X_orig_test = X_orig[test_idx]
X_eng_train = X_eng[train_idx]
X_eng_test = X_eng[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"   Train: {len(train_idx):,} samples ({y_train.sum()} hits)")
print(f"   Test:  {len(test_idx):,} samples ({y_test.sum()} hits)")

# Scale features
print("\n5. Scaling features...")
scaler_orig = StandardScaler()
X_orig_train_scaled = scaler_orig.fit_transform(X_orig_train)
X_orig_test_scaled = scaler_orig.transform(X_orig_test)

scaler_eng = StandardScaler()
X_eng_train_scaled = scaler_eng.fit_transform(X_eng_train)
X_eng_test_scaled = scaler_eng.transform(X_eng_test)

# Train Logistic Regression models
print("\n6. Training Logistic Regression models...")
print("   Original features...")
lr_orig = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_orig.fit(X_orig_train_scaled, y_train)
y_pred_lr_orig = lr_orig.predict(X_orig_test_scaled)
y_proba_lr_orig = lr_orig.predict_proba(X_orig_test_scaled)[:, 1]

print("   Engineered features...")
lr_eng = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_eng.fit(X_eng_train_scaled, y_train)
y_pred_lr_eng = lr_eng.predict(X_eng_test_scaled)
y_proba_lr_eng = lr_eng.predict_proba(X_eng_test_scaled)[:, 1]

# Train XGBoost models
print("\n7. Training XGBoost models...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("   Original features...")
xgb_orig = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)
xgb_orig.fit(X_orig_train_scaled, y_train, verbose=False)
y_pred_xgb_orig = xgb_orig.predict(X_orig_test_scaled)
y_proba_xgb_orig = xgb_orig.predict_proba(X_orig_test_scaled)[:, 1]

print("   Engineered features...")
xgb_eng = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)
xgb_eng.fit(X_eng_train_scaled, y_train, verbose=False)
y_pred_xgb_eng = xgb_eng.predict(X_eng_test_scaled)
y_proba_xgb_eng = xgb_eng.predict_proba(X_eng_test_scaled)[:, 1]

# Calculate metrics
print("\n8. Calculating metrics...")

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba)
    }

metrics_lr_orig = calculate_metrics(y_test, y_pred_lr_orig, y_proba_lr_orig)
metrics_lr_eng = calculate_metrics(y_test, y_pred_lr_eng, y_proba_lr_eng)
metrics_xgb_orig = calculate_metrics(y_test, y_pred_xgb_orig, y_proba_xgb_orig)
metrics_xgb_eng = calculate_metrics(y_test, y_pred_xgb_eng, y_proba_xgb_eng)

# Save models
print("\n9. Saving models...")
joblib.dump(lr_eng, models_dir / 'logreg_engineered.pkl')
joblib.dump(xgb_eng, models_dir / 'xgboost_engineered.pkl')
joblib.dump(scaler_eng, models_dir / 'scaler_engineered.pkl')
print("   ✅ Saved engineered models")

# Save metrics
print("\n10. Saving metrics...")
comparison_df = pd.DataFrame({
    'Model': ['LogReg-Original', 'LogReg-Engineered', 'XGBoost-Original', 'XGBoost-Engineered'],
    'Features': [len(original_features), len(engineered_features),
                 len(original_features), len(engineered_features)],
    'Accuracy': [metrics_lr_orig['accuracy'], metrics_lr_eng['accuracy'],
                 metrics_xgb_orig['accuracy'], metrics_xgb_eng['accuracy']],
    'Precision': [metrics_lr_orig['precision'], metrics_lr_eng['precision'],
                  metrics_xgb_orig['precision'], metrics_xgb_eng['precision']],
    'Recall': [metrics_lr_orig['recall'], metrics_lr_eng['recall'],
               metrics_xgb_orig['recall'], metrics_xgb_eng['recall']],
    'F1': [metrics_lr_orig['f1_score'], metrics_lr_eng['f1_score'],
           metrics_xgb_orig['f1_score'], metrics_xgb_eng['f1_score']],
    'ROC-AUC': [metrics_lr_orig['roc_auc'], metrics_lr_eng['roc_auc'],
                metrics_xgb_orig['roc_auc'], metrics_xgb_eng['roc_auc']],
    'PR-AUC': [metrics_lr_orig['pr_auc'], metrics_lr_eng['pr_auc'],
               metrics_xgb_orig['pr_auc'], metrics_xgb_eng['pr_auc']]
})

comparison_df.to_csv(models_dir / 'engineered_features_comparison.csv', index=False)
print("   ✅ Saved comparison metrics")

# Print results
print("\n" + "="*80)
print("RESULTS: ORIGINAL vs ENGINEERED FEATURES")
print("="*80)
print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

for model_type in ['LogReg', 'XGBoost']:
    orig_f1 = comparison_df[comparison_df['Model'] == f'{model_type}-Original']['F1'].values[0]
    eng_f1 = comparison_df[comparison_df['Model'] == f'{model_type}-Engineered']['F1'].values[0]
    improvement = ((eng_f1 - orig_f1) / orig_f1 * 100) if orig_f1 > 0 else 0

    print(f"\n{model_type}:")
    print(f"   Original F1:    {orig_f1:.4f}")
    print(f"   Engineered F1:  {eng_f1:.4f}")
    print(f"   Improvement:    {improvement:+.2f}%")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
