#!/usr/bin/env python3
"""
Test the web app functionality without running Streamlit
"""

import joblib
import numpy as np
from pathlib import Path

print("="*80)
print("TESTING WEB APP FUNCTIONALITY")
print("="*80)

project_root = Path.cwd()
models_dir = project_root / 'models'
figures_dir = project_root / 'figures'

# Test 1: Model Loading
print("\n1. Testing model loading...")
try:
    model = joblib.load(models_dir / 'final_xgboost.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')
    print("   ✅ XGBoost model loaded successfully")
    print(f"   ✅ Scaler loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading XGBoost: {e}")
    try:
        model = joblib.load(models_dir / 'baseline_logreg.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        print("   ✅ Baseline model loaded as fallback")
    except Exception as e2:
        print(f"   ❌ Error loading baseline: {e2}")
        model = None
        scaler = None

if model is None:
    print("\n❌ Cannot proceed - no models available")
    exit(1)

# Test 2: Prediction with Sample Features
print("\n2. Testing prediction with sample song features...")
# Example: A typical hit song (danceable, energetic, positive)
sample_features = np.array([[
    0.75,   # danceability
    0.80,   # energy
    -5.0,   # loudness (dB)
    0.05,   # speechiness
    0.15,   # acousticness
    0.0,    # instrumentalness
    0.12,   # liveness
    0.70,   # valence (happiness)
    125.0   # tempo (BPM)
]])

try:
    features_scaled = scaler.transform(sample_features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    print("   ✅ Prediction successful")
    print(f"   Sample song features: Danceable, Energetic, Happy")
    print(f"   Prediction: {'HIT' if prediction == 1 else 'NON-HIT'}")
    print(f"   Probability: {probability*100:.1f}%")
except Exception as e:
    print(f"   ❌ Error making prediction: {e}")
    exit(1)

# Test 3: Check Required Figures
print("\n3. Checking visualization files...")
required_figures = [
    'model_comparison.png',
    'shap_feature_importance.png',
    'logreg_confusion_matrix.png',
    'xgboost_confusion_matrix.png'
]

for fig in required_figures:
    if (figures_dir / fig).exists():
        size = (figures_dir / fig).stat().st_size / 1024
        print(f"   ✅ {fig:35s} ({size:>6.1f} KB)")
    else:
        print(f"   ⚠️  {fig:35s} (missing)")

# Test 4: Test edge cases
print("\n4. Testing edge cases...")

# Test case 1: All minimum values
min_features = np.array([[0.0, 0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
min_scaled = scaler.transform(min_features)
min_pred = model.predict_proba(min_scaled)[0][1]
print(f"   Min values prediction: {min_pred*100:.1f}%")

# Test case 2: All maximum values
max_features = np.array([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 200.0]])
max_scaled = scaler.transform(max_features)
max_pred = model.predict_proba(max_scaled)[0][1]
print(f"   Max values prediction: {max_pred*100:.1f}%")

# Test case 3: Typical non-hit (low energy, acoustic, instrumental)
nonhit_features = np.array([[0.3, 0.2, -20.0, 0.02, 0.9, 0.8, 0.1, 0.3, 90.0]])
nonhit_scaled = scaler.transform(nonhit_features)
nonhit_pred = model.predict_proba(nonhit_scaled)[0][1]
print(f"   Typical non-hit prediction: {nonhit_pred*100:.1f}%")

# Test 5: Check model type
print("\n5. Model information...")
print(f"   Model type: {type(model).__name__}")
if hasattr(model, 'n_estimators'):
    print(f"   Number of estimators: {model.n_estimators}")
if hasattr(model, 'max_depth'):
    print(f"   Max depth: {model.max_depth}")

print("\n" + "="*80)
print("✅ WEB APP FUNCTIONALITY TEST COMPLETE")
print("="*80)
print("\nTo run the web app:")
print("   streamlit run app.py")
print("\nThen open: http://localhost:8501")
print("="*80)
