# Hit Song Prediction Using Spotify Audio Features

A machine learning project to predict whether a song will become a hit based on its audio features, using Logistic Regression and XGBoost with SHAP interpretability.

## Project Overview

This project analyzes Spotify audio features to predict hit songs. We compare interpretable (Logistic Regression) and high-performance (XGBoost) models, and use SHAP (SHapley Additive exPlanations) to understand which musical characteristics contribute to a song's success.

### Key Features

- **Comprehensive Data Pipeline**: Automated data loading, cleaning, and labeling
- **Dual Modeling Approach**: Both interpretable and high-performance models
- **Advanced Interpretation**: SHAP analysis for feature importance and effects
- **Production-Quality Code**: Well-documented Jupyter notebooks with visualizations
- **Handles Class Imbalance**: Specialized techniques for imbalanced datasets

## Dataset

This project uses two Kaggle datasets:

1. **Spotify Tracks Dataset** - Large collection of songs with audio features
2. **Billboard/Spotify Top 100** - Chart-topping hits for labeling

### Audio Features Analyzed

- **Danceability**: How suitable for dancing (0-1)
- **Energy**: Intensity and activity measure (0-1)
- **Loudness**: Overall loudness in dB
- **Speechiness**: Presence of spoken words (0-1)
- **Acousticness**: Confidence measure of acoustic sound (0-1)
- **Instrumentalness**: Predicts whether track has vocals (0-1)
- **Liveness**: Detects presence of audience (0-1)
- **Valence**: Musical positivity/happiness (0-1)
- **Tempo**: Overall tempo in BPM

## Project Structure

```
hit-song-prediction/
│
├── notebooks/               # Jupyter notebooks (main workflow)
│   ├── 00_Setup_and_Installation.ipynb
│   ├── 01_Week1_Data_Setup_EDA.ipynb
│   ├── 02_Week2_Baseline_Modeling.ipynb
│   └── 03_Week3_XGBoost_SHAP.ipynb
│
├── data/
│   ├── raw/                # Original datasets (gitignored)
│   │   ├── tracks.csv
│   │   └── top100_tracks.csv
│   └── processed/          # Cleaned datasets
│       └── hits_dataset.csv
│
├── models/                 # Trained models
│   ├── baseline_logreg.pkl
│   ├── final_xgboost.pkl
│   ├── scaler.pkl
│   └── shap_values.npy
│
├── figures/               # Generated visualizations
│   ├── tracks_by_year.png
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── logreg_confusion_matrix.png
│   ├── logreg_coefficients.png
│   ├── xgboost_confusion_matrix.png
│   ├── model_comparison.png
│   ├── shap_feature_importance.png
│   └── shap_summary_detailed.png
│
├── src/                   # Python scripts (optional)
├── reports/               # Final reports and presentations
│
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Conda for environment management

### Setup Instructions

#### Option 1: Using pip (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd hit-song-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

#### Option 2: Using Conda

```bash
# 1. Clone and navigate
git clone <repository-url>
cd hit-song-prediction

# 2. Create conda environment
conda create -n hits python=3.10
conda activate hits

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Download Datasets

1. Visit Kaggle and download:
   - "Spotify Tracks Dataset" → Save as `data/raw/tracks.csv`
   - "Spotify Top 100" or "Billboard Hot 100" → Save as `data/raw/top100_tracks.csv`

2. Verify files are in correct location:
   ```bash
   ls data/raw/
   # Should show: tracks.csv  top100_tracks.csv
   ```

## Usage

### Quick Start (5 minutes)

1. **Setup Environment**
   ```bash
   jupyter notebook
   ```

2. **Run Setup Notebook**
   - Open `notebooks/00_Setup_and_Installation.ipynb`
   - Run all cells to verify installation

3. **Verify Data**
   - Ensure datasets are in `data/raw/`
   - Run verification cells in setup notebook

### Full Workflow (60-90 minutes)

Execute notebooks in order:

#### Week 1: Data Preparation & EDA (20 minutes)
**Notebook**: `01_Week1_Data_Setup_EDA.ipynb`

- Load Spotify tracks and Top 100 datasets
- Create HIT/NON-HIT labels through dataset matching
- Handle missing values and outliers
- Explore feature distributions
- Generate correlation analysis
- Save processed dataset

**Outputs:**
- `data/processed/hits_dataset.csv`
- `figures/tracks_by_year.png`
- `figures/feature_distributions.png`
- `figures/correlation_matrix.png`

#### Week 2: Baseline Modeling (15 minutes)
**Notebook**: `02_Week2_Baseline_Modeling.ipynb`

- Train/test split with stratification
- Feature scaling (StandardScaler)
- Dummy baseline for comparison
- Logistic Regression with class balancing
- Comprehensive evaluation metrics
- Coefficient interpretation

**Outputs:**
- `models/baseline_logreg.pkl`
- `models/scaler.pkl`
- `figures/logreg_confusion_matrix.png`
- `figures/logreg_roc_curve.png`
- `figures/logreg_pr_curve.png`
- `figures/logreg_coefficients.png`

#### Week 3: Advanced Modeling & Interpretation (30-45 minutes)
**Notebook**: `03_Week3_XGBoost_SHAP.ipynb`

- XGBoost with class imbalance handling
- Hyperparameter tuning (optional: set `SKIP_TUNING=True` for faster run)
- Model evaluation and comparison
- SHAP analysis for interpretability
- Feature importance visualization
- Generate presentation materials

**Outputs:**
- `models/final_xgboost.pkl`
- `models/shap_values.npy`
- `figures/xgboost_confusion_matrix.png`
- `figures/model_comparison.png`
- `figures/shap_feature_importance.png`
- `figures/shap_summary_detailed.png`
- `figures/shap_dependence_plots.png`

## Key Results

### Expected Performance

| Metric | Logistic Regression | XGBoost | Improvement |
|--------|-------------------|---------|-------------|
| **Accuracy** | 75-85% | 80-90% | +5-10% |
| **Precision** | 50-70% | 60-80% | +10-15% |
| **Recall** | 60-75% | 70-85% | +10-15% |
| **F1 Score** | 55-72% | 65-82% | +10-15% |
| **ROC-AUC** | 0.75-0.85 | 0.80-0.92 | +5-10% |

*Note: Actual results depend on dataset quality and class imbalance ratio*

### Key Insights (Typical Findings)

Based on similar Spotify audio feature research:

1. **Important Features for Hits:**
   - ✅ **Danceability** (+): More danceable songs tend to chart
   - ✅ **Energy** (+): Energetic songs perform better
   - ✅ **Valence** (+): Positive, happy songs are favored
   - ❌ **Acousticness** (-): Less acoustic = more likely to chart
   - ❌ **Instrumentalness** (-): Vocal tracks outperform instrumental

2. **Model Comparison:**
   - Logistic Regression: Simple, interpretable, linear assumptions
   - XGBoost: Captures non-linear patterns, higher performance
   - SHAP bridges the gap: Makes XGBoost interpretable

3. **Class Imbalance Challenge:**
   - Severe imbalance (~10-50:1 non-hits to hits)
   - Addressed through `class_weight` and `scale_pos_weight`
   - Focus on F1, Recall, and PR-AUC over accuracy

## Methodology

### 1. Data Preprocessing

- **Fuzzy Matching**: Match hits across datasets using track name + artist
- **Labeling Strategy**:
  - HIT (1): Appears in Top 100 dataset
  - NON-HIT (0): Does not appear in Top 100
- **Temporal Filtering**: Focus on 2010-2020 (configurable)
- **Missing Values**: Drop rows with missing audio features

### 2. Handling Class Imbalance

#### Logistic Regression
```python
LogisticRegression(class_weight='balanced')
```
Automatically adjusts weights inversely proportional to class frequencies.

#### XGBoost
```python
scale_pos_weight = (# negative samples) / (# positive samples)
XGBClassifier(scale_pos_weight=scale_pos_weight)
```
Gives more weight to minority class (hits) during training.

### 3. Evaluation Strategy

**Primary Metrics** (for imbalanced data):
- **F1 Score**: Harmonic mean of precision and recall
- **Recall**: Percentage of actual hits correctly identified
- **PR-AUC**: Precision-Recall Area Under Curve

**Secondary Metrics**:
- Accuracy: Overall correctness
- ROC-AUC: Discrimination ability

**Why not just accuracy?**
With 95% non-hits, a model predicting "all non-hits" gets 95% accuracy but 0% recall!

### 4. SHAP Interpretation

SHAP (SHapley Additive exPlanations) explains predictions:

- **Global Importance**: Which features matter most overall
- **Local Importance**: Why a specific song was predicted as a hit
- **Directional Effects**: How feature values affect predictions
- **Interactions**: Complex relationships between features

## Customization

### Adjust Year Range

In `01_Week1_Data_Setup_EDA.ipynb`:
```python
YEAR_START = 2015  # Change from default 2010
YEAR_END = 2023    # Change from default 2020
```

### Select Different Features

In `01_Week1_Data_Setup_EDA.ipynb`:
```python
audio_features = [
    'danceability', 'energy', 'valence',  # Keep these
    # Add or remove features as needed
]
```

### Speed Up Hyperparameter Tuning

In `03_Week3_XGBoost_SHAP.ipynb`:
```python
SKIP_TUNING = True  # Use default parameters (faster)
# Or reduce iterations:
n_iter = 10  # Default is 20
```

### Adjust Train/Test Split

In `02_Week2_Baseline_Modeling.ipynb`:
```python
TEST_SIZE = 0.3  # Default is 0.2 (20%)
```

## Troubleshooting

### Installation Issues

**Problem**: `pip install` fails for certain packages

**Solution**:
```bash
# Update pip first
pip install --upgrade pip

# Install problematic packages separately
pip install xgboost==2.0.0
pip install shap==0.42.0
```

### Dataset Issues

**Problem**: Column names don't match

**Solution**: Manually set column names in `01_Week1_Data_Setup_EDA.ipynb`:
```python
# After loading datasets, set these manually:
track_name_col = 'your_column_name'
artist_col = 'your_artist_column_name'
```

**Problem**: Very few hits matched (< 50)

**Solutions**:
1. Use fuzzy matching (uncomment fuzzy matching cell in Week 1)
2. Try a different Top 100 dataset
3. Lower matching threshold

### Memory Issues

**Problem**: Out of memory error during SHAP analysis

**Solution**: Reduce sample size in `03_Week3_XGBoost_SHAP.ipynb`:
```python
sample_size = min(500, len(X_test))  # Default is 1000
```

### Slow Performance

**Problem**: Hyperparameter tuning takes too long

**Solutions**:
1. Set `SKIP_TUNING = True`
2. Reduce `n_iter` from 20 to 10
3. Use fewer CPU cores: `n_jobs=2` instead of `n_jobs=-1`

## Limitations

### Data Limitations

- **Audio features only**: Doesn't account for marketing, artist fame, timing, luck
- **Temporal bias**: Music trends change over time
- **Limited scope**: Only Spotify features, may miss cultural context
- **Survivorship bias**: Only includes songs released on Spotify

### Model Limitations

- **Correlation ≠ Causation**: Features are associated with hits, not causing them
- **Class imbalance**: Severe imbalance makes prediction challenging
- **Generalization**: Model trained on past hits may not predict future trends
- **Feature engineering**: Simple features, no interaction terms or temporal features

## Future Improvements

### Short-term (Quick Wins)

1. **Feature Engineering**
   - Interaction terms (e.g., energy × danceability)
   - Temporal features (month of release, day of week)
   - Normalized features (percentiles within year)

2. **Advanced Sampling**
   - SMOTE (Synthetic Minority Oversampling)
   - ADASYN (Adaptive Synthetic Sampling)
   - Ensemble with different sampling strategies

3. **Additional Models**
   - Random Forest
   - LightGBM
   - Neural Networks

### Long-term (Research Projects)

1. **External Data**
   - Artist popularity metrics
   - Lyrics sentiment analysis
   - Social media trends
   - Music video views

2. **Temporal Modeling**
   - Time series analysis of music trends
   - Seasonal effects
   - Genre evolution over time

3. **Deep Learning**
   - Audio spectrograms with CNNs
   - Transformer models for music
   - Multi-modal learning (audio + lyrics + metadata)

4. **Deployment**
   - Web application for predictions
   - API for real-time inference
   - Dashboard for music industry professionals

## Citation

If you use this project in your research or presentation, please cite:

```
Hit Song Prediction Using Spotify Audio Features
Machine Learning Course Project
[Your Name/Team], [University/Institution], [Year]
```

## License

This project is for educational purposes. Dataset licenses:
- Spotify data: Check Kaggle dataset licenses
- Code: MIT License (if applicable)

## Acknowledgments

- **Datasets**: Kaggle contributors for Spotify and Billboard datasets
- **Libraries**: scikit-learn, XGBoost, SHAP, pandas, seaborn
- **Inspiration**: Music information retrieval research community

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email]
- Course: [Course Name and Number]

---

## Advanced Features (Bonus)

### 🚀 Additional Notebooks

Beyond the core 3-week curriculum, we've included advanced techniques:

#### 04 - SMOTE & Sampling Techniques
**Notebook**: `04_Advanced_SMOTE_Sampling.ipynb`

Explores advanced sampling methods for handling class imbalance:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **Borderline-SMOTE**
- **SMOTE + Tomek Links** (combined over/under-sampling)

**Runtime**: ~15 minutes

**Key Insights**:
- When SMOTE helps vs when class weighting is sufficient
- Trade-offs between precision and recall
- Comparison of all sampling strategies

#### 05 - Feature Engineering
**Notebook**: `05_Feature_Engineering.ipynb`

Creates advanced features to boost model performance:
- **Interaction Terms**: energy×danceability, valence×energy, etc.
- **Polynomial Features**: Squared terms for non-linear relationships
- **Domain-Specific Features**: party_factor, acoustic_contrast
- **Temporal Features**: year_normalized, year_period

**Runtime**: ~10 minutes

**Impact**: Typically 2-5% improvement in F1 score

### 🌐 Web Application

#### Interactive Streamlit App
**File**: `app.py`

A beautiful web interface for making predictions!

**Features**:
- 🎯 Real-time hit prediction
- 📊 Interactive sliders for all audio features
- 📈 Probability visualization
- 🔍 Feature importance insights
- 💡 Model explanations

**Launch the app**:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Screenshot Features**:
- Adjust 9 audio feature sliders
- Get instant hit probability prediction
- View feature contributions
- See model performance metrics
- Access SHAP visualizations

## Quick Reference

### Installation
```bash
pip install -r requirements.txt
```

### Run Core Notebooks
```bash
jupyter notebook
# Then run notebooks in order: 00 → 01 → 02 → 03
```

### Run Advanced Notebooks (Optional)
```bash
jupyter notebook
# Run: 04 (SMOTE), 05 (Feature Engineering)
```

### Launch Web App
```bash
streamlit run app.py
# Opens in browser at localhost:8501
```

### Expected Runtime
**Core Notebooks:**
- Setup: 5 minutes
- Week 1: 20 minutes
- Week 2: 15 minutes
- Week 3: 30-45 minutes (or 10 minutes with SKIP_TUNING=True)

**Advanced Notebooks (Optional):**
- SMOTE Sampling: 15 minutes
- Feature Engineering: 10 minutes

### Key Outputs
- **Models**: `models/final_xgboost.pkl`, `models/best_sampling_model.pkl`
- **Best Figures**: `figures/shap_feature_importance.png`, `figures/model_comparison.png`
- **Datasets**: `data/processed/hits_dataset.csv`, `data/processed/hits_dataset_engineered.csv`
- **Web App**: Interactive prediction interface

---

**Happy Analyzing! 🎵🎶**
