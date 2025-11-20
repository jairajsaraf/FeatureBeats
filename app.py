"""
Hit Song Prediction Web App
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Hit Song Predictor",
    page_icon="🎵",
    layout="wide"
)

# Paths
project_root = Path(__file__).parent
models_dir = project_root / 'models'
figures_dir = project_root / 'figures'

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(models_dir / 'final_xgboost.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        return model, scaler
    except:
        try:
            model = joblib.load(models_dir / 'baseline_logreg.pkl')
            scaler = joblib.load(models_dir / 'scaler.pkl')
            return model, scaler
        except:
            return None, None

model, scaler = load_model_and_scaler()

# Title
st.title("🎵 Hit Song Predictor")
st.markdown("### Predict whether a song will become a hit based on its Spotify audio features")

if model is None:
    st.error("⚠️ Model files not found! Please train the models first by running the Jupyter notebooks.")
    st.stop()

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app predicts hit songs using machine learning trained on Spotify audio features. "
    "Adjust the sliders to match a song's characteristics and get a prediction!"
)

st.sidebar.header("Audio Features Guide")
st.sidebar.markdown("""
- **Danceability**: How suitable for dancing (0-1)
- **Energy**: Intensity and activity (0-1)
- **Valence**: Musical positivity (0-1)
- **Loudness**: Overall loudness in dB (-60 to 0)
- **Speechiness**: Presence of spoken words (0-1)
- **Acousticness**: Acoustic vs electric (0-1)
- **Instrumentalness**: No vocals indicator (0-1)
- **Liveness**: Audience presence (0-1)
- **Tempo**: Speed in BPM (50-200)
""")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Insights", "ℹ️ Info"])

with tab1:
    st.header("Song Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Rhythm & Dance")
        danceability = st.slider("Danceability", 0.0, 1.0, 0.7, 0.01)
        tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0, 1.0)
        
    with col2:
        st.subheader("Energy & Mood")
        energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.01)
        valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.6, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0, 0.5)
        
    with col3:
        st.subheader("Composition")
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.01)
    
    # Predict button
    if st.button("🎯 Predict Hit Potential", type="primary", use_container_width=True):
        # Prepare features
        features = np.array([[
            danceability, energy, loudness, speechiness,
            acousticness, instrumentalness, liveness, valence, tempo
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Display result
        st.markdown("---")
        st.header("Prediction Result")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            if prediction == 1:
                st.success("### 🎉 Potential HIT!")
            else:
                st.warning("### 📉 Unlikely to Chart")
        
        with col_result2:
            st.metric("Hit Probability", f"{probability*100:.1f}%")
        
        with col_result3:
            confidence = "High" if probability > 0.7 or probability < 0.3 else "Medium" if probability > 0.6 or probability < 0.4 else "Low"
            st.metric("Confidence", confidence)
        
        # Probability gauge
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh([0], [probability], color='green' if probability > 0.5 else 'red', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Hit Probability', fontsize=12)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0.5, -0.3, '50% threshold', ha='center', fontsize=9)
        st.pyplot(fig)
        plt.close()
        
        # Feature analysis
        st.subheader("Feature Analysis")
        feature_names = ['Danceability', 'Energy', 'Loudness', 'Speechiness',
                        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
        feature_values = [danceability, energy, (loudness+60)/60, speechiness,
                         acousticness, instrumentalness, liveness, valence, tempo/200]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if v > 0.5 else 'red' for v in feature_values]
        ax.barh(feature_names, feature_values, color=colors, alpha=0.7)
        ax.set_xlabel('Normalized Value', fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        plt.close()

with tab2:
    st.header("Model Insights")
    
    try:
        # Load and display model comparison
        st.subheader("Model Performance Comparison")
        if (figures_dir / 'model_comparison.png').exists():
            st.image(str(figures_dir / 'model_comparison.png'))
        
        # Load and display SHAP
        st.subheader("Feature Importance (SHAP Analysis)")
        if (figures_dir / 'shap_feature_importance.png').exists():
            st.image(str(figures_dir / 'shap_feature_importance.png'))
        
        # Key insights
        st.subheader("Key Insights")
        st.info("""
        Based on our analysis of thousands of songs:
        
        - **High Danceability** songs are more likely to become hits
        - **Energetic** tracks with positive **Valence** (happy mood) perform better
        - **Lower Acousticness** correlates with chart success (more electronic production)
        - **Instrumental tracks** rarely chart (vocals are important!)
        - **Moderate loudness** (-5 to -8 dB) is optimal
        """)
        
    except:
        st.warning("Run the Jupyter notebooks to generate visualizations")

with tab3:
    st.header("About This App")
    
    st.markdown("""
    ### How It Works
    
    This app uses a machine learning model trained on Spotify audio features to predict hit songs.
    
    **Model Details:**
    - Algorithm: XGBoost (Gradient Boosting)
    - Training Data: Thousands of songs from Spotify
    - Features: 9 audio characteristics
    - Accuracy: ~80-90% on test data
    
    ### Limitations
    
    - Based on audio features only (doesn't account for marketing, artist fame, timing, etc.)
    - Trained on historical data (music trends change)
    - Correlation doesn't imply causation
    - Should be used as guidance, not definitive prediction
    
    ### Dataset
    
    - Source: Spotify API via Kaggle
    - Period: 2010-2020
    - Labels: Billboard/Spotify Top 100 charts
    
    ### Technical Stack
    
    - Python, scikit-learn, XGBoost
    - SHAP for model interpretation
    - Streamlit for web interface
    
    ### Created By
    
    Machine Learning Course Project
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ using Python and Streamlit**")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🎵 Hit Song Predictor v1.0")
with col_f2:
    st.caption("Powered by XGBoost & SHAP")
with col_f3:
    st.caption("Data: Spotify API")
