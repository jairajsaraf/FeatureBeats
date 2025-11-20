"""
Visualization utilities for Spotify data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot distributions of audio features.

    Args:
        df: DataFrame with Spotify data.
        features: List of features to plot. If None, plots all audio features.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if features is None:
        features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'speechiness', 'valence', 'loudness', 'tempo'
        ]

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        if feature in df.columns:
            axes[idx].hist(df[feature].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{feature.capitalize()} Distribution')
            axes[idx].set_xlabel(feature.capitalize())
            axes[idx].set_ylabel('Frequency')

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_temporal_trends(
    df: pd.DataFrame,
    feature: str = 'popularity',
    group_by: str = 'year',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot how a feature changes over time.

    Args:
        df: DataFrame with Spotify data.
        feature: Feature to plot (e.g., 'popularity', 'tempo', 'energy').
        group_by: Time grouping ('year' or 'decade').
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Average over time
    temporal_avg = df.groupby(group_by)[feature].mean()
    temporal_avg.plot(ax=ax1, marker='o', linewidth=2)
    ax1.set_title(f'Average {feature.capitalize()} by {group_by.capitalize()}')
    ax1.set_xlabel(group_by.capitalize())
    ax1.set_ylabel(f'Average {feature.capitalize()}')
    ax1.grid(True, alpha=0.3)

    # Distribution over time (boxplot for decades)
    if group_by == 'decade' or 'decade' in df.columns:
        decade_col = 'decade' if 'decade' in df.columns else group_by
        decades = sorted(df[decade_col].unique())
        data_by_decade = [df[df[decade_col] == d][feature].dropna() for d in decades]

        ax2.boxplot(data_by_decade, labels=decades)
        ax2.set_title(f'{feature.capitalize()} Distribution by Decade')
        ax2.set_xlabel('Decade')
        ax2.set_ylabel(feature.capitalize())
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot correlation matrix of features.

    Args:
        df: DataFrame with Spotify data.
        features: List of features to include. If None, uses all numerical features.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if features is None:
        features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'speechiness', 'valence', 'tempo', 'loudness',
            'duration_ms', 'popularity', 'key', 'mode'
        ]

    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    corr_matrix = df[available_features].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)

    plt.tight_layout()
    return fig


def plot_top_artists(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot top artists by track count.

    Args:
        df: DataFrame with Spotify data.
        top_n: Number of top artists to show.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    # Parse artists (simple approach - split by comma)
    all_artists = []
    for artists_str in df['artists'].dropna():
        artists = str(artists_str).replace('[', '').replace(']', '').replace("'", "").split(',')
        all_artists.extend([a.strip() for a in artists if a.strip()])

    artist_counts = pd.Series(all_artists).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    artist_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Top {top_n} Artists by Track Count', fontsize=14)
    ax.set_xlabel('Number of Tracks')
    ax.set_ylabel('Artist')
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_key_mode_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot distribution of musical keys and modes.

    Args:
        df: DataFrame with Spotify data.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Key distribution
    key_counts = df['key'].value_counts().sort_index()
    ax1.bar(range(12), [key_counts.get(i, 0) for i in range(12)], color='skyblue', edgecolor='black')
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(key_names)
    ax1.set_title('Distribution of Musical Keys')
    ax1.set_xlabel('Key')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.3)

    # Mode distribution
    mode_counts = df['mode'].value_counts()
    ax2.bar(['Minor', 'Major'], [mode_counts.get(0, 0), mode_counts.get(1, 0)],
            color=['coral', 'lightgreen'], edgecolor='black')
    ax2.set_title('Distribution of Modes')
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Frequency')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_popularity_vs_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot popularity vs various audio features.

    Args:
        df: DataFrame with Spotify data.
        features: List of features to plot against popularity.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if features is None:
        features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness']

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        if feature in df.columns:
            axes[idx].scatter(df[feature], df['popularity'], alpha=0.3, s=10)
            axes[idx].set_xlabel(feature.capitalize())
            axes[idx].set_ylabel('Popularity')
            axes[idx].set_title(f'Popularity vs {feature.capitalize()}')

            # Add trend line
            z = np.polyfit(df[feature].dropna(), df.loc[df[feature].notna(), 'popularity'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
            axes[idx].plot(x_trend, p(x_trend), 'r-', linewidth=2, alpha=0.7)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig
