"""
Quick start example for FeatureBeats.

This script demonstrates basic usage of the FeatureBeats library
for loading and analyzing Spotify data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from featurebeats.utils.data_loader import quick_load, SpotifyDataLoader
from featurebeats.utils.visualization import (
    plot_feature_distributions,
    plot_temporal_trends,
    plot_correlation_matrix,
)
from featurebeats.models.spotify_track import SpotifyTrack


def main():
    """Main function demonstrating FeatureBeats usage."""

    print("=" * 60)
    print("FeatureBeats - Quick Start Example")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading Spotify dataset...")
    try:
        # For a quick test, load only 1000 rows
        # Remove nrows parameter to load full dataset
        df = quick_load(nrows=1000)
        print(f"   Loaded {len(df)} tracks")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print("\n   Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020")
        print("   And place data.csv in the data/raw/ directory")
        return

    # 2. Basic statistics
    print("\n2. Dataset Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Years: {df['year'].min()} - {df['year'].max()}")
    print(f"   Average popularity: {df['popularity'].mean():.1f}")
    print(f"   Average tempo: {df['tempo'].mean():.1f} BPM")

    # 3. Explore a single track
    print("\n3. Exploring a sample track:")
    sample_row = df.iloc[0]
    track = SpotifyTrack(**sample_row.to_dict())

    print(f"   Name: {track.name}")
    print(f"   Artists: {track.artists}")
    print(f"   Year: {track.year} ({track.era})")
    print(f"   Duration: {track.get_duration_minutes():.2f} minutes")
    print(f"   Key: {track.get_musical_key_name()} {track.get_mode_name()}")
    print(f"   Tempo: {track.tempo:.1f} BPM")
    print(f"   Popularity: {track.popularity}/100")
    print(f"   Explicit: {'Yes' if track.is_explicit() else 'No'}")

    # 4. Feature analysis
    print("\n4. Audio Feature Averages:")
    audio_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'speechiness', 'valence'
    ]

    for feature in audio_features:
        avg = df[feature].mean()
        print(f"   {feature.capitalize():20s}: {avg:.3f}")

    # 5. Decade analysis
    print("\n5. Average Energy by Decade:")
    decade_energy = df.groupby('decade')['energy'].mean()
    for decade, energy in decade_energy.items():
        print(f"   {decade}s: {energy:.3f}")

    # 6. Popularity insights
    print("\n6. Popularity Analysis:")
    popular_threshold = 70
    popular_tracks = df[df['popularity'] >= popular_threshold]
    print(f"   Tracks with popularity >= {popular_threshold}: {len(popular_tracks)}")
    print(f"   Percentage: {len(popular_tracks)/len(df)*100:.1f}%")

    if len(popular_tracks) > 0:
        print(f"\n   Average features of popular tracks:")
        for feature in ['danceability', 'energy', 'valence']:
            avg = popular_tracks[feature].mean()
            print(f"   {feature.capitalize():20s}: {avg:.3f}")

    # 7. Feature correlation with popularity
    print("\n7. Top Features Correlated with Popularity:")
    correlations = df[audio_features].corrwith(df['popularity']).sort_values(ascending=False)
    for feature, corr in correlations.head(3).items():
        print(f"   {feature.capitalize():20s}: {corr:+.3f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Explore the full dataset by removing the nrows parameter")
    print("- Run the EDA notebook: notebooks/01_exploratory_data_analysis.ipynb")
    print("- Check out the visualization examples in the notebook")
    print("- Build your own ML models for popularity prediction")
    print("=" * 60)


if __name__ == "__main__":
    main()
