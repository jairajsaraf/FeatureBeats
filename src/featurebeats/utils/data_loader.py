"""
Utilities for loading and processing Spotify dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import logging

from ..models.spotify_track import SpotifyTrack, SpotifyDatasetStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpotifyDataLoader:
    """
    Loader for Spotify dataset with validation and processing capabilities.
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the data.csv file. If None, uses default location.
        """
        if data_path is None:
            # Default to data/raw/data.csv
            self.data_path = Path(__file__).parents[3] / "data" / "raw" / "data.csv"
        else:
            self.data_path = Path(data_path)

        self.df: Optional[pd.DataFrame] = None
        self.validated_tracks: List[SpotifyTrack] = []

    def load_csv(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Spotify CSV file into a pandas DataFrame.

        Args:
            nrows: Number of rows to load (useful for testing). None loads all.

        Returns:
            DataFrame containing the Spotify data.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.data_path}\n"
                "Please download the dataset from: "
                "https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020\n"
                "And place data.csv in the data/raw/ directory."
            )

        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, nrows=nrows)
        logger.info(f"Loaded {len(self.df)} tracks")

        return self.df

    def validate_data(
        self,
        df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None,
        raise_on_error: bool = False
    ) -> tuple[List[SpotifyTrack], List[Dict[str, Any]]]:
        """
        Validate data against the SpotifyTrack schema.

        Args:
            df: DataFrame to validate. If None, uses self.df.
            sample_size: Validate only a random sample (for large datasets).
            raise_on_error: If True, raise exception on first validation error.

        Returns:
            Tuple of (valid_tracks, validation_errors)
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_csv() first.")
            df = self.df

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Validating sample of {sample_size} tracks")

        valid_tracks: List[SpotifyTrack] = []
        errors: List[Dict[str, Any]] = []

        logger.info("Validating data against schema...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            try:
                track = SpotifyTrack(**row.to_dict())
                valid_tracks.append(track)
            except Exception as e:
                error_info = {
                    'index': idx,
                    'id': row.get('id', 'unknown'),
                    'name': row.get('name', 'unknown'),
                    'error': str(e)
                }
                errors.append(error_info)

                if raise_on_error:
                    raise

        logger.info(f"Validation complete: {len(valid_tracks)} valid, {len(errors)} errors")

        if errors:
            logger.warning(f"Found {len(errors)} validation errors")
            # Log first few errors
            for err in errors[:5]:
                logger.warning(f"Error at index {err['index']}: {err['error']}")

        self.validated_tracks = valid_tracks
        return valid_tracks, errors

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess the dataset.

        Args:
            df: DataFrame to clean. If None, uses self.df.

        Returns:
            Cleaned DataFrame.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_csv() first.")
            df = self.df.copy()
        else:
            df = df.copy()

        logger.info("Cleaning data...")

        # Remove duplicates based on track ID
        initial_len = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate tracks")

        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.info("Missing values per column:")
            logger.info(missing_counts[missing_counts > 0])

            # Drop rows with missing critical fields
            critical_fields = ['id', 'name', 'artists', 'year']
            df = df.dropna(subset=critical_fields)

            # Fill missing numerical features with median
            numerical_features = [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'speechiness', 'valence', 'tempo', 'loudness',
                'popularity', 'duration_ms'
            ]
            for feature in numerical_features:
                if feature in df.columns and df[feature].isnull().any():
                    median_val = df[feature].median()
                    df[feature].fillna(median_val, inplace=True)
                    logger.info(f"Filled {feature} missing values with median: {median_val:.2f}")

        # Ensure correct data types
        df['year'] = df['year'].astype(int)
        df['duration_ms'] = df['duration_ms'].astype(int)
        df['popularity'] = df['popularity'].astype(int)
        df['key'] = df['key'].astype(int)
        df['mode'] = df['mode'].astype(int)
        df['explicit'] = df['explicit'].astype(int)

        # Add derived features
        df['duration_seconds'] = df['duration_ms'] / 1000.0
        df['duration_minutes'] = df['duration_ms'] / 60000.0
        df['decade'] = (df['year'] // 10) * 10

        logger.info(f"Cleaning complete. Final dataset size: {len(df)} tracks")
        self.df = df

        return df

    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> SpotifyDatasetStats:
        """
        Calculate statistics about the dataset.

        Args:
            df: DataFrame to analyze. If None, uses self.df.

        Returns:
            SpotifyDatasetStats object.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_csv() first.")
            df = self.df

        # Count unique artists (handles list format)
        unique_artists = set()
        for artists_str in df['artists'].dropna():
            # Simple split by comma or bracket
            artists = str(artists_str).replace('[', '').replace(']', '').replace("'", "").split(',')
            unique_artists.update([a.strip() for a in artists if a.strip()])

        stats = SpotifyDatasetStats(
            total_tracks=len(df),
            unique_artists=len(unique_artists),
            year_range=(int(df['year'].min()), int(df['year'].max())),
            avg_duration_ms=float(df['duration_ms'].mean()),
            avg_popularity=float(df['popularity'].mean()),
            avg_tempo=float(df['tempo'].mean()),
            most_common_key=int(df['key'].mode()[0]),
            most_common_mode=int(df['mode'].mode()[0]),
            explicit_percentage=float((df['explicit'] == 1).mean() * 100)
        )

        return stats

    def get_feature_matrix(
        self,
        df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract feature matrix for ML models.

        Args:
            df: DataFrame to extract from. If None, uses self.df.
            features: List of feature names. If None, uses default audio features.

        Returns:
            NumPy array of shape (n_samples, n_features).
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data loaded. Call load_csv() first.")
            df = self.df

        if features is None:
            # Default audio features for ML
            features = [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'speechiness', 'valence', 'tempo', 'loudness',
                'duration_ms', 'key', 'mode', 'explicit'
            ]

        return df[features].values

    def export_cleaned_data(self, output_path: Optional[Path] = None) -> Path:
        """
        Export cleaned data to CSV.

        Args:
            output_path: Path for output file. If None, uses data/processed/cleaned_data.csv.

        Returns:
            Path to exported file.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        if output_path is None:
            output_path = Path(__file__).parents[3] / "data" / "processed" / "cleaned_data.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Exported cleaned data to {output_path}")

        return output_path


def quick_load(
    data_path: Optional[Path] = None,
    clean: bool = True,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Quick load and optionally clean the Spotify dataset.

    Args:
        data_path: Path to data.csv. If None, uses default location.
        clean: Whether to apply cleaning.
        nrows: Number of rows to load (for testing).

    Returns:
        DataFrame with Spotify data.
    """
    loader = SpotifyDataLoader(data_path)
    df = loader.load_csv(nrows=nrows)

    if clean:
        df = loader.clean_data(df)

    return df
