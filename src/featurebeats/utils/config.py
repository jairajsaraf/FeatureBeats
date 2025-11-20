"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for FeatureBeats.
    Loads settings from config.yaml and environment variables.
    """

    def __init__(self, config_path: Optional[Path] = None, load_env: bool = True):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
            load_env: Whether to load .env file.
        """
        # Load environment variables
        if load_env:
            env_path = Path(__file__).parents[3] / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment variables from {env_path}")

        # Load YAML config
        if config_path is None:
            config_path = Path(__file__).parents[3] / "config.yaml"

        self.config_path = config_path
        self._config: Dict[str, Any] = {}

        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")

        # Set project root
        self.project_root = Path(__file__).parents[3]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'data.raw_data_dir')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value
        """
        return os.getenv(key, default)

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        data_dir = self.get('data.raw_data_dir', 'data/raw')
        return self.project_root / data_dir

    @property
    def processed_data_dir(self) -> Path:
        """Get processed data directory path."""
        processed_dir = self.get('data.processed_data_dir', 'data/processed')
        return self.project_root / processed_dir

    @property
    def spotify_csv_path(self) -> Path:
        """Get path to Spotify CSV file."""
        csv_path = self.get('data.spotify_csv', 'data/raw/data.csv')
        return self.project_root / csv_path

    @property
    def random_seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.get('training.random_seed', 42)

    @property
    def test_size(self) -> float:
        """Get test set size."""
        return self.get('training.test_size', 0.2)

    @property
    def audio_features(self) -> list:
        """Get list of audio features."""
        return self.get('training.features.audio_features', [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'speechiness', 'valence', 'tempo', 'loudness'
        ])

    @property
    def all_features(self) -> list:
        """Get all features (audio + metadata)."""
        audio = self.audio_features
        metadata = self.get('training.features.metadata_features', [
            'duration_ms', 'key', 'mode', 'explicit', 'year'
        ])
        return audio + metadata

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


# Global config instance
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Args:
        reload: Whether to reload configuration

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config()

    return _config_instance
