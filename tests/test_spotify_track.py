"""
Tests for SpotifyTrack model.
"""

import pytest
from pydantic import ValidationError
from featurebeats.models.spotify_track import SpotifyTrack


def get_sample_track_data():
    """Return sample track data for testing."""
    return {
        'id': '7qiZfU4dY1lWllzX7mPBI',
        'name': 'Test Song',
        'artists': "['Artist One', 'Artist Two']",
        'release_date': '2020-01-15',
        'acousticness': 0.5,
        'danceability': 0.7,
        'energy': 0.8,
        'instrumentalness': 0.1,
        'liveness': 0.2,
        'speechiness': 0.05,
        'valence': 0.6,
        'duration_ms': 200000,
        'popularity': 75,
        'tempo': 120.0,
        'loudness': -5.0,
        'year': 2020,
        'key': 0,
        'mode': 1,
        'explicit': 0
    }


class TestSpotifyTrack:
    """Test cases for SpotifyTrack model."""

    def test_valid_track_creation(self):
        """Test creating a valid SpotifyTrack."""
        data = get_sample_track_data()
        track = SpotifyTrack(**data)

        assert track.id == data['id']
        assert track.name == data['name']
        assert track.year == 2020
        assert track.popularity == 75

    def test_get_musical_key_name(self):
        """Test musical key name conversion."""
        data = get_sample_track_data()
        track = SpotifyTrack(**data)

        assert track.get_musical_key_name() == 'C'

        # Test with different keys
        data['key'] = 1
        track = SpotifyTrack(**data)
        assert track.get_musical_key_name() == 'C#'

        data['key'] = 11
        track = SpotifyTrack(**data)
        assert track.get_musical_key_name() == 'B'

    def test_get_mode_name(self):
        """Test mode name conversion."""
        data = get_sample_track_data()

        # Test Major mode
        data['mode'] = 1
        track = SpotifyTrack(**data)
        assert track.get_mode_name() == 'Major'

        # Test Minor mode
        data['mode'] = 0
        track = SpotifyTrack(**data)
        assert track.get_mode_name() == 'Minor'

    def test_is_explicit(self):
        """Test explicit content check."""
        data = get_sample_track_data()

        data['explicit'] = 0
        track = SpotifyTrack(**data)
        assert track.is_explicit() is False

        data['explicit'] = 1
        track = SpotifyTrack(**data)
        assert track.is_explicit() is True

    def test_duration_conversions(self):
        """Test duration conversion methods."""
        data = get_sample_track_data()
        data['duration_ms'] = 180000  # 3 minutes
        track = SpotifyTrack(**data)

        assert track.get_duration_seconds() == 180.0
        assert track.get_duration_minutes() == 3.0

    def test_decade_property(self):
        """Test decade property calculation."""
        data = get_sample_track_data()

        data['year'] = 1985
        track = SpotifyTrack(**data)
        assert track.decade == 1980

        data['year'] = 2005
        track = SpotifyTrack(**data)
        assert track.decade == 2000

    def test_era_property(self):
        """Test era categorization."""
        data = get_sample_track_data()

        test_cases = [
            (1945, "Pre-1950s"),
            (1955, "1950s"),
            (1965, "1960s"),
            (1975, "1970s"),
            (1985, "1980s"),
            (1995, "1990s"),
            (2005, "2000s"),
            (2015, "2010s"),
            (2020, "2020s"),
        ]

        for year, expected_era in test_cases:
            data['year'] = year
            track = SpotifyTrack(**data)
            assert track.era == expected_era, f"Year {year} should be in {expected_era}"

    def test_get_artists_list(self):
        """Test parsing artists string into list."""
        data = get_sample_track_data()

        # Test with list format
        data['artists'] = "['Artist One', 'Artist Two']"
        track = SpotifyTrack(**data)
        artists = track.get_artists_list()
        assert len(artists) == 2
        assert 'Artist One' in artists
        assert 'Artist Two' in artists

    def test_invalid_acousticness(self):
        """Test validation of acousticness range."""
        data = get_sample_track_data()
        data['acousticness'] = 1.5  # Invalid: > 1.0

        with pytest.raises(ValidationError):
            SpotifyTrack(**data)

    def test_invalid_popularity(self):
        """Test validation of popularity range."""
        data = get_sample_track_data()
        data['popularity'] = 150  # Invalid: > 100

        with pytest.raises(ValidationError):
            SpotifyTrack(**data)

    def test_invalid_key(self):
        """Test validation of musical key range."""
        data = get_sample_track_data()
        data['key'] = 15  # Invalid: > 11

        with pytest.raises(ValidationError):
            SpotifyTrack(**data)

    def test_invalid_year(self):
        """Test validation of year range."""
        data = get_sample_track_data()
        data['year'] = 1900  # Invalid: < 1921

        with pytest.raises(ValidationError):
            SpotifyTrack(**data)

    def test_empty_name(self):
        """Test validation of required name field."""
        data = get_sample_track_data()
        data['name'] = ''

        # Pydantic will coerce empty string, but it should still work
        # as the field is required
        track = SpotifyTrack(**data)
        assert track.name == ''

    def test_missing_required_field(self):
        """Test that missing required fields raise error."""
        data = get_sample_track_data()
        del data['id']

        with pytest.raises(ValidationError):
            SpotifyTrack(**data)
