# Contributing to FeatureBeats

Thank you for your interest in contributing to FeatureBeats! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/FeatureBeats.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit and push
7. Create a Pull Request

## Development Setup

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode with all dependencies
pip install -e ".[all]"
```

### Download Dataset

```bash
# Place your Kaggle credentials in ~/.kaggle/kaggle.json
# Then download the dataset
kaggle datasets download -d ektanegi/spotifydata-19212020
unzip spotifydata-19212020.zip -d data/raw/
```

## Code Quality

### Formatting

We use `black` for code formatting:

```bash
black src/ tests/
```

### Linting

We use `ruff` for linting:

```bash
ruff src/ tests/
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy src/
```

### Run All Quality Checks

```bash
# Format
black src/ tests/

# Lint
ruff src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/ -v
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=featurebeats --cov-report=html

# Run specific test file
pytest tests/test_spotify_track.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use clear, descriptive test names
- Include docstrings explaining what each test does
- Aim for high code coverage

Example:

```python
def test_feature_name():
    """Test that feature works correctly with valid input."""
    # Arrange
    data = get_sample_data()

    # Act
    result = function_to_test(data)

    # Assert
    assert result == expected_output
```

## Pull Request Process

1. **Update Documentation**: If you add new features, update the README.md and relevant docstrings
2. **Add Tests**: Ensure your code is tested and doesn't break existing tests
3. **Follow Code Style**: Use black, ruff, and mypy to ensure code quality
4. **Write Clear Commits**: Use descriptive commit messages
5. **Update CHANGELOG**: Add a note about your changes
6. **Small PRs**: Keep pull requests focused on a single feature or fix

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
Add feature for tempo prediction

- Implement tempo prediction model
- Add tests for tempo predictor
- Update documentation
```

Good commit messages:
- Use present tense ("Add feature" not "Added feature")
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep first line under 50 characters
- Add detailed description if needed

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions, classes, and modules
- Keep functions small and focused
- Use descriptive variable names

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Longer description if needed, explaining the function's
    purpose and behavior.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is negative.
    """
    pass
```

## Project Structure

```
FeatureBeats/
├── src/featurebeats/      # Main package code
│   ├── models/            # Data models and schemas
│   ├── utils/             # Utility functions
│   └── analysis/          # Analysis scripts
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── examples/              # Example scripts
├── data/                  # Data directory (not tracked)
└── docs/                  # Documentation
```

## Areas for Contribution

### Current Priorities

1. **Data Analysis**: Add new analysis functions and visualizations
2. **Machine Learning Models**: Implement popularity prediction, genre classification, etc.
3. **Feature Engineering**: Create new derived features from audio data
4. **Documentation**: Improve docs, add examples, write tutorials
5. **Testing**: Increase test coverage
6. **Performance**: Optimize data loading and processing

### Feature Ideas

- Song recommendation system
- Genre classification model
- Decade/era prediction
- Artist similarity analysis
- Playlist generation
- Time series analysis of music trends
- Integration with Spotify API for real-time data
- Interactive dashboards (Streamlit/Dash)

## Questions or Issues?

- Check existing issues on GitHub
- Create a new issue if your question isn't answered
- Tag issues appropriately (bug, enhancement, question, etc.)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person

## License

By contributing to FeatureBeats, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make FeatureBeats better for everyone. We appreciate your time and effort!
