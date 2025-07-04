# BayesianResampling

A novel oversampling method to balance continuous target variable datasets into a "more uniform" distribution with the goal of exposing less frequent ranges of data to an ML model.

## Overview

Machine learning models often struggle with long-tailed distributions in regression problems, where rare but valuable samples (like high-affinity molecules or extreme events) are underrepresented. This package provides tools to address this imbalance using Bayesian blocks to find optimal data partitions and resampling strategies to create more uniform density distributions.

## Key Features

- **Adaptive Partitioning**: Uses Bayesian blocks algorithm to find statistically optimal partitions of the target variable
- **Two Resampling Strategies**:
  - `BayesianBlocksResampler`: Balances the distribution through both oversampling and undersampling
  - `BayesianBlocksOversampler`: Preserves all original data points while only oversampling underrepresented regions
- **Parallel Processing**: Optimizes hyperparameters using parallel computing for efficiency
- **Flexible Input Formats**: Handles various data types including:
  - NumPy arrays
  - Pandas DataFrames
  - Sparse matrices
  - AnnData objects
- **Metadata Preservation**: Retains column names, indices, and other metadata throughout the resampling process

## Methods Explained

### Bayesian Blocks Algorithm

At the core of both resamplers is the Bayesian blocks algorithm, which finds optimal piecewise-constant representations of data. The algorithm works by:

1. Finding the optimal segmentation of the target variable into blocks
2. Using mutual information and entropy-based metrics to score different partitions
3. Applying a regularization term to prevent overfitting with too many blocks

### Resampling Strategies

#### BayesianBlocksResampler

This implementation can both oversample and undersample to achieve the target uniformity:

- Blocks with low density (underrepresented regions) are oversampled
- Blocks with high density (overrepresented regions) are undersampled
- A target uniformity parameter controls how aggressively to balance the distribution

#### BayesianBlocksOversampler

This implementation only oversamples, preserving all original data points:

- Blocks with low density are oversampled to approach the target uniformity
- Blocks with sufficient density are left unchanged
- This approach never removes any original data points, only adds new ones

## Dependencies

The package requires:

- Python 3.7+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Astropy (for the Bayesian blocks algorithm)
- Optional: AnnData (for single-cell genomics data)


### Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from sklearn.datasets import make_regression
from bayesian_resampling import BayesianBlocksResampler

# Create a synthetic long-tailed dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=5)
y = np.exp(y / y.std())  # Make it long-tailed

# Resample to create a more balanced distribution
resampler = BayesianBlocksResampler(target_uniformity=0.8, random_state=42)
X_balanced, y_balanced = resampler.fit_resample(X, y, verbose=True)
```

### Oversampling-Only Strategy

```python
from bayesian_resampling import BayesianBlocksOversampler

# Using the oversampling-only strategy (preserves all original data points)
oversampler = BayesianBlocksOversampler(target_uniformity=0.8, random_state=42)
X_balanced, y_balanced = oversampler.fit_resample(X, y, verbose=True)
```

### Convenience Functions

```python
from bayesian_resampling import bayesian_blocks_resample, bayesian_blocks_oversample

# Using the convenience functions
X_balanced, y_balanced = bayesian_blocks_resample(X, y, target_uniformity=0.8, verbose=True)
X_balanced2, y_balanced2 = bayesian_blocks_oversample(X, y, target_uniformity=0.8, verbose=True)
```

## Detailed API Documentation

### BayesianBlocksResampler

```python
resampler = BayesianBlocksResampler(
    target_uniformity=0.75,  # How uniform to make the distribution (0=no change, 1=perfectly uniform)
    random_state=None,       # Random seed for reproducibility
    preserve_metadata=True,  # Whether to preserve column names, indices, etc.
    n_jobs=-1                # Number of parallel jobs (-1 = all available cores)
)

# Find optimal alpha parameter (optional if you know a good value)
alpha = resampler.find_optimal_alpha(
    y,                      # Target values
    initial_points=100,     # Number of alpha values to try initially
    refinement_levels=3,    # Number of refinement iterations
    verbose=False           # Whether to print progress information
)

# Resample the data
X_resampled, y_resampled = resampler.fit_resample(
    X,                      # Feature matrix
    y,                      # Target values
    find_alpha=True,        # Whether to find optimal alpha
    alpha=None,             # Alpha value to use (if find_alpha=False)
    verbose=False           # Whether to print progress information
)

# Get block assignments for new data
block_assignments = resampler.get_block_assignments(y_new)
```

### BayesianBlocksOversampler

The API for `BayesianBlocksOversampler` is identical to `BayesianBlocksResampler`, but with a different resampling strategy (oversampling only).

## Technical Details

### Mutual Information Optimization

The optimal partitioning is determined by maximizing the mutual information between the target variable and block assignments, with regularization to prevent overfitting. This is done through a hierarchical grid search over the alpha parameter:

1. Start with a coarse grid over a wide range of alpha values
2. Evaluate the score for each alpha value in parallel
3. Identify promising regions with high mutual information
4. Refine the search in those regions

### DataHandler

The `DataHandler` class follows the Adapter pattern to handle different input data types:

- Detects the input type and converts it to a standard format for processing
- Preserves metadata like column names, indices, and data types
- Reconstructs the output in the same format as the input after resampling

## Benchmarks

TBD

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
@software{bayesian_resampling,
  author = {Ethan Loo},
  title = {BayesianResampling: Density Balancing for Regression Problems},
  year = {2025},
  url = {https://github.com/Ethan-Loo/BayesianResampling}
}
```