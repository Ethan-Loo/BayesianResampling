"""
Bayesian Blocks Density Balancing for Regression

This module provides tools for addressing long-tailed distributions in regression
problems by using Bayesian blocks to find optimal data partitions and then
resampling to create more uniform density distributions.

Author: EL
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from functools import partial

# Find global number of CPU cores using os
def get_global_cpu_count() -> int:
    return mp.cpu_count() - 1

try:
    from astropy.stats import bayesian_blocks
except ImportError:
    warnings.warn(
        "astropy not installed. Please install with: pip install astropy",
        ImportWarning
    )

@dataclass
class BlockInfo:
    """Container for information about a single block"""
    lower_edge: float
    upper_edge: float
    n_samples: int
    density: float
    indices: np.ndarray
    target_samples: int

# Let's also handle sparse matrices which are common in genomics
from scipy import sparse

class DataHandler:
    """
    A helper class that handles different input data types and converts them
    to a standard format for processing, while preserving metadata.
    
    This follows the Adapter pattern - we adapt various data types to our needs
    while keeping track of how to convert back.
    """
    
    def __init__(self):
        self.input_type = None
        self.metadata = {}

    def extract_data(self, X: any, y: any = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract numpy arrays from various data types while preserving metadata.
        
        This method detects the input type and calls the appropriate handler.
        """
        # Handle y first if it exists (it's usually simpler)
        if y is not None:
            y_array = self._extract_target(y)
        else:
            y_array = None
            
        # Handle X based on its type
        if isinstance(X, pd.DataFrame):
            X_array = self._extract_from_dataframe(X)
        elif hasattr(X, 'X') and hasattr(X, 'obs'):  # Likely an AnnData object
            X_array = self._extract_from_anndata(X)
        elif sparse.issparse(X):
            X_array = self._extract_from_sparse(X)
        elif isinstance(X, np.ndarray):
            X_array = X
            self.input_type = 'numpy'
        else:
            # Try to convert to numpy array as fallback
            try:
                X_array = np.asarray(X)
                self.input_type = 'array_like'
            except Exception as e:
                raise TypeError(f"Cannot convert input type {type(X)} to numpy array: {e}")
                
        return X_array, y_array
    
    def _extract_target(self, y: any) -> np.ndarray:
        """Extract target values as numpy array."""
        if isinstance(y, pd.Series):
            self.metadata['y_name'] = y.name
            self.metadata['y_index'] = y.index
            return y.values
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"y DataFrame must have exactly 1 column, got {y.shape[1]}")
            self.metadata['y_name'] = y.columns[0]
            self.metadata['y_index'] = y.index
            return y.values.ravel()
        elif isinstance(y, np.ndarray):
            return y.ravel()
        else:
            # Try to convert
            return np.asarray(y).ravel()
    
    def _extract_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Extract data from pandas DataFrame while preserving metadata."""
        self.input_type = 'pandas'
        self.metadata['columns'] = df.columns
        self.metadata['index'] = df.index
        self.metadata['dtypes'] = df.dtypes
        return df.values
    
    def _extract_from_anndata(self, adata: any) -> np.ndarray:
        """
        Extract data from AnnData object.
        
        AnnData objects are commonly used in single-cell genomics and have
        a specific structure we need to handle carefully.
        """
        self.input_type = 'anndata'
        
        # Store important metadata
        self.metadata['obs_names'] = adata.obs_names
        self.metadata['var_names'] = adata.var_names
        self.metadata['shape'] = adata.shape
        
        # Extract the main data matrix
        # AnnData can store data in different slots
        if hasattr(adata, 'X'):
            if sparse.issparse(adata.X):
                # Convert sparse to dense if needed
                # Be careful with memory for large datasets!
                if adata.X.shape[0] * adata.X.shape[1] > 1e8:
                    warnings.warn(
                        "Converting large sparse matrix to dense. "
                        "This may use significant memory.",
                        ResourceWarning
                    )
                return adata.X.toarray()
            else:
                return adata.X
        else:
            raise ValueError("AnnData object has no .X attribute")
    
    def _extract_from_sparse(self, X_sparse: any) -> np.ndarray:
        """Extract data from sparse matrix."""
        self.input_type = 'sparse'
        self.metadata['sparse_format'] = type(X_sparse).__name__
        self.metadata['shape'] = X_sparse.shape
        
        # Warning for large sparse matrices
        if X_sparse.shape[0] * X_sparse.shape[1] > 1e8:
            warnings.warn(
                "Converting large sparse matrix to dense. "
                "Consider using a different approach for very large sparse data.",
                ResourceWarning
            )
        
        return X_sparse.toarray()
    
    def reconstruct_output(self, X_array: np.ndarray, y_array: np.ndarray, 
                          indices: np.ndarray) -> Tuple[any, any]:
        """
        Reconstruct the output in the same format as the input.
        
        This is where we use the preserved metadata to return data in a format
        that matches what the user provided.
        """
        if self.input_type == 'pandas':
            return self._reconstruct_pandas(X_array, y_array, indices)
        elif self.input_type == 'anndata':
            return self._reconstruct_anndata(X_array, y_array, indices)
        elif self.input_type == 'sparse':
            # Convert back to sparse format
            X_sparse = sparse.csr_matrix(X_array)
            return X_sparse, y_array
        else:
            # Return as numpy arrays
            return X_array, y_array
    
    def _reconstruct_pandas(self, X_array: np.ndarray, y_array: np.ndarray, 
                           indices: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
        """Reconstruct pandas DataFrame and Series with appropriate metadata."""
        # Create new index based on the resampling
        # We'll use the original indices where possible
        if 'index' in self.metadata:
            original_index = self.metadata['index']
            # Map the resampled indices to original index values
            new_index = original_index[indices % len(original_index)]
            # Add suffix for duplicated indices
            if len(new_index) != len(set(new_index)):
                # Handle duplicates by adding a counter
                new_index = self._make_unique_index(new_index)
        else:
            new_index = None
            
        # Reconstruct DataFrame
        X_df = pd.DataFrame(
            X_array,
            columns=self.metadata.get('columns'),
            index=new_index
        )
        
        # Reconstruct Series
        y_series = pd.Series(
            y_array,
            name=self.metadata.get('y_name', 'target'),
            index=new_index
        )
        
        return X_df, y_series
    
    def _make_unique_index(self, index):
        """Make index unique by adding suffixes to duplicates."""
        counts = defaultdict(int)
        new_index = []
        
        for idx in index:
            if counts[idx] == 0:
                new_index.append(idx)
            else:
                new_index.append(f"{idx}_{counts[idx]}")
            counts[idx] += 1
            
        return pd.Index(new_index)
    
    def _reconstruct_anndata(self, X_array: np.ndarray, y_array: np.ndarray, 
                            indices: np.ndarray) -> Tuple[any, np.ndarray]:
        """
        Reconstruct AnnData object with resampled data.
        
        Note: This requires the anndata package to be installed.
        """
        try:
            import anndata
        except ImportError:
            warnings.warn(
                "anndata package not installed. Returning numpy arrays instead.",
                ImportWarning
            )
            return X_array, y_array
        
        # Create new obs names based on resampling
        original_obs = self.metadata['obs_names']
        new_obs_names = []
        
        # Track how many times each observation has been sampled
        obs_counts = defaultdict(int)
        
        for idx in indices:
            original_idx = idx % len(original_obs)
            obs_name = original_obs[original_idx]
            
            if obs_counts[obs_name] == 0:
                new_obs_names.append(obs_name)
            else:
                new_obs_names.append(f"{obs_name}_{obs_counts[obs_name]}")
            
            obs_counts[obs_name] += 1
        
        # Create new AnnData object
        adata_resampled = anndata.AnnData(
            X=X_array,
            obs=pd.DataFrame(index=new_obs_names),
            var=pd.DataFrame(index=self.metadata['var_names'])
        )
        
        # Add the target values as an observation
        adata_resampled.obs['target'] = y_array
        
        return adata_resampled, y_array


def _evaluate_alpha_batch(alpha_batch: List[float], y: np.ndarray, 
                         random_state: Optional[int] = None) -> List[Tuple[float, float, int]]:
    """
    Evaluate a batch of alpha values in parallel.
    
    Returns list of (alpha, score, n_blocks) tuples.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    results = []
    
    for alpha in alpha_batch:
        try:
            # Get Bayesian blocks for this alpha
            edges = bayesian_blocks(y, fitness='events', p0=alpha)
            n_blocks = len(edges) - 1
            
            # Skip if we get too many or too few blocks
            if n_blocks < 2 or n_blocks > len(y) // 10:
                results.append((alpha, -np.inf, n_blocks))
                continue
            
            # Calculate block assignments
            block_assignments = np.digitize(y, edges) - 1
            
            # Calculate mutual information
            mi = _calculate_mutual_information(y, block_assignments)
            
            # Add regularization to prefer simpler models
            regularization = 0.01 * np.log(n_blocks)
            score = mi - regularization
            
            results.append((alpha, score, n_blocks))
            
        except Exception as e:
            results.append((alpha, -np.inf, 0))
    
    return results


def _calculate_mutual_information(y: np.ndarray, block_assignments: np.ndarray) -> float:
    """
    Calculate mutual information between target values and block assignments.
    
    This is a standalone function so it can be used in parallel processing.
    """
    # Discretize y for entropy calculation
    n_bins = int(np.sqrt(len(y)))
    y_hist, y_edges = np.histogram(y, bins=n_bins)
    y_probs = y_hist / y_hist.sum()
    
    # Calculate entropy of y
    h_y = -np.sum(y_probs[y_probs > 0] * np.log(y_probs[y_probs > 0]))
    
    # Calculate conditional entropy H(Y|Block)
    h_y_given_block = 0
    unique_blocks = np.unique(block_assignments)
    
    for block_id in unique_blocks:
        block_mask = block_assignments == block_id
        y_in_block = y[block_mask]
        
        if len(y_in_block) == 0:
            continue
            
        p_block = block_mask.sum() / len(y)
        
        block_hist, _ = np.histogram(y_in_block, bins=y_edges)
        if block_hist.sum() > 0:
            block_probs = block_hist / block_hist.sum()
            block_entropy = -np.sum(block_probs[block_probs > 0] * 
                                   np.log(block_probs[block_probs > 0]))
            h_y_given_block += p_block * block_entropy
    
    return h_y - h_y_given_block


class BayesianBlocksOversampler:
    """
    A class for resampling regression datasets using Bayesian blocks to achieve
    more uniform density distributions, particularly useful for long-tailed data.
    
    The approach works by:
    1. Finding optimal blocks using Bayesian blocks algorithm
    2. Calculating density within each block
    3. Oversampling underrepresented blocks to achieve target uniformity
    
    Note: This implementation only oversamples (never undersamples) to preserve
    all original data diversity while addressing imbalanced distributions.
    """
    
    def __init__(self, target_uniformity: float = 0.75, 
                 random_state: Optional[int] = None,
                 preserve_metadata: bool = True,
                 n_jobs: int = -1):
        """
        Initialize the Bayesian Blocks Resampler.
        
        Parameters:
        -----------
        target_uniformity : float, default=0.8
            How much to boost underrepresented regions (0=no change, 1=perfectly uniform).
            A value of 0.8 means underrepresented blocks will be oversampled to reach
            75% of what they would need for perfect uniformity.
            
        random_state : int or None, default=None
            Random seed for reproducibility
        preserve_metadata : bool, default=True
            Whether to preserve and restore metadata (column names, indices, etc.)
            Setting to False can save memory for very large datasets.
        n_jobs : int, default=-1
            Number of parallel jobs for alpha discovery. -1 uses all available cores.
        """
        self.target_uniformity = target_uniformity
        self.random_state = random_state
        self.preserve_metadata = preserve_metadata
        self.n_jobs = n_jobs if n_jobs != -1 else get_global_cpu_count()
        self.alpha_ = None
        self.blocks_info_ = None
        self.edges_ = None
        self._data_handler = DataHandler() if preserve_metadata else None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def find_optimal_alpha(self, y: np.ndarray, 
                          initial_points: int = 100,
                          refinement_levels: int = 3,
                          verbose: bool = False) -> float:
        """
        Find the optimal alpha parameter using parallelized hierarchical grid search.
        
        This method explores the alpha parameter space efficiently by:
        1. Starting with a coarse grid over a wide range
        2. Identifying promising regions with high MI in parallel
        3. Refining the search in those regions
        
        Parameters:
        -----------
        y : array-like
            Target values to find optimal blocks for
        initial_points : int, default=100
            Number of alpha values to try in the initial coarse search
        refinement_levels : int, default=3
            Number of refinement iterations
        verbose : bool, default=False
            Whether to print progress information
            
        Returns:
        --------
        float : The optimal alpha value
        """
        # Start with a wide range of alpha values on a log scale
        current_alphas = np.logspace(-10, -0.1, initial_points)
        best_alpha = None
        best_score = -np.inf
        
        for level in range(refinement_levels):
            if verbose:
                print(f"\nRefinement level {level + 1}/{refinement_levels}")
                print(f"Evaluating {len(current_alphas)} alpha values using {self.n_jobs} processes")
            
            # Split alphas into batches for parallel processing
            batch_size = max(1, len(current_alphas) // self.n_jobs)
            alpha_batches = [current_alphas[i:i + batch_size] 
                           for i in range(0, len(current_alphas), batch_size)]
            
            # Process batches in parallel
            all_results = []
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(
                        _evaluate_alpha_batch, 
                        batch, 
                        y, 
                        self.random_state + i if self.random_state is not None else None
                    ): i for i, batch in enumerate(alpha_batches)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    all_results.extend(batch_results)
            
            # Process results
            level_scores = []
            level_n_blocks = []
            
            for alpha, score, n_blocks in all_results:
                level_scores.append(score)
                level_n_blocks.append(n_blocks)
                
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
                    if verbose:
                        print(f"  New best: alpha={alpha:.2e}, "
                              f"blocks={n_blocks}, score={score:.4f}")
            
            # For the next level, refine around the best regions
            if level < refinement_levels - 1:
                level_scores = np.array(level_scores)
                
                # Identify promising regions (local maxima)
                promising_indices = []
                for i in range(1, len(level_scores) - 1):
                    if (level_scores[i] > level_scores[i-1] and 
                        level_scores[i] > level_scores[i+1] and
                        level_scores[i] > -np.inf):
                        promising_indices.append(i)
                
                # Always include the global maximum
                if best_alpha in current_alphas:
                    best_idx = np.where(current_alphas == best_alpha)[0][0]
                    if best_idx not in promising_indices:
                        promising_indices.append(best_idx)
                
                # Create finer grid around promising alphas
                new_alphas = []
                for idx in promising_indices:
                    # Define a refined range around this alpha
                    if idx > 0 and idx < len(current_alphas) - 1:
                        alpha_low = current_alphas[idx - 1]
                        alpha_high = current_alphas[idx + 1]
                    else:
                        alpha_center = current_alphas[idx]
                        alpha_low = alpha_center * 0.5
                        alpha_high = alpha_center * 2.0
                    
                    # Create finer grid in this range
                    fine_alphas = np.logspace(
                        np.log10(alpha_low), 
                        np.log10(alpha_high), 
                        20
                    )
                    new_alphas.extend(fine_alphas)
                
                current_alphas = np.unique(new_alphas)
                
                if verbose:
                    print(f"  Found {len(promising_indices)} promising regions")
        
        self.alpha_ = best_alpha
        return best_alpha
    
    def _create_blocks(self, y: np.ndarray, alpha: Optional[float] = None) -> List[BlockInfo]:
        """
        Create blocks using Bayesian blocks algorithm and calculate block statistics.
        
        This method uses the new oversampling-only strategy.
        """
        if alpha is None:
            alpha = self.alpha_
        if alpha is None:
            raise ValueError("No alpha value provided and none fitted. "
                           "Run find_optimal_alpha first or provide alpha.")
        
        # Get the optimal blocks
        edges = bayesian_blocks(y, fitness='events', p0=alpha)
        self.edges_ = edges
        
        # Calculate statistics for each block
        blocks_info = []
        total_range = edges[-1] - edges[0]
        uniform_density = len(y) / total_range
        
        for i in range(len(edges) - 1):
            # Find samples in this block
            mask = (y >= edges[i]) & (y < edges[i+1])
            # Handle the last edge inclusively
            if i == len(edges) - 2:
                mask = (y >= edges[i]) & (y <= edges[i+1])
            
            indices = np.where(mask)[0]
            n_samples = len(indices)
            block_range = edges[i+1] - edges[i]
            
            # Calculate current density (samples per unit of y)
            current_density = n_samples / block_range if block_range > 0 else 0
            
            # New strategy: Only oversample underrepresented blocks
            if current_density < uniform_density:
                # Calculate how much to boost this block
                target_density = (
                    current_density + 
                    self.target_uniformity * (uniform_density - current_density)
                )
                target_samples = int(np.round(target_density * block_range))
            else:
                # Keep all samples in well-represented blocks
                target_samples = n_samples
            
            # Ensure we never downsample
            target_samples = max(target_samples, n_samples)
            
            block_info = BlockInfo(
                lower_edge=edges[i],
                upper_edge=edges[i+1],
                n_samples=n_samples,
                density=current_density,
                indices=indices,
                target_samples=target_samples
            )
            
            blocks_info.append(block_info)
        
        self.blocks_info_ = blocks_info
        return blocks_info
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray, 
                    find_alpha: bool = True,
                    alpha: Optional[float] = None,
                    verbose: bool = False) -> Tuple[any, any]:
        """
        Fit the resampler and return resampled data in the same format as input.
        
        This method now uses an oversampling-only strategy to preserve all original
        data diversity while addressing imbalanced distributions.
        
        Parameters:
        -----------
        X : array-like, DataFrame, AnnData, or sparse matrix
            Feature matrix in any supported format
        y : array-like, Series, or 1D array
            Target values
        find_alpha : bool, default=True
            Whether to find optimal alpha using parallel search
        alpha : float or None
            Alpha value to use if find_alpha is False
        verbose : bool, default=False
            Whether to print progress information
            
        Returns:
        --------
        X_resampled : same type as input X
            Resampled features in the same format as input
        y_resampled : same type as input y
            Resampled target values in the same format as input
        """
        
        # Extract numpy arrays for processing
        if self.preserve_metadata and self._data_handler:
            X_array, y_array = self._data_handler.extract_data(X, y)
        else:
            # Simple conversion without metadata preservation
            X_array = np.asarray(X)
            y_array = np.asarray(y).ravel()
        
        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same number of samples")
        
        # Store the resampling indices so we can reconstruct metadata
        self._resampling_indices = []
        
        # Now run the core algorithm with numpy arrays
        X_resampled, y_resampled = self._fit_resample_core(
            X_array, y_array, find_alpha, alpha, verbose
        )
        
        # Reconstruct output in the original format
        if self.preserve_metadata and self._data_handler:
            return self._data_handler.reconstruct_output(
                X_resampled, y_resampled, 
                np.array(self._resampling_indices)
            )
        else:
            return X_resampled, y_resampled
    
    def _fit_resample_core(self, X: np.ndarray, y: np.ndarray,
                          find_alpha: bool, alpha: Optional[float],
                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Core resampling logic that works with numpy arrays.
        
        This now implements the oversampling-only strategy.
        """
        # Step 1: Find optimal alpha if requested
        if find_alpha:
            if verbose:
                print("Finding optimal alpha value using parallel search...")
            self.find_optimal_alpha(y, verbose=verbose)
        elif alpha is not None:
            self.alpha_ = alpha
        
        # Step 2: Create blocks
        if verbose:
            print(f"\nCreating blocks with alpha={self.alpha_:.2e}")
        blocks_info = self._create_blocks(y, self.alpha_)
        
        if verbose:
            print(f"Created {len(blocks_info)} blocks")
            print("\nBlock statistics (oversampling-only strategy):")
            print("Block | Range | Current Samples | Target Samples | Oversample Ratio")
            print("-" * 75)
            for i, block in enumerate(blocks_info):
                ratio = block.target_samples / max(block.n_samples, 1)
                print(f"{i+1:5d} | [{block.lower_edge:6.2f}, {block.upper_edge:6.2f}] | "
                      f"{block.n_samples:15d} | {block.target_samples:14d} | {ratio:16.2f}x")
        
        # Step 3: Resample within each block (oversampling only)
        X_resampled = []
        y_resampled = []
        
        for block in blocks_info:
            if block.n_samples == 0:
                continue
                
            # Get data for this block
            block_X = X[block.indices]
            block_y = y[block.indices]
            
            if block.target_samples > block.n_samples:
                # Oversample: sample with replacement
                resample_indices = np.random.choice(
                    block.n_samples, 
                    size=block.target_samples, 
                    replace=True
                )
            else:
                # Keep all samples (no downsampling)
                resample_indices = np.arange(block.n_samples)
            
            # Track which original indices we're using
            original_indices = block.indices[resample_indices]
            self._resampling_indices.extend(original_indices)
            
            # Add resampled data
            X_resampled.append(block_X[resample_indices])
            y_resampled.append(block_y[resample_indices])
        
        # Concatenate all blocks
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        
        # Shuffle the resampled data to avoid any ordering artifacts
        shuffle_indices = np.random.permutation(len(y_resampled))
        X_resampled = X_resampled[shuffle_indices]
        y_resampled = y_resampled[shuffle_indices]
        
        # Also shuffle the resampling indices to maintain consistency
        self._resampling_indices = np.array(self._resampling_indices)[shuffle_indices]
        
        if verbose:
            print(f"\nResampling complete:")
            print(f"Original samples: {len(y)}")
            print(f"Resampled samples: {len(y_resampled)}")
            print(f"Size increase: {len(y_resampled) / len(y):.2f}x")
            
            # Show density improvement
            original_std = np.std([len(block.indices) for block in blocks_info])
            resampled_std = np.std([block.target_samples for block in blocks_info])
            print(f"Block size std deviation: {original_std:.1f} → {resampled_std:.1f}")
        
        return X_resampled, y_resampled
    
    def get_block_assignments(self, y: np.ndarray) -> np.ndarray:
        """
        Get block assignments for a given set of target values.
        
        Useful for understanding which block each sample belongs to.
        """
        if self.edges_ is None:
            raise ValueError("No blocks fitted yet. Run fit_resample first.")
        
        return np.digitize(y, self.edges_) - 1


# Convenience function for simple usage
def bayesian_blocks_oversample(X: np.ndarray, y: np.ndarray, 
                           target_uniformity: float = 0.8,
                           random_state: Optional[int] = None,
                           n_jobs: int = -1,
                           verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for resampling data using Bayesian blocks.
    
    This function uses the new oversampling-only strategy and parallelized alpha discovery.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target values
    target_uniformity : float, default=0.8
        How much to boost underrepresented regions (0=no change, 1=perfectly uniform)
    random_state : int or None, default=None
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs for alpha discovery
    verbose : bool, default=False
        Whether to print progress information
        
    Returns:
    --------
    X_resampled : array-like
        Resampled feature matrix (same or larger size)
    y_resampled : array-like
        Resampled target values (same or larger size)
        
    Example:
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10, noise=5)
    >>> # Create a long-tailed distribution
    >>> y = np.exp(y / y.std())
    >>> X_balanced, y_balanced = bayesian_blocks_oversample(X, y, verbose=True)
    """
    resampler = BayesianBlocksOversampler(
        target_uniformity=target_uniformity,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    return resampler.fit_resample(X, y, find_alpha=True, verbose=verbose)


# Example usage and testing
if __name__ == "__main__":
    # Create a synthetic long-tailed dataset for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    # Generate features
    X = np.random.randn(n_samples, 5)
    
    # Create a long-tailed target distribution
    # Most values are low, with rare high values
    y = np.random.exponential(scale=2, size=n_samples)
    # Add some rare very high values
    rare_mask = np.random.rand(n_samples) < 0.01
    y[rare_mask] *= 10
    
    print("Original distribution statistics:")
    print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"Percentiles - 50th: {np.percentile(y, 50):.2f}, "
          f"90th: {np.percentile(y, 90):.2f}, "
          f"99th: {np.percentile(y, 99):.2f}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Apply Bayesian blocks resampling with parallel alpha discovery
    print("\n" + "="*70)
    print("Applying Bayesian Blocks Resampling (Oversampling-Only Strategy)")
    print("="*70)
    
    resampler = BayesianBlocksOversampler(
        target_uniformity=0.5, 
        random_state=42,
        # n_jobs=4  # Use 4 cores for demonstration
    )
    X_train_balanced, y_train_balanced = resampler.fit_resample(
        X_train, y_train, verbose=True
    )
    
    print("\nBalanced distribution statistics:")
    print(f"Mean: {y_train_balanced.mean():.2f}, Std: {y_train_balanced.std():.2f}")
    print(f"Percentiles - 50th: {np.percentile(y_train_balanced, 50):.2f}, "
          f"90th: {np.percentile(y_train_balanced, 90):.2f}, "
          f"99th: {np.percentile(y_train_balanced, 99):.2f}")