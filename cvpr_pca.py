"""
PCA (Principal Component Analysis) Utilities for Feature Descriptors
EEE3032 - Computer Vision and Pattern Recognition

This module provides functions for dimensionality reduction using PCA
and projection of feature descriptors into lower-dimensional spaces.
"""

import numpy as np
import scipy.io as sio
import os


def compute_pca(all_features, normalize=True):
    """
    Compute PCA (Principal Component Analysis) on feature descriptors.
    
    PCA finds the directions of maximum variance in the data and projects
    features onto these principal components. This reduces dimensionality
    while preserving most of the information.
    
    Args:
        all_features: numpy.ndarray of shape (N, D)
            All feature descriptors where N is number of samples and D is feature dimension
        normalize: bool
            Whether to normalize features (zero mean, unit variance) before PCA
    
    Returns:
        dict: PCA model containing:
            - 'components': Principal components (eigenvectors) shape (D, D)
            - 'mean': Mean of original features shape (D,)
            - 'std': Standard deviation of original features shape (D,) if normalize=True
            - 'explained_variance': Variance explained by each component shape (D,)
            - 'explained_variance_ratio': Percentage of variance explained shape (D,)
            - 'cumulative_variance_ratio': Cumulative variance explained shape (D,)
    """
    # Ensure features are 2D array
    if len(all_features.shape) == 1:
        all_features = all_features.reshape(1, -1)
    
    N, D = all_features.shape
    
    # Compute mean of features
    mean = np.mean(all_features, axis=0)
    
    # Center the data (subtract mean)
    centered_features = all_features - mean
    
    # Normalize if requested (zero mean, unit variance)
    std = None
    if normalize:
        std = np.std(centered_features, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        centered_features = centered_features / std
    
    # Compute covariance matrix: C = (1/(N-1)) * X^T * X
    # Note: centered_features is already centered, so we can use dot product
    covariance_matrix = np.dot(centered_features.T, centered_features) / (N - 1)
    
    # Eigenvalue decomposition: C = V * Λ * V^T
    # eigenvalues: Λ (variance along each principal component)
    # eigenvectors: V (principal components - directions of maximum variance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Compute explained variance ratios
    explained_variance = eigenvalues
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    return {
        'components': eigenvectors,  # Principal components (columns are PCs)
        'mean': mean,
        'std': std,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'normalize': normalize
    }


def project_descriptors(features, pca_model, n_components=None, variance_threshold=None):
    """
    Project feature descriptors onto principal components to reduce dimensionality.
    
    Args:
        features: numpy.ndarray of shape (N, D) or (D,)
            Feature descriptors to project
        pca_model: dict
            PCA model returned by compute_pca()
        n_components: int, optional
            Number of principal components to use (default: use all)
        variance_threshold: float, optional
            Keep components that explain at least this fraction of variance (0-1)
            e.g., 0.95 means keep components explaining 95% of variance
    
    Returns:
        numpy.ndarray: Projected features of shape (N, d) where d <= D
        int: Number of components used
    """
    # Ensure features are 2D
    original_shape = features.shape
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    N, D = features.shape
    
    # Center features using PCA mean
    centered_features = features - pca_model['mean']
    
    # Normalize if PCA was normalized
    if pca_model['normalize'] and pca_model['std'] is not None:
        centered_features = centered_features / pca_model['std']
    
    # Determine number of components to use
    if variance_threshold is not None:
        # Find number of components needed to explain variance_threshold
        cumulative_var = pca_model['cumulative_variance_ratio']
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1
        if n_components == 0:
            n_components = D  # Use all if threshold not met
    elif n_components is None:
        n_components = D  # Use all components
    
    # Ensure n_components doesn't exceed available components
    n_components = min(n_components, D)
    
    # Project onto principal components: X_reduced = X_centered * V[:, :n_components]
    components = pca_model['components'][:, :n_components]
    projected_features = np.dot(centered_features, components)
    
    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        projected_features = projected_features.flatten()
    
    return projected_features, n_components


def get_components_for_variance(pca_model, variance_threshold=0.95):
    """
    Determine how many PCA components are needed to explain a given variance threshold.
    
    Args:
        pca_model: dict
            PCA model returned by compute_pca()
        variance_threshold: float
            Desired variance explained (0-1), e.g., 0.95 for 95%
    
    Returns:
        int: Number of components needed
    """
    cumulative_var = pca_model['cumulative_variance_ratio']
    n_components = np.argmax(cumulative_var >= variance_threshold) + 1
    if n_components == 0:
        n_components = len(cumulative_var)  # Use all if threshold not met
    return n_components


def save_pca_model(pca_model, filepath):
    """
    Save PCA model to a .mat file for later use.
    
    Args:
        pca_model: dict
            PCA model returned by compute_pca()
        filepath: str
            Path to save the PCA model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save to .mat file
    sio.savemat(filepath, {
        'components': pca_model['components'],
        'mean': pca_model['mean'],
        'std': pca_model['std'] if pca_model['std'] is not None else np.array([]),
        'explained_variance': pca_model['explained_variance'],
        'explained_variance_ratio': pca_model['explained_variance_ratio'],
        'cumulative_variance_ratio': pca_model['cumulative_variance_ratio'],
        'normalize': np.array([1 if pca_model['normalize'] else 0])
    })
    print(f"PCA model saved to: {filepath}")


def load_pca_model(filepath):
    """
    Load PCA model from a .mat file.
    
    Args:
        filepath: str
            Path to the saved PCA model
    
    Returns:
        dict: PCA model compatible with project_descriptors()
    """
    data = sio.loadmat(filepath)
    
    # Handle empty std array
    std = data['std']
    if std.size == 0:
        std = None
    else:
        std = std.flatten()
    
    normalize = bool(data['normalize'].item()) if 'normalize' in data else True
    
    return {
        'components': data['components'],
        'mean': data['mean'].flatten(),
        'std': std,
        'explained_variance': data['explained_variance'].flatten(),
        'explained_variance_ratio': data['explained_variance_ratio'].flatten(),
        'cumulative_variance_ratio': data['cumulative_variance_ratio'].flatten(),
        'normalize': normalize
    }


def plot_pca_statistics(pca_model, save_path=None):
    """
    Plot PCA statistics: explained variance and cumulative variance.
    
    Args:
        pca_model: dict
            PCA model returned by compute_pca()
        save_path: str, optional
            Path to save the plot (default: display only)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Explained variance ratio
        n_components_show = min(50, len(pca_model['explained_variance_ratio']))
        ax1.bar(range(1, n_components_show + 1), 
                pca_model['explained_variance_ratio'][:n_components_show])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Variance Explained by Each Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative variance
        ax2.plot(range(1, len(pca_model['cumulative_variance_ratio']) + 1),
                 pca_model['cumulative_variance_ratio'] * 100)
        ax2.axhline(y=95, color='r', linestyle='--', label='95% threshold')
        ax2.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"PCA statistics plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")


def print_pca_summary(pca_model, n_components_list=[10, 20, 50, 100]):
    """
    Print summary statistics of PCA model.
    
    Args:
        pca_model: dict
            PCA model returned by compute_pca()
        n_components_list: list
            List of component counts to display variance for
    """
    D = len(pca_model['explained_variance'])
    cumulative_var = pca_model['cumulative_variance_ratio']
    
    print("\n" + "="*60)
    print("PCA SUMMARY")
    print("="*60)
    print(f"Original feature dimension: {D}")
    print(f"Total variance explained by all components: {cumulative_var[-1]*100:.2f}%")
    print("\nVariance explained by different numbers of components:")
    
    for n in n_components_list:
        if n <= D:
            var_explained = cumulative_var[n-1] * 100
            print(f"  {n:3d} components: {var_explained:6.2f}%")
        else:
            print(f"  {n:3d} components: N/A (exceeds dimension {D})")
    
    # Find common thresholds
    thresholds = [0.90, 0.95, 0.99]
    print("\nComponents needed for variance thresholds:")
    for threshold in thresholds:
        n_comp = get_components_for_variance(pca_model, threshold)
        print(f"  {threshold*100:.0f}% variance: {n_comp} components")
    
    print("="*60)

