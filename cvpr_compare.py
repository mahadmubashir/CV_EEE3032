"""
Distance Metrics for Feature Comparison
EEE3032 - Computer Vision and Pattern Recognition

This module provides various distance metrics for comparing image descriptors.
"""

import numpy as np


def _validate_descriptors(F1, F2):
    """
    Validate and flatten descriptor arrays.
    
    Args:
        F1, F2: Feature descriptors (can be 1D or 2D)
    
    Returns:
        tuple: Flattened arrays (F1_flat, F2_flat)
    """
    F1_flat = F1.flatten()
    F2_flat = F2.flatten()
    
    if len(F1_flat) != len(F2_flat):
        raise ValueError(f"Descriptor dimensions don't match: {len(F1_flat)} vs {len(F2_flat)}")
    
    return F1_flat, F2_flat


def euclidean_distance(F1, F2):
    """
    Compute the Euclidean distance (L2 norm) between two feature descriptors.
    
    Euclidean distance measures the "straight-line distance" between two points
    in feature space. Formula: sqrt(sum((F1[i] - F2[i])^2))
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors
    
    Returns:
        float: Euclidean distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    diff = F1_flat - F2_flat
    return np.sqrt(np.sum(diff ** 2))


def manhattan_distance(F1, F2):
    """
    Compute the Manhattan distance (L1 norm) between two feature descriptors.
    
    Manhattan distance is the sum of absolute differences. It's more robust to
    outliers than Euclidean distance. Formula: sum(|F1[i] - F2[i]|)
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors
    
    Returns:
        float: Manhattan distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    return np.sum(np.abs(F1_flat - F2_flat))


def chebyshev_distance(F1, F2):
    """
    Compute the Chebyshev distance (L∞ norm) between two feature descriptors.
    
    Chebyshev distance is the maximum of the absolute differences along any
    coordinate dimension. It measures the maximum change in any single feature.
    Formula: max(|F1[i] - F2[i]|)
    
    Also known as the chessboard distance or maximum metric.
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors
    
    Returns:
        float: Chebyshev distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    return np.max(np.abs(F1_flat - F2_flat))


def minkowski_distance(F1, F2, p=3):
    """
    Compute the Minkowski distance between two feature descriptors.
    
    Minkowski distance is a generalization of Euclidean (p=2) and Manhattan (p=1).
    Formula: (sum(|F1[i] - F2[i]|^p))^(1/p)
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors
        p: int or float - Minkowski order parameter
            - p=1: Manhattan distance
            - p=2: Euclidean distance
            - p=inf: Chebyshev distance
    
    Returns:
        float: Minkowski distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    diff = np.abs(F1_flat - F2_flat)
    
    if p == np.inf:
        return chebyshev_distance(F1, F2)
    elif p <= 0:
        raise ValueError(f"Minkowski parameter p must be positive, got {p}")
    
    return np.power(np.sum(np.power(diff, p)), 1.0 / p)


def mahalanobis_distance(F1, F2, covariance_matrix=None, all_features=None):
    """
    Compute the Mahalanobis distance between two feature descriptors.
    
    Mahalanobis distance accounts for correlations between features and scale
    differences. It's useful when features have different variances or are correlated.
    Formula: sqrt((F1 - F2)^T * C^(-1) * (F1 - F2))
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors
        covariance_matrix: numpy.ndarray (optional) - Precomputed covariance matrix
            If None, will compute from all_features
        all_features: numpy.ndarray (optional) - All feature vectors for covariance
            computation. Required if covariance_matrix is None.
    
    Returns:
        float: Mahalanobis distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    diff = F1_flat - F2_flat
    
    # Compute covariance matrix if not provided
    if covariance_matrix is None:
        if all_features is None:
            raise ValueError("Either covariance_matrix or all_features must be provided")
        
        # Flatten all features if needed
        features_flat = np.array([f.flatten() for f in all_features])
        covariance_matrix = np.cov(features_flat.T)
    
    # Ensure covariance matrix is positive definite (add small epsilon for numerical stability)
    covariance_matrix = covariance_matrix + np.eye(len(covariance_matrix)) * 1e-6
    
    try:
        # Compute inverse covariance matrix
        inv_cov = np.linalg.inv(covariance_matrix)
        
        # Compute Mahalanobis distance: sqrt(diff^T * inv_cov * diff)
        mahal_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
        return mahal_dist
    except np.linalg.LinAlgError:
        # If matrix inversion fails, fall back to Euclidean distance
        print("Warning: Covariance matrix inversion failed, using Euclidean distance")
        return euclidean_distance(F1, F2)


def bhattacharyya_distance(F1, F2):
    """
    Compute the Bhattacharyya distance between two feature descriptors.
    
    Bhattacharyya distance measures the similarity between two probability distributions.
    It's particularly useful for comparing histograms. Formula: -ln(sum(sqrt(F1[i] * F2[i])))
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors (should be normalized histograms)
    
    Returns:
        float: Bhattacharyya distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    
    # Compute Bhattacharyya coefficient: sum(sqrt(F1[i] * F2[i]))
    # Add small epsilon to avoid issues with zero values
    epsilon = 1e-10
    bc = np.sum(np.sqrt(np.maximum(F1_flat * F2_flat, epsilon)))
    
    # Compute Bhattacharyya distance: -ln(BC)
    # Clamp BC to avoid log(0) or log(negative)
    bc = np.clip(bc, epsilon, 1.0)
    bhattacharyya_dist = -np.log(bc)
    
    return bhattacharyya_dist


def chi_square_distance(F1, F2):
    """
    Compute the Chi-square distance between two feature descriptors.
    
    Chi-square distance is commonly used for comparing histograms. It measures the
    difference between two distributions. Formula: sum((F1[i] - F2[i])^2 / (F1[i] + F2[i]))
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors (typically normalized histograms)
    
    Returns:
        float: Chi-square distance (lower = more similar)
    """
    F1_flat, F2_flat = _validate_descriptors(F1, F2)
    
    # Compute chi-square distance: sum((F1[i] - F2[i])^2 / (F1[i] + F2[i] + epsilon))
    # Add small epsilon to denominator to avoid division by zero
    epsilon = 1e-10
    diff = F1_flat - F2_flat
    denominator = F1_flat + F2_flat + epsilon
    chi_square_dist = np.sum((diff ** 2) / denominator)
    
    return chi_square_dist


def cvpr_compare(F1, F2):
    """
    Default comparison function using Euclidean distance.
    
    This function is kept for backward compatibility with existing code.
    It computes the Euclidean distance between two descriptors.
    
    Args:
        F1, F2: numpy.ndarray - Feature descriptors
    
    Returns:
        float: Euclidean distance between the two descriptors
    """
    return euclidean_distance(F1, F2)


