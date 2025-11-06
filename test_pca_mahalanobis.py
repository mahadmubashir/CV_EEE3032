"""
Test script for PCA and Mahalanobis distance implementation
This script tests different descriptor and distance metric combinations
"""

import os
import numpy as np
import sys
sys.path.insert(0, '.')

from cvpr_visualsearch import (
    load_descriptors, 
    compute_distances, 
    load_pca_model_for_descriptor,
    get_class_id_from_path,
    print_ranked_results
)
from cvpr_compare import (
    euclidean_distance,
    mahalanobis_distance,
    manhattan_distance
)

# Configuration
DESCRIPTOR_FOLDER = './descriptors'
DATASET_IMAGES_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
TOP_K = 10
QUERY_IDX = 50

def test_descriptor_distance_combination(descriptor_subfolder, distance_func, distance_name, pca_model=None):
    """
    Test a specific descriptor and distance metric combination.
    
    Args:
        descriptor_subfolder: Subfolder name for descriptors
        distance_func: Distance function to use
        distance_name: Name of distance metric
        pca_model: PCA model if using PCA descriptors
    
    Returns:
        dict: Test results including top K results
    """
    print("\n" + "="*70)
    print(f"TEST: {descriptor_subfolder} + {distance_name.upper()}")
    print("="*70)
    
    # Load descriptors
    try:
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_subfolder)
        print(f"✓ Loaded {len(all_features)} descriptors")
        print(f"✓ Descriptor dimension: {all_features.shape[1]}")
    except Exception as e:
        print(f"✗ Failed to load descriptors: {e}")
        return None
    
    # Load PCA model if needed
    if pca_model is None and '_PCA' in descriptor_subfolder:
        pca_model = load_pca_model_for_descriptor(descriptor_subfolder)
        if pca_model:
            n_comp = all_features.shape[1]
            var_explained = pca_model['cumulative_variance_ratio'][n_comp - 1] * 100
            print(f"✓ PCA model loaded: {n_comp} components ({var_explained:.2f}% variance)")
    
    # Select query
    query_feature = all_features[QUERY_IDX]
    query_path = image_paths[QUERY_IDX]
    query_class = get_class_id_from_path(query_path)
    
    print(f"\nQuery image: {os.path.basename(query_path)}")
    print(f"Query class ID: {query_class}")
    
    # Compute distances
    print(f"\nComputing distances using {distance_name}...")
    distances = compute_distances(query_feature, all_features, distance_func, 
                                 distance_name, pca_model=pca_model)
    
    # Exclude query from results
    ranked_results = [(d, i) for (d, i) in distances if i != QUERY_IDX]
    
    # Analyze top K results
    print(f"\nTop {TOP_K} Results:")
    print("-" * 70)
    correct_matches = 0
    for rank, (distance, idx) in enumerate(ranked_results[:TOP_K], start=1):
        result_class = get_class_id_from_path(image_paths[idx])
        match = "✓" if result_class == query_class else "✗"
        if result_class == query_class:
            correct_matches += 1
        print(f"{rank:2d}. {match} Class: {result_class:2d}, Distance: {distance:.4f}, File: {os.path.basename(image_paths[idx])}")
    
    precision_at_k = correct_matches / TOP_K
    print("-" * 70)
    print(f"Precision@{TOP_K}: {precision_at_k:.2%} ({correct_matches}/{TOP_K} correct)")
    
    return {
        'descriptor': descriptor_subfolder,
        'distance': distance_name,
        'ranked_results': ranked_results[:TOP_K],
        'precision_at_k': precision_at_k,
        'query_class': query_class
    }


def main():
    """Run comprehensive tests."""
    print("="*70)
    print("PCA AND MAHALANOBIS DISTANCE TESTING")
    print("="*70)
    
    results = []
    
    # Test 1: Global RGB Histogram (original) + Euclidean
    print("\n[TEST 1] Baseline: Global RGB Histogram + Euclidean Distance")
    result1 = test_descriptor_distance_combination(
        'globalRGBhisto', 
        euclidean_distance, 
        'euclidean'
    )
    if result1:
        results.append(result1)
    
    # Test 2: Global RGB Histogram (PCA 10) + Euclidean
    print("\n[TEST 2] PCA Reduction: Global RGB Histogram (PCA 10) + Euclidean Distance")
    result2 = test_descriptor_distance_combination(
        'globalRGBhisto_PCA10', 
        euclidean_distance, 
        'euclidean'
    )
    if result2:
        results.append(result2)
    
    # Test 3: Global RGB Histogram (PCA 10) + Mahalanobis
    print("\n[TEST 3] PCA + Mahalanobis: Global RGB Histogram (PCA 10) + Mahalanobis Distance")
    result3 = test_descriptor_distance_combination(
        'globalRGBhisto_PCA10', 
        mahalanobis_distance, 
        'mahalanobis'
    )
    if result3:
        results.append(result3)
    
    # Test 4: Global RGB Histogram (original) + Mahalanobis
    print("\n[TEST 4] Original + Mahalanobis: Global RGB Histogram + Mahalanobis Distance")
    result4 = test_descriptor_distance_combination(
        'globalRGBhisto', 
        mahalanobis_distance, 
        'mahalanobis'
    )
    if result4:
        results.append(result4)
    
    # Test 5: Spatial Grid (PCA 20) + Euclidean
    print("\n[TEST 5] Spatial Grid PCA: Spatial Grid RGB Histogram (PCA 20) + Euclidean Distance")
    result5 = test_descriptor_distance_combination(
        'spatialGridRGBhisto_PCA20', 
        euclidean_distance, 
        'euclidean'
    )
    if result5:
        results.append(result5)
    
    # Test 6: Spatial Grid (PCA 20) + Mahalanobis
    print("\n[TEST 6] Spatial Grid PCA + Mahalanobis: Spatial Grid RGB Histogram (PCA 20) + Mahalanobis Distance")
    result6 = test_descriptor_distance_combination(
        'spatialGridRGBhisto_PCA20', 
        mahalanobis_distance, 
        'mahalanobis'
    )
    if result6:
        results.append(result6)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL TESTS")
    print("="*70)
    print(f"{'Test':<40} {'Precision@10':<15}")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        desc = result['descriptor']
        dist = result['distance']
        prec = result['precision_at_k']
        print(f"{desc:<30} {dist:<10} {prec:>6.2%}")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)
    
    # Analysis
    print("\nANALYSIS:")
    print("-" * 70)
    if len(results) >= 2:
        baseline = results[0]['precision_at_k']
        print(f"Baseline (Global RGB + Euclidean): {baseline:.2%}")
        
        if len(results) >= 2:
            pca_euclidean = results[1]['precision_at_k']
            improvement = pca_euclidean - baseline
            print(f"PCA 10 + Euclidean: {pca_euclidean:.2%} ({improvement:+.2%})")
        
        if len(results) >= 3:
            pca_mahalanobis = results[2]['precision_at_k']
            improvement = pca_mahalanobis - baseline
            print(f"PCA 10 + Mahalanobis: {pca_mahalanobis:.2%} ({improvement:+.2%})")
        
        if len(results) >= 4:
            orig_mahalanobis = results[3]['precision_at_k']
            improvement = orig_mahalanobis - baseline
            print(f"Original + Mahalanobis: {orig_mahalanobis:.2%} ({improvement:+.2%})")


if __name__ == "__main__":
    main()

