"""
Evaluation Framework for Visual Search System
EEE3032 - Computer Vision and Pattern Recognition

This module provides functions to evaluate the performance of different
descriptors and distance metrics using precision-recall statistics.
"""

import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# Dataset Analysis Functions
# ============================================================================

def get_class_id_from_filename(filename):
    """
    Extract the integer class ID from an image filename.
    
    Args:
        filename: Image filename (e.g., '12_33_s.bmp')
    
    Returns:
        int: Class ID (e.g., 12) or -1 if parsing fails
    """
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return -1


def analyze_dataset(dataset_folder):
    """
    Analyze the dataset to extract all classes and count images per class.
    
    Args:
        dataset_folder: Path to the dataset Images folder
    
    Returns:
        dict: Dictionary with:
            - 'classes': sorted list of all class IDs
            - 'class_counts': dict mapping class_id -> count
            - 'total_images': total number of images
            - 'image_to_class': dict mapping image filename -> class_id
    """
    print("="*60)
    print("ANALYZING DATASET")
    print("="*60)
    
    # Get all image files
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.bmp')]
    total_images = len(image_files)
    
    # Extract classes and count images per class
    class_counts = defaultdict(int)
    image_to_class = {}
    
    for filename in image_files:
        class_id = get_class_id_from_filename(filename)
        if class_id != -1:
            class_counts[class_id] += 1
            image_to_class[filename] = class_id
    
    # Get sorted list of classes
    classes = sorted(class_counts.keys())
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {total_images}")
    print(f"   Number of classes: {len(classes)}")
    print(f"\n📋 Images per class:")
    for class_id in classes:
        print(f"   Class {class_id:2d}: {class_counts[class_id]:3d} images")
    
    print("="*60)
    
    return {
        'classes': classes,
        'class_counts': dict(class_counts),
        'total_images': total_images,
        'image_to_class': image_to_class
    }


# ============================================================================
# Precision-Recall Computation Functions
# ============================================================================

def compute_precision_recall(ranked_results, image_paths, query_class, k=None):
    """
    Compute precision and recall for a single query at rank k.
    
    Precision@k = (number of relevant items in top k) / k
    Recall@k = (number of relevant items in top k) / (total relevant items)
    
    Args:
        ranked_results: List of (distance, index) tuples, sorted by distance
        image_paths: List of image file paths corresponding to indices
        query_class: Class ID of the query image
        k: Rank at which to compute precision/recall (if None, uses all results)
    
    Returns:
        tuple: (precision@k, recall@k, num_relevant_in_top_k, total_relevant)
    """
    if k is None:
        k = len(ranked_results)
    else:
        k = min(k, len(ranked_results))
    
    # Count relevant items in top k
    num_relevant_in_top_k = 0
    for i in range(k):
        _, idx = ranked_results[i]
        result_path = image_paths[idx]
        result_class = get_class_id_from_filename(os.path.basename(result_path))
        
        if result_class == query_class:
            num_relevant_in_top_k += 1
    
    # Count total relevant items in dataset (excluding query itself)
    total_relevant = 0
    for path in image_paths:
        result_class = get_class_id_from_filename(os.path.basename(path))
        if result_class == query_class:
            total_relevant += 1
    
    # Exclude query from total relevant count
    total_relevant = max(0, total_relevant - 1)  # Subtract 1 for query itself
    
    # Compute precision and recall
    if k > 0:
        precision = num_relevant_in_top_k / k
    else:
        precision = 0.0
    
    if total_relevant > 0:
        recall = num_relevant_in_top_k / total_relevant
    else:
        recall = 0.0
    
    return precision, recall, num_relevant_in_top_k, total_relevant


def compute_precision_recall_curve(ranked_results, image_paths, query_class, max_k=None):
    """
    Compute precision and recall at each rank k to generate a PR curve.
    
    Args:
        ranked_results: List of (distance, index) tuples, sorted by distance
        image_paths: List of image file paths corresponding to indices
        query_class: Class ID of the query image
        max_k: Maximum rank to evaluate (if None, uses all results)
    
    Returns:
        dict: Dictionary with:
            - 'precisions': list of precision values at each k
            - 'recalls': list of recall values at each k
            - 'k_values': list of k values
    """
    if max_k is None:
        max_k = len(ranked_results)
    else:
        max_k = min(max_k, len(ranked_results))
    
    precisions = []
    recalls = []
    k_values = []
    
    # Count total relevant items (excluding query)
    total_relevant = 0
    for path in image_paths:
        result_class = get_class_id_from_filename(os.path.basename(path))
        if result_class == query_class:
            total_relevant += 1
    total_relevant = max(0, total_relevant - 1)  # Subtract 1 for query itself
    
    # Compute precision and recall at each rank k
    num_relevant_so_far = 0
    for k in range(1, max_k + 1):
        _, idx = ranked_results[k - 1]
        result_path = image_paths[idx]
        result_class = get_class_id_from_filename(os.path.basename(result_path))
        
        if result_class == query_class:
            num_relevant_so_far += 1
        
        precision = num_relevant_so_far / k
        recall = num_relevant_so_far / total_relevant if total_relevant > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        k_values.append(k)
    
    return {
        'precisions': precisions,
        'recalls': recalls,
        'k_values': k_values
    }


def compute_average_precision(ranked_results, image_paths, query_class, max_k=None):
    """
    Compute Average Precision (AP) for a single query.
    
    AP = (1/R) * sum(P(k) * rel(k)) where:
    - R = total number of relevant items
    - P(k) = precision at rank k
    - rel(k) = 1 if item at rank k is relevant, 0 otherwise
    
    Args:
        ranked_results: List of (distance, index) tuples, sorted by distance
        image_paths: List of image file paths corresponding to indices
        query_class: Class ID of the query image
        max_k: Maximum rank to evaluate (if None, uses all results)
    
    Returns:
        float: Average Precision (AP) score
    """
    if max_k is None:
        max_k = len(ranked_results)
    else:
        max_k = min(max_k, len(ranked_results))
    
    # Count total relevant items (excluding query)
    total_relevant = 0
    for path in image_paths:
        result_class = get_class_id_from_filename(os.path.basename(path))
        if result_class == query_class:
            total_relevant += 1
    total_relevant = max(0, total_relevant - 1)  # Subtract 1 for query itself
    
    if total_relevant == 0:
        return 0.0
    
    # Compute AP using 11-point interpolated precision
    num_relevant_so_far = 0
    precision_sum = 0.0
    
    for k in range(1, max_k + 1):
        _, idx = ranked_results[k - 1]
        result_path = image_paths[idx]
        result_class = get_class_id_from_filename(os.path.basename(result_path))
        
        if result_class == query_class:
            num_relevant_so_far += 1
            precision_at_k = num_relevant_so_far / k
            precision_sum += precision_at_k
    
    ap = precision_sum / total_relevant
    return ap


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_all_queries(all_features, image_paths, distance_func, distance_name='euclidean',
                        pca_model=None, top_k=10, max_queries=None):
    """
    Evaluate all queries (or a subset) in the dataset.
    Computes precision, recall, average precision, and PR curves for all queries.
    
    Note: PR curves and Average Precision are computed using ALL ranked results,
    not limited to top_k. Only precision@top_k and recall@top_k are computed
    at the specific rank for reporting purposes.
    
    Args:
        all_features: Array of all feature descriptors
        image_paths: List of image file paths
        distance_func: Distance function to use
        distance_name: Name of distance metric
        pca_model: PCA model if using PCA descriptors
        top_k: Number of top results for precision@k reporting (PR curves use all results)
        max_queries: Maximum number of queries to evaluate (None = all)
    
    Returns:
        dict: Aggregated evaluation results containing:
            - 'mean_precision': Mean precision@top_k
            - 'mean_recall': Mean recall@top_k
            - 'mean_ap': Mean Average Precision (mAP) computed from all results
            - 'all_precisions': List of precision@top_k for each query
            - 'all_recalls': List of recall@top_k for each query
            - 'all_aps': List of AP for each query (computed from all results)
            - 'all_pr_curves': List of PR curve data for each query (computed from all results)
            - 'num_queries': Number of queries evaluated
    """
    from cvpr_visualsearch import compute_distances
    
    num_images = len(image_paths)
    if max_queries is None:
        max_queries = num_images
    
    max_queries = min(max_queries, num_images)
    
    print(f"\n{'='*60}")
    print(f"EVALUATING {max_queries} QUERIES")
    print(f"{'='*60}")
    
    all_precisions = []
    all_recalls = []
    all_aps = []
    all_pr_curves = []
    
    for query_idx in range(max_queries):
        if (query_idx + 1) % 50 == 0:
            print(f"  Processing query {query_idx + 1}/{max_queries}...")
        
        # Get query information
        query_feature = all_features[query_idx]
        query_path = image_paths[query_idx]
        query_class = get_class_id_from_filename(os.path.basename(query_path))
        
        # Compute distances
        distances = compute_distances(query_feature, all_features, distance_func, 
                                     distance_name, pca_model=pca_model)
        
        # Exclude query from results
        ranked_results = [(d, i) for (d, i) in distances if i != query_idx]
        
        # Compute precision and recall at top_k (for reporting purposes)
        precision_k, recall_k, _, _ = compute_precision_recall(
            ranked_results, image_paths, query_class, k=top_k
        )
        
        # Compute Average Precision using ALL results (not limited to top_k)
        ap = compute_average_precision(ranked_results, image_paths, query_class, max_k=None)
        
        # Compute PR curve using ALL results (not limited to top_k)
        pr_curve = compute_precision_recall_curve(ranked_results, image_paths, query_class, max_k=None)
        
        all_precisions.append(precision_k)
        all_recalls.append(recall_k)
        all_aps.append(ap)
        all_pr_curves.append(pr_curve)
    
    # Compute mean metrics
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_ap = np.mean(all_aps)
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Mean Precision@{top_k}: {mean_precision:.4f}")
    print(f"   Mean Recall@{top_k}: {mean_recall:.4f}")
    print(f"   Mean Average Precision (mAP): {mean_ap:.4f}")
    
    return {
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_ap': mean_ap,
        'all_precisions': all_precisions,
        'all_recalls': all_recalls,
        'all_aps': all_aps,
        'all_pr_curves': all_pr_curves,
        'num_queries': max_queries
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_pr_curve(pr_curve_data, title="Precision-Recall Curve", save_path=None):
    """
    Plot a precision-recall curve.
    
    Args:
        pr_curve_data: Dictionary with 'precisions' and 'recalls' lists
        title: Plot title
        save_path: Path to save the plot (None = display only)
    """
    precisions = pr_curve_data['precisions']
    recalls = pr_curve_data['recalls']
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_average_pr_curve(all_pr_curves, title="Average Precision-Recall Curve", save_path=None):
    """
    Plot average precision-recall curve across multiple queries.
    
    Args:
        all_pr_curves: List of PR curve dictionaries
        title: Plot title
        save_path: Path to save the plot (None = display only)
    """
    # Find maximum recall value to determine interpolation points
    max_recall = 0
    for pr_curve in all_pr_curves:
        if pr_curve['recalls']:
            max_recall = max(max_recall, max(pr_curve['recalls']))
    
    # Interpolate precision at standard recall levels (0.0 to 1.0 in steps of 0.1)
    recall_levels = np.arange(0, 1.1, 0.1)
    interpolated_precisions = []
    
    for recall_level in recall_levels:
        precisions_at_level = []
        for pr_curve in all_pr_curves:
            recalls = np.array(pr_curve['recalls'])
            precisions = np.array(pr_curve['precisions'])
            
            # Find maximum precision at this recall level or higher
            if len(recalls) > 0:
                mask = recalls >= recall_level
                if np.any(mask):
                    max_precision = np.max(precisions[mask])
                    precisions_at_level.append(max_precision)
        
        if precisions_at_level:
            interpolated_precisions.append(np.mean(precisions_at_level))
        else:
            interpolated_precisions.append(0.0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_levels, interpolated_precisions, 'b-', linewidth=2, marker='o', label='Average PR Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Average PR curve saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_descriptor_distance_combination(descriptor_subfolder, distance_func, distance_name,
                                           descriptor_folder='./descriptors',
                                           dataset_folder='./MSRC_ObjCategImageDatabase_v2/Images',
                                           pca_model=None, top_k=10, max_queries=None):
    """
    Complete evaluation pipeline for a descriptor-distance combination.
    
    Note: PR curves and Average Precision are computed using ALL ranked results.
    The top_k parameter is only used for precision@k and recall@k reporting.
    
    Args:
        descriptor_subfolder: Subfolder name for descriptors
        distance_func: Distance function to use
        distance_name: Name of distance metric
        descriptor_folder: Base folder containing descriptors
        dataset_folder: Folder containing dataset images
        pca_model: PCA model if using PCA descriptors
        top_k: Number of top results for precision@k/recall@k reporting (PR curves use all results)
        max_queries: Maximum number of queries to evaluate (None = all)
    
    Returns:
        dict: Complete evaluation results
    """
    from cvpr_visualsearch import load_descriptors
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {descriptor_subfolder} + {distance_name.upper()}")
    print(f"{'='*60}")
    
    # Load descriptors
    print(f"\nLoading descriptors from: {descriptor_subfolder}")
    all_features, image_paths = load_descriptors(descriptor_folder, descriptor_subfolder)
    print(f"Loaded {len(all_features)} descriptors")
    
    # Analyze dataset
    dataset_info = analyze_dataset(dataset_folder)
    
    # Evaluate all queries
    evaluation_results = evaluate_all_queries(
        all_features, image_paths, distance_func, distance_name,
        pca_model, top_k, max_queries
    )
    
    return {
        'descriptor': descriptor_subfolder,
        'distance': distance_name,
        'dataset_info': dataset_info,
        'evaluation': evaluation_results
    }

