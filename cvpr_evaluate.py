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
import scipy.io as sio
from cvpr_compare import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
    mahalanobis_distance
)


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

def get_distance_list(query_feature, all_features, image_paths, distance_function, distance_name='euclidean'):
    """
    calculate the distance between the query and the database images using the distance function from cvpr_compare.py
    Args:
        query_feature: query feature
        all_features: list of all features
        image_paths: list of image paths corresponding to all_features
        distance_function: distance function from cvpr_compare.py
        distance_name: name of the distance metric (for Mahalanobis distance handling)
    Returns:
        distance_list: sorted list of (distance, image_path) tuples
    """
    distances = []
    
    # For Mahalanobis distance, compute covariance matrix once
    if distance_name == 'mahalanobis':
        # Compute covariance matrix from all features
        features_flat = np.array([f.flatten() for f in all_features])
        covariance_matrix = np.cov(features_flat.T)
        # Add small epsilon for numerical stability
        covariance_matrix = covariance_matrix + np.eye(len(covariance_matrix)) * 1e-6
        
        for i, feature in enumerate(all_features):
            dist = distance_function(query_feature, feature, covariance_matrix=covariance_matrix)
            distances.append((dist, image_paths[i]))
    else:
        # For other distance metrics
        for i, feature in enumerate(all_features):
            dist = distance_function(query_feature, feature)
            distances.append((dist, image_paths[i]))
    
    # Sort by distance (ascending - smaller distance = more similar)
    distances.sort(key=lambda x: x[0])
    return distances


def compute_precision_recall(query_path, distance_list, dataset_info, top_k=10, verbose=False, query_idx=None):
    """
    compute the precision and recall for each query and return the precision and recall for the first 10 results
    precision is the number of relevant images in the top 10 results divided by the number of images in the top 10 results
    recall is the number of relevant images in the top 10 results divided by the number of relevant images in the database
    the relevant images are the images that have the same class as the query image
    the number of relevant images in the database is the number of images in the database that have the same class as the query image
    use the analyze_dataset function to get the number of relevant images in the database
    use the get_class_id_from_filename function to get the class id of the query image and the images in the distance list
    Args:
        query_path: path to query image
        distance_list: sorted list of (distance, image_path) tuples
        dataset_info: dictionary from analyze_dataset function
        top_k: number of top results to evaluate
        verbose: if True, print detailed information
        query_idx: query index for verbose output
    Returns:
        precision_at_k: precision for the top k results (list of precision at each rank 1..k)
        recall_at_k: recall for the top_k results (list of recall at each rank 1..k)
        average_precision: average precision (AP) for this query
    """
    # Get query class
    query_filename = os.path.basename(query_path)
    query_class = get_class_id_from_filename(query_filename)
    
    # Get total relevant images in database (excluding query itself)
    class_counts = dataset_info['class_counts']
    total_relevant = class_counts.get(query_class, 0) - 1  # Subtract 1 for query itself
    total_relevant = max(0, total_relevant)
    
    if verbose and query_idx is not None:
        print(f"\n  Query {query_idx}: {query_filename}")
        print(f"    Query class: {query_class}")
        print(f"    Total relevant images in database (excluding query): {total_relevant}")
        print(f"    Distance list length: {len(distance_list)}")
        print(f"    Top {top_k} results being evaluated")
    
    if total_relevant == 0:
        if verbose:
            print(f"    Warning: No relevant images found for class {query_class}")
        return [0.0] * top_k, [0.0] * top_k, 0.0
    
    # Get top k results
    top_k_results = distance_list[:top_k]
    
    # Compute precision and recall at each rank k
    precisions = []
    recalls = []
    num_relevant_so_far = 0
    precision_sum = 0.0
    
    if verbose:
        print(f"    Computing precision and recall at each rank k=1..{top_k}:")
    
    for k in range(1, top_k + 1):
        # Check if k-th result is relevant
        _, result_path = top_k_results[k - 1]
        result_filename = os.path.basename(result_path)
        result_class = get_class_id_from_filename(result_filename)
        
        is_relevant = (result_class == query_class)
        if is_relevant:
            num_relevant_so_far += 1
        
        precision_at_k = num_relevant_so_far / k
        recall_at_k = num_relevant_so_far / total_relevant
        
        if is_relevant:
            precision_sum += precision_at_k
        
        precisions.append(precision_at_k)
        recalls.append(recall_at_k)
        
        if verbose and k <= 5:  # Print first 5 ranks
            print(f"      Rank {k}: class={result_class}, relevant={is_relevant}, "
                  f"precision@{k}={precision_at_k:.4f}, recall@{k}={recall_at_k:.4f}, "
                  f"relevant_so_far={num_relevant_so_far}/{k}")
    
    if verbose:
        print(f"    Precision@{top_k}: {precisions[-1]:.4f}")
        print(f"    Recall@{top_k}: {recalls[-1]:.4f}")
        print(f"    Precision sum (for AP): {precision_sum:.4f}")
    
    # Compute Average Precision (AP)
    average_precision = precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    if verbose:
        print(f"    Average Precision (AP): {precision_sum:.4f} / {total_relevant} = {average_precision:.4f}")
        print(f"    Precision list shape: {len(precisions)} (one value per rank 1..{top_k})")
        print(f"    Recall list shape: {len(recalls)} (one value per rank 1..{top_k})")
    
    return precisions, recalls, average_precision


def evaluate_all_queries(all_features, image_paths, distance_function, dataset_folder, top_k=50, experiment_folder=None, distance_name='euclidean'):
    """
    1. treat each feature as a vector and calculate the distance between the query and the database images using function2
    2. sort the distances and store the top 10 distances and the corresponding image paths against each query in a list
    3. compute the confusion matrix for the top 1 results across all queries using compute_confusion_matrix function
    4. compute the precision and recall for each query and return the average precision and recall over all queries for the top_k results
    5. pass the averageprecision and average recall to plot the precision and recall curve using plot_precision_recall_curve function for top_k results
    6. return the average precision and average recall
    Args:
        all_features: numpy array of all features
        image_paths: list of image paths corresponding to all_features
        distance_function: distance function from cvpr_compare.py
        dataset_folder: path to dataset folder
        top_k: number of top results to evaluate
        experiment_folder: path to experiment folder to save plots (if None, plots are displayed)
        distance_name: name of the distance metric (for Mahalanobis distance handling)
    Returns:
        mean_precision_at_10: average precision@10 over all queries
        mean_recall_at_10: average recall@10 over all queries
        mean_average_precision: mean of all AP values (one per query)
        all_precisions: list of precision lists for each query (for PR curve)
        all_recalls: list of recall lists for each query (for PR curve)
        confusion_matrix: confusion matrix for top 1 results
    """
    # Analyze dataset to get class information
    dataset_info = analyze_dataset(dataset_folder)
    
    # Create experiment folder if provided
    if experiment_folder:
        os.makedirs(experiment_folder, exist_ok=True)
    
    # Print initial information
    print("\n" + "="*60)
    print("EVALUATION SETUP")
    print("="*60)
    print(f"All features shape: {all_features.shape}")
    print(f"  - Number of images: {all_features.shape[0]}")
    print(f"  - Feature dimension: {all_features.shape[1]}")
    print(f"Number of image paths: {len(image_paths)}")
    print(f"Top K: {top_k}")
    print("="*60)
    
    # Store distance lists for all queries (for confusion matrix)
    all_distance_lists = []
    
    # Store precision and recall for all queries
    all_precisions = []
    all_recalls = []
    all_aps = []
    
    num_queries = len(all_features)
    
    print(f"\nEvaluating {num_queries} queries...")
    print("Showing detailed output for first 3 queries as examples:\n")
    
    # Loop through each query
    for query_idx in range(num_queries):
        query_feature = all_features[query_idx]
        query_path = image_paths[query_idx]
        
        # Verbose output for first few queries
        verbose = (query_idx < 3)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PROCESSING QUERY {query_idx + 1}/{num_queries}")
            print("="*60)
            print(f"Query feature shape: {query_feature.shape}")
            print(f"All features shape: {all_features.shape}")
        
        # Get distance list (excluding query itself)
        all_distance_list = get_distance_list(query_feature, all_features, image_paths, distance_function, distance_name)
        
        if verbose:
            print(f"Distance list length (before removing query): {len(all_distance_list)}")
            print(f"Distance list type: list of (distance, image_path) tuples")
            print(f"First 3 distances: {[(d, os.path.basename(p)) for d, p in all_distance_list[:3]]}")
        
        # Remove query from results
        distance_list = [(d, path) for d, path in all_distance_list if path != query_path]
        
        if verbose:
            print(f"Distance list length (after removing query): {len(distance_list)}")
            print(f"Top {top_k} results (distances, filenames):")
            for i, (d, p) in enumerate(distance_list[:top_k], 1):
                print(f"  {i}. {d:.4f} - {os.path.basename(p)}")
        
        # Store for confusion matrix
        all_distance_lists.append((query_path, distance_list))
        
        # Compute precision and recall
        precisions, recalls, ap = compute_precision_recall(query_path, distance_list, dataset_info, top_k, verbose=verbose, query_idx=query_idx+1)
        
        if verbose:
            print(f"\n  Results for query {query_idx + 1}:")
            print(f"    Precisions shape: {len(precisions)} (list of {top_k} values)")
            print(f"    Recalls shape: {len(recalls)} (list of {top_k} values)")
            print(f"    Precision values: {[f'{p:.4f}' for p in precisions]}")
            print(f"    Recall values: {[f'{r:.4f}' for r in recalls]}")
            print(f"    Average Precision (AP): {ap:.4f}")
        
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_aps.append(ap)
        
        # Progress indicator
        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{num_queries} queries...")
    
    print(f"\n✅ Completed processing all {num_queries} queries")
    
    # Print aggregation information
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)
    print(f"All precisions shape: {len(all_precisions)} lists")
    print(f"  - Each list has {len(all_precisions[0])} values (one per rank 1..{top_k})")
    print(f"  - Total: {len(all_precisions)} queries × {len(all_precisions[0])} ranks")
    print(f"All recalls shape: {len(all_recalls)} lists")
    print(f"  - Each list has {len(all_recalls[0])} values (one per rank 1..{top_k})")
    print(f"All APs shape: {len(all_aps)} values (one per query)")
    print(f"  - AP values range: [{min(all_aps):.4f}, {max(all_aps):.4f}]")
    print(f"  - Sample AP values (first 5): {[f'{ap:.4f}' for ap in all_aps[:5]]}")
    print(f"All ARs shape: {len(all_aps)} values (one per query)")

    
    # Show example precision and recall lists
    print(f"\nExample precision lists (first 3 queries):")
    for i in range(min(3, len(all_precisions))):
        print(f"  Query {i+1}: {[f'{p:.4f}' for p in all_precisions[i]]}")
    
    print(f"\nExample recall lists (first 3 queries):")
    for i in range(min(3, len(all_recalls))):
        print(f"  Query {i+1}: {[f'{r:.4f}' for r in all_recalls[i]]}")
    
    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    confusion_matrix, class_labels = compute_confusion_matrix(all_distance_lists, dataset_info)
    print(f"Confusion matrix shape: {confusion_matrix.shape}")
    print(f"Class labels: {class_labels}")
    
    # Compute mean metrics
    print("\n" + "="*60)
    print("COMPUTING MEAN METRICS")
    print("="*60)
    
    # Extract precision@10 and recall@10 for all queries
    precision_at_10_values = [p[-1] for p in all_precisions]
    recall_at_10_values = [r[-1] for r in all_recalls]
    
    print(f"Precision@10 values shape: {len(precision_at_10_values)} (one per query)")
    print(f"  - First 5 values: {[f'{p:.4f}' for p in precision_at_10_values[:5]]}")
    print(f"  - Range: [{min(precision_at_10_values):.4f}, {max(precision_at_10_values):.4f}]")
    
    print(f"\nRecall@10 values shape: {len(recall_at_10_values)} (one per query)")
    print(f"  - First 5 values: {[f'{r:.4f}' for r in recall_at_10_values[:5]]}")
    print(f"  - Range: [{min(recall_at_10_values):.4f}, {max(recall_at_10_values):.4f}]")
    
    print(f"\nAP values shape: {len(all_aps)} (one per query)")
    print(f"  - First 5 values: {[f'{ap:.4f}' for ap in all_aps[:5]]}")
    print(f"  - Range: [{min(all_aps):.4f}, {max(all_aps):.4f}]")
    
    # Compute means
    mean_precision_at_10 = np.mean(precision_at_10_values)
    mean_recall_at_10 = np.mean(recall_at_10_values)
    mean_average_precision = np.mean(all_aps)
    
    print(f"\nMean Precision@10 = sum(precision@10) / {len(precision_at_10_values)}")
    print(f"  = {np.sum(precision_at_10_values):.4f} / {len(precision_at_10_values)}")
    print(f"  = {mean_precision_at_10:.4f}")
    
    print(f"\nMean Recall@10 = sum(recall@10) / {len(recall_at_10_values)}")
    print(f"  = {np.sum(recall_at_10_values):.4f} / {len(recall_at_10_values)}")
    print(f"  = {mean_recall_at_10:.4f}")
    
    print(f"\nMean Average Precision (mAP) = sum(AP) / {len(all_aps)}")
    print(f"  = {np.sum(all_aps):.4f} / {len(all_aps)}")
    print(f"  = {mean_average_precision:.4f}")
    
    # Plot and save precision-recall curve
    pr_curve_path = None
    if experiment_folder:
        pr_curve_path = os.path.join(experiment_folder, f"pr_curve_top{top_k}.png")
    plot_precision_recall_curve(all_precisions, all_recalls, title=f"Precision-Recall Curve (Top {top_k})", save_path=pr_curve_path)
    
    # Plot and save confusion matrix
    confusion_matrix_path = None
    if experiment_folder:
        confusion_matrix_path = os.path.join(experiment_folder, "confusion_matrix_top1.png")
    plot_confusion_matrix(confusion_matrix, class_labels, title="Confusion Matrix (Top 1)", save_path=confusion_matrix_path)
    
    return mean_precision_at_10, mean_recall_at_10, mean_average_precision, all_precisions, all_recalls, confusion_matrix


def plot_precision_recall_curve(all_precisions, all_recalls, title="Precision-Recall Curve", save_path=None):
    """
    plot the precision and recall curve
    Args:
        all_precisions: list of precision lists for each query
        all_recalls: list of recall lists for each query
        title: plot title
        save_path: path to save the plot as PNG (if None, displays the plot)
    """
    # Average precision and recall at each rank k across all queries
    num_queries = len(all_precisions)
    top_k = len(all_precisions[0])
    print("top_k: ", top_k)
    averaged_precisions = []
    averaged_recalls = []
    
    for k in range(top_k):
        precisions_at_k = [p[k] for p in all_precisions]
        recalls_at_k = [r[k] for r in all_recalls]
        if k == 0:
            print(f"precisions_at_k: {precisions_at_k}")
            print(f"recalls_at_k: {recalls_at_k}")
            print(f"np.mean(precisions_at_k): {np.mean(precisions_at_k)}")
            print(f"np.mean(recalls_at_k): {np.mean(recalls_at_k)}")
        averaged_precisions.append(np.mean(precisions_at_k))
        averaged_recalls.append(np.mean(recalls_at_k))
        
    print(f"averaged_precisions: {averaged_precisions}")
    print(f"averaged_recalls: {averaged_recalls}")
    plt.figure(figsize=(8, 6))
    plt.plot(averaged_recalls, averaged_precisions, 'g-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to max values
    max_recall = max(averaged_recalls) if averaged_recalls else 1.0
    max_precision = max(averaged_precisions) if averaged_precisions else 1.0
    
    # Set x-axis: 0 to max_recall with 0.01 intervals
    x_max = max_recall * 1.05  # Add small margin (5%)
    print(f"max_recall: {max_recall}, x_max: {x_max}")
    x_ticks = np.arange(0, x_max + 0.01, 0.01)
    print(f"x_ticks: {x_ticks}")
    plt.xlim([0, x_max])
    plt.xticks(x_ticks)
    
    # Set y-axis: 0 to max_precision with 0.1 intervals
    y_max = max_precision * 1.05  # Add small margin (5%)
    print(f"max_precision: {max_precision}, y_max: {y_max}")
    y_ticks = np.arange(0, y_max + 0.1, 0.1)
    print(f"y_ticks: {y_ticks}")
    plt.ylim([0, y_max])
    plt.yticks(y_ticks)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def compute_confusion_matrix(all_distance_lists, dataset_info):
    """
    compute the confusion matrix for the top 1 results across all queries
    the confusion matrix is a matrix of size (number of classes, number of classes)
    the confusion matrix is computed by counting the number of times a class is predicted as another class in the top 1 results across all queries
    use the get_class_id_from_filename function to get the class id of the query image and the images in the distance list
    Args:
        all_distance_lists: list of (query_path, distance_list) tuples for each query
        dataset_info: dictionary from analyze_dataset function
    Returns:
        confusion_matrix: confusion matrix for the top 1 results across all queries (numpy array)
        class_labels: list of class labels
    """
    classes = dataset_info['classes']
    num_classes = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Process each query
    for query_path, distance_list in all_distance_lists:
        if len(distance_list) == 0:
            continue
        
        # Get true class (query class)
        query_filename = os.path.basename(query_path)
        true_class = get_class_id_from_filename(query_filename)
        
        if true_class == -1 or true_class not in class_to_idx:
            continue
        
        # Get predicted class (top 1 result)
        _, top1_path = distance_list[0]
        top1_filename = os.path.basename(top1_path)
        predicted_class = get_class_id_from_filename(top1_filename)
        
        if predicted_class == -1 or predicted_class not in class_to_idx:
            continue
        
        # Update confusion matrix
        true_idx = class_to_idx[true_class]
        pred_idx = class_to_idx[predicted_class]
        confusion_matrix[true_idx, pred_idx] += 1
    
    return confusion_matrix, classes


def plot_confusion_matrix(confusion_matrix, class_labels, title="Confusion Matrix", save_path=None):
    """
    Plot and save the confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: numpy array of shape (num_classes, num_classes)
        class_labels: list of class labels
        title: plot title
        save_path: path to save the plot as PNG (if None, displays the plot)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           title=title,
           ylabel='True Class',
           xlabel='Predicted Class')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def load_descriptors(descriptor_folder, subfolder):
    """
    Load all image descriptors from .mat files.
    
    Args:
        descriptor_folder: Base folder containing descriptors
        subfolder: Subfolder name (e.g., 'globalRGBhisto')
    
    Returns:
        tuple: (features_array, image_file_paths)
            - features_array: numpy array of shape (N, feature_dim)
            - image_file_paths: list of image file paths
    """
    features = []
    image_paths = []
    descriptor_path = os.path.join(descriptor_folder, subfolder)
    
    if not os.path.exists(descriptor_path):
        raise FileNotFoundError(f"Descriptor folder not found: {descriptor_path}")
    
    dataset_images_folder = './MSRC_ObjCategImageDatabase_v2/Images'
    
    for filename in sorted(os.listdir(descriptor_path)):
        if filename.endswith('.mat'):
            # Load descriptor
            desc_path = os.path.join(descriptor_path, filename)
            desc_data = sio.loadmat(desc_path)
            features.append(desc_data['F'][0])
            
            # Construct corresponding image path
            image_filename = filename.replace('.mat', '.bmp')
            image_path = os.path.join(dataset_images_folder, image_filename)
            image_paths.append(image_path)
    
    return np.array(features), image_paths


def get_available_descriptors(descriptor_folder):
    """
    Get list of available descriptor subfolders.
    
    Args:
        descriptor_folder: Base folder containing descriptors
    
    Returns:
        list: List of available descriptor subfolder names (excluding pca_models)
    """
    if not os.path.exists(descriptor_folder):
        return []
    
    all_items = os.listdir(descriptor_folder)
    descriptors = []
    
    for item in all_items:
        item_path = os.path.join(descriptor_folder, item)
        if os.path.isdir(item_path) and item != 'pca_models':
            # Check if it contains .mat files
            if any(f.endswith('.mat') for f in os.listdir(item_path)):
                descriptors.append(item)
    
    return sorted(descriptors)


def get_user_descriptor_choice(available_descriptors):
    """
    Get user input for descriptor type selection.
    
    Args:
        available_descriptors: list of available descriptor subfolder names
    
    Returns:
        str: Selected descriptor subfolder name
    """
    print("\n" + "="*60)
    print("DESCRIPTOR TYPE SELECTION")
    print("="*60)
    for idx, desc in enumerate(available_descriptors, 1):
        print(f"  {idx}. {desc}")
    print("="*60)
    
    while True:
        try:
            choice = int(input(f"Select descriptor type (1-{len(available_descriptors)}): ").strip())
            if 1 <= choice <= len(available_descriptors):
                selected = available_descriptors[choice - 1]
                print(f"Selected: {selected}")
                return selected
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(available_descriptors)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_user_distance_metric():
    """
    Get user input for distance metric selection.
    
    Returns:
        tuple: (distance_name, distance_function, distance_display)
    """
    distance_metrics = {
        '1': ('euclidean', euclidean_distance, 'Euclidean Distance (L2)'),
        '2': ('manhattan', manhattan_distance, 'Manhattan Distance (L1)'),
        '3': ('chebyshev', chebyshev_distance, 'Chebyshev Distance (L∞)'),
        '4': ('minkowski', minkowski_distance, 'Minkowski Distance'),
        '5': ('mahalanobis', mahalanobis_distance, 'Mahalanobis Distance')
    }
    
    print("\n" + "="*60)
    print("DISTANCE METRIC SELECTION")
    print("="*60)
    for key, (_, _, name) in distance_metrics.items():
        print(f"  {key}. {name}")
    print("="*60)
    
    while True:
        choice = input("Select distance metric (1-5): ").strip()
        if choice in distance_metrics:
            distance_name, distance_func, distance_display = distance_metrics[choice]
            print(f"Selected: {distance_display}")
            return distance_name, distance_func, distance_display
        else:
            print("Invalid choice. Please enter 1-5.")


def main():
    """
    Main function to evaluate descriptors with user-selected distance metrics.
    """
    print("="*60)
    print("EVALUATION FRAMEWORK FOR VISUAL SEARCH SYSTEM")
    print("="*60)
    
    # Configuration
    DESCRIPTOR_FOLDER = './descriptors'
    DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
    EXPERIMENTS_FOLDER = './experiments'
    TOP_K = 30
    
    # Get available descriptors
    print("\nScanning for available descriptors...")
    available_descriptors = get_available_descriptors(DESCRIPTOR_FOLDER)
    
    if not available_descriptors:
        print(f"❌ No descriptors found in {DESCRIPTOR_FOLDER}")
        print("Please run cvpr_computedescriptors.py first to generate descriptors.")
        return
    
    print(f"✅ Found {len(available_descriptors)} descriptor(s): {', '.join(available_descriptors)}")
    
    # Get user input for descriptor selection
    selected_descriptor = get_user_descriptor_choice(available_descriptors)
    
    # Get user input for distance metric selection
    distance_name, distance_func, distance_display = get_user_distance_metric()
    
    # Load descriptors
    print(f"\nLoading descriptors from: {selected_descriptor}")
    try:
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, selected_descriptor)
        num_images = len(image_paths)
        print(f"✅ Loaded {num_images} image descriptors")
        print(f"   Descriptor dimension: {all_features.shape[1]}")
    except Exception as e:
        print(f"❌ Error loading descriptors: {e}")
        return
    
    # Create experiment folder
    experiment_subfolder = f"{selected_descriptor}_{distance_name}"
    experiment_path = os.path.join(EXPERIMENTS_FOLDER, experiment_subfolder)
    
    print(f"\n{'='*60}")
    print("EVALUATION SETTINGS")
    print("="*60)
    print(f"Descriptor: {selected_descriptor}")
    print(f"Distance Metric: {distance_display}")
    print(f"Top K: {TOP_K}")
    print(f"Experiment Folder: {experiment_path}")
    print("="*60)
    
    # Run evaluation
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    print("This may take a while as we evaluate all 591 queries...")
    print("="*60)
    
    try:
        mean_precision_at_10, mean_recall_at_10, mean_average_precision, \
        all_precisions, all_recalls, confusion_matrix = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_function=distance_func,
            dataset_folder=DATASET_FOLDER,
            top_k=TOP_K,
            experiment_folder=experiment_path,
            distance_name=distance_name
        )
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Precision@10: {mean_precision_at_10:.4f}")
        print(f"Mean Recall@10: {mean_recall_at_10:.4f}")
        print(f"Mean Average Precision (mAP): {mean_average_precision:.4f}")
        print("="*60)
        print(f"\n✅ Results saved to: {experiment_path}")
        print(f"   - PR Curve: pr_curve_top{TOP_K}.png")
        print(f"   - Confusion Matrix: confusion_matrix_top1.png")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()