"""
Visual Search System for Image Retrieval
EEE3032 - Computer Vision and Pattern Recognition

This module performs content-based image retrieval using RGB histogram descriptors
and evaluates the system using precision-recall metrics.
"""

import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import (
    cvpr_compare, 
    euclidean_distance, 
    manhattan_distance, 
    chebyshev_distance, 
    minkowski_distance, 
    mahalanobis_distance
)

# Configuration
import matplotlib.pyplot as plt

# ============================================================================
# Configuration Constants
# ============================================================================
DESCRIPTOR_FOLDER = './descriptors'
DATASET_IMAGES_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
TOP_K = 20
DEBUG = False  # Set to True for verbose debug output
TILE_SIZE = (320, 212)  # Width, Height for visualization tiles

# Descriptor type mapping to subfolders
DESCRIPTOR_TYPES = {
    '1': ('globalRGBhisto', 'Global RGB Histogram'),
    '2': ('spatialGridRGBhisto', 'Spatial Grid RGB Histogram')
}

# Distance metric mapping
DISTANCE_METRICS = {
    '1': ('euclidean', euclidean_distance, 'Euclidean Distance (L2)'),
    '2': ('manhattan', manhattan_distance, 'Manhattan Distance (L1)'),
    '3': ('chebyshev', chebyshev_distance, 'Chebyshev Distance (L∞)'),
    '4': ('minkowski', minkowski_distance, 'Minkowski Distance'),
    '5': ('mahalanobis', mahalanobis_distance, 'Mahalanobis Distance')
}


# ============================================================================
# User Input Functions
# ============================================================================

def get_user_descriptor_choice():
    """
    Get user input for descriptor type selection.
    
    Returns:
        tuple: (subfolder_name, descriptor_name)
    """
    print("\n" + "="*60)
    print("DESCRIPTOR TYPE SELECTION")
    print("="*60)
    for key, (subfolder, name) in DESCRIPTOR_TYPES.items():
        print(f"  {key}. {name}")
    print("="*60)
    
    while True:
        choice = input("Select descriptor type (1-2): ").strip()
        if choice in DESCRIPTOR_TYPES:
            subfolder, name = DESCRIPTOR_TYPES[choice]
            print(f"Selected: {name}")
            return subfolder, name
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_user_distance_metric():
    """
    Get user input for distance metric selection.
    
    Returns:
        tuple: (metric_name, distance_function, display_name)
    """
    print("\n" + "="*60)
    print("DISTANCE METRIC SELECTION")
    print("="*60)
    for key, (name, func, display) in DISTANCE_METRICS.items():
        print(f"  {key}. {display}")
    print("="*60)
    
    while True:
        choice = input("Select distance metric (1-5): ").strip()
        if choice in DISTANCE_METRICS:
            name, func, display = DISTANCE_METRICS[choice]
            print(f"Selected: {display}")
            
            # Additional input for Minkowski distance
            if choice == '4':
                while True:
                    try:
                        p = float(input("Enter Minkowski parameter p (e.g., 1.5, 2, 3): "))
                        if p > 0:
                            # Create a wrapper function with fixed p
                            def minkowski_p(F1, F2):
                                return minkowski_distance(F1, F2, p=p)
                            return name, minkowski_p, f'Minkowski Distance (p={p})'
                        else:
                            print("Parameter p must be positive.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            
            return name, func, display
        else:
            print("Invalid choice. Please enter 1-5.")


# ============================================================================
# Utility Functions
# ============================================================================

def get_class_id_from_path(path):
    """
    Extract the integer class id from an image path.
    
    Args:
        path: Image file path (e.g., './.../12_33_s.bmp')
    
    Returns:
        int: Class ID (e.g., 12) or -1 if parsing fails
    """
    base = os.path.basename(path)
    try:
        return int(base.split('_')[0])
    except (ValueError, IndexError):
        return -1


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
    
    for filename in os.listdir(descriptor_path):
        if filename.endswith('.mat'):
            # Load descriptor
            desc_path = os.path.join(descriptor_path, filename)
            desc_data = sio.loadmat(desc_path)
            features.append(desc_data['F'][0])
            
            # Construct corresponding image path
            image_filename = filename.replace('.mat', '.bmp')
            image_path = os.path.join(DATASET_IMAGES_FOLDER, image_filename)
            image_paths.append(image_path)
    
    return np.array(features), image_paths


def compute_distances(query_feature, all_features, distance_func=euclidean_distance, distance_name='euclidean'):
    """
    Compute distances between query and all database features.
    
    Args:
        query_feature: Feature vector of the query image
        all_features: Array of all feature vectors
        distance_func: Distance function to use
        distance_name: Name of the distance metric (for Mahalanobis)
    
    Returns:
        list: List of (distance, index) tuples sorted by distance
    """
    distances = []
    
    # For Mahalanobis distance, we need to compute covariance matrix from all features
    if distance_name == 'mahalanobis':
        # Flatten all features for covariance computation
        features_flat = np.array([f.flatten() for f in all_features])
        covariance_matrix = np.cov(features_flat.T)
        # Add small epsilon for numerical stability
        covariance_matrix = covariance_matrix + np.eye(len(covariance_matrix)) * 1e-6
        
        for i, candidate in enumerate(all_features):
            distance = distance_func(query_feature, candidate, covariance_matrix=covariance_matrix)
            distances.append((distance, i))
    else:
        # For other distance metrics
        for i, candidate in enumerate(all_features):
            distance = distance_func(query_feature, candidate)
            distances.append((distance, i))
    
    distances.sort(key=lambda x: x[0])  # Sort by distance (ascending)
    return distances


# ============================================================================
# Visualization Functions
# ============================================================================

def load_and_label_image(image_path, label_text, target_size=(160, 106)):
    """
    Load an image, resize it, and overlay a label.
    
    Args:
        image_path: Path to the image file
        label_text: Text label to overlay
        target_size: Target (width, height) for resizing
    
    Returns:
        numpy.ndarray: Labeled and resized image
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create placeholder if image loading fails
        placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(placeholder, 'LOAD FAIL', (10, target_size[1] // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, label_text, (10, target_size[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return placeholder
    
    img = cv2.resize(img, target_size)
    # Draw label background bar
    cv2.rectangle(img, (0, target_size[1] - 18), (target_size[0], target_size[1]),
                  (0, 0, 0), thickness=-1)
    cv2.putText(img, label_text, (6, target_size[1] - 6),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def create_results_montage(query_path, ranked_results, image_paths, top_k=10):
    """
    Create a montage visualization showing query and top results.
    
    Layout:
    - Row 1: Query image (centered in 5-column row)
    - Row 2+: Results displayed in groups of 5 per row
    
    Args:
        query_path: Path to query image
        ranked_results: List of (distance, index) tuples (sorted, excluding query)
        image_paths: List of all image paths
        top_k: Number of top results to display
    
    Returns:
        numpy.ndarray: Montage image canvas
    """
    tile_w, tile_h = TILE_SIZE
    
    # Add query image
    query_tile = load_and_label_image(query_path, 'Query', TILE_SIZE)
    
    # Add top K results
    result_tiles = []
    for rank, (distance, idx) in enumerate(ranked_results[:top_k], start=1):
        img_path = image_paths[idx]
        label = f'{rank}: {distance:.3f}'
        result_tiles.append(load_and_label_image(img_path, label, TILE_SIZE))
    
    # Layout: 5 columns per row, query row + result rows
    cols_per_row = 5
    num_result_rows = int(np.ceil(top_k / cols_per_row))
    total_rows = 1 + num_result_rows  # 1 for query, rest for results
    
    # Create canvas
    canvas = np.zeros((total_rows * tile_h, cols_per_row * tile_w, 3), dtype=np.uint8)
    
    # Row 1: Query image centered (in middle of 5 columns = column 2, index 2)
    query_col = cols_per_row // 2  # Column 2 (0-indexed)
    y0, y1 = 0, tile_h
    x0, x1 = query_col * tile_w, (query_col + 1) * tile_w
    canvas[y0:y1, x0:x1] = query_tile
    
    # Remaining rows: Results in groups of 5
    for row_idx in range(num_result_rows):
        start_idx = row_idx * cols_per_row
        end_idx = min(start_idx + cols_per_row, top_k)
        
        y0, y1 = (row_idx + 1) * tile_h, (row_idx + 2) * tile_h
        
        for col_idx in range(end_idx - start_idx):
            tile = result_tiles[start_idx + col_idx]
            x0, x1 = col_idx * tile_w, (col_idx + 1) * tile_w
            canvas[y0:y1, x0:x1] = tile
    
    return canvas


def print_ranked_results(ranked_results, image_paths, top_k=10):
    """
    Print ranked results with class IDs and distances.
    
    Args:
        ranked_results: List of (distance, index) tuples (sorted, excluding query)
        image_paths: List of all image paths
        top_k: Number of results to print
    """
    print(f"\nTop {top_k} ranked results with class IDs:")
    for rank, (distance, idx) in enumerate(ranked_results[:top_k], start=1):
        class_id = get_class_id_from_path(image_paths[idx])
        filename = os.path.basename(image_paths[idx])
        print(f"{rank:4d}: class={class_id:2d}, dist={distance:.4f}, file={filename}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("VISUAL SEARCH SYSTEM")
    print("="*60)
    
    # Get user input for descriptor type
    descriptor_subfolder, descriptor_name = get_user_descriptor_choice()
    
    # Get user input for distance metric
    distance_name, distance_func, distance_display = get_user_distance_metric()
    
    # Load descriptors
    print(f"\nLoading descriptors from: {descriptor_subfolder}")
    all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_subfolder)
    num_images = len(image_paths)
    print(f"Loaded {num_images} image descriptors")
    print(f"Descriptor dimension: {all_features.shape[1]}")
    
    # Select random query
    # query_idx = randint(0, num_images - 1)
    query_idx = 50
    query_path = image_paths[query_idx]
    print(f"\nQuery image: {query_path}")
    print(f"Descriptor: {descriptor_name}")
    print(f"Distance metric: {distance_display}")
    
    # Compute distances and rank results
    query_feature = all_features[query_idx]
    distances = compute_distances(query_feature, all_features, distance_func, distance_name)
    
    # Exclude query from results for visualization
    ranked_results = [(d, i) for (d, i) in distances if i != query_idx]
    
    # Visualize results
    print(f"\nTop {TOP_K} most similar images:")
    montage = create_results_montage(query_path, ranked_results, image_paths, TOP_K)
    cv2.imshow('Query + Top Results (press any key)', montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print ranked results
    print_ranked_results(ranked_results, image_paths, TOP_K)


if __name__ == "__main__":
    main()
