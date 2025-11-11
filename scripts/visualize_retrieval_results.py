"""
Visual Search Results Visualization Script
EEE3032 - Computer Vision and Pattern Recognition

This script generates visualizations showing query images and their top 10
retrieved results for different descriptor-distance combinations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvpr_evaluate import load_descriptors
from cvpr_compare import (
    euclidean_distance,
    chi_square_distance,
    mahalanobis_distance
)
from cvpr_pca import load_pca_model

DESCRIPTOR_FOLDER = './descriptors'
DATASET_IMAGES_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
PCA_MODELS_FOLDER = './descriptors/pca_models'
OUTPUT_FOLDER = './experiments/retrieval_visualizations'

# Configuration: n random queries
N_QUERIES = 1
TOP_K = 10

# Descriptor-distance combinations to visualize
DESCRIPTOR_CONFIGS = [
    {
        'descriptor': 'globalRGBhisto_8bins',
        'distance_func': euclidean_distance,
        'distance_name': 'euclidean',
        'display_name': 'Global RGB Histogram (8 bins) + Euclidean Distance',
        'filename': 'gch_8bins_euclidean'
    },
    {
        'descriptor': 'spatialGridRGBhisto_8bins_4x4',
        'distance_func': chi_square_distance,
        'distance_name': 'chi_square',
        'display_name': 'Spatial Grid RGB Histogram (8 bins, 4×4) + Chi-square Distance',
        'filename': 'spatial_4x4_chi_square'
    },
    {
        'descriptor': 'globalRGBhisto_8bins_PCA18',
        'distance_func': mahalanobis_distance,
        'distance_name': 'mahalanobis',
        'display_name': 'Global RGB Histogram PCA18 + Mahalanobis Distance',
        'filename': 'gch_pca18_mahalanobis'
    },
    {
        'descriptor': 'spatialGridRGBhisto_8bins_4x4_PCA16',
        'distance_func': mahalanobis_distance,
        'distance_name': 'mahalanobis',
        'display_name': 'Spatial Grid RGB Histogram PCA16 + Mahalanobis Distance',
        'filename': 'spatial_pca16_mahalanobis'
    }
]


def load_and_resize_image(image_path, target_size=(160, 120)):
    """
    Load an image and resize it to target size.
    
    Args:
        image_path: Path to the image file
        target_size: Target (width, height) for resizing
    
    Returns:
        numpy.ndarray: Resized image (BGR format)
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create placeholder if image loading fails
        placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(placeholder, 'LOAD FAIL', (10, target_size[1] // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return placeholder
    
    img_resized = cv2.resize(img, target_size)
    return img_resized


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


def add_label_to_image(img, label_text, position='bottom'):
    """
    Add a text label to an image.
    
    Args:
        img: Image array (BGR format)
        label_text: Text to add
        position: 'top' or 'bottom'
    
    Returns:
        numpy.ndarray: Image with label
    """
    img_labeled = img.copy()
    h, w = img.shape[:2]
    
    # Draw background bar (make it taller for longer text)
    bar_height = 25
    if position == 'bottom':
        cv2.rectangle(img_labeled, (0, h - bar_height), (w, h), (0, 0, 0), thickness=-1)
        cv2.putText(img_labeled, label_text, (5, h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:  # top
        cv2.rectangle(img_labeled, (0, 0), (w, bar_height), (0, 0, 0), thickness=-1)
        cv2.putText(img_labeled, label_text, (5, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_labeled


def compute_distances(query_feature, all_features, distance_func, distance_name='euclidean', pca_model=None):
    """
    Compute distances between query and all database features.
    
    Args:
        query_feature: Feature vector of the query image
        all_features: Array of all feature vectors
        distance_func: Distance function to use
        distance_name: Name of the distance metric
        pca_model: PCA model if using PCA descriptors (for Mahalanobis)
    
    Returns:
        list: List of (distance, index) tuples sorted by distance
    """
    distances = []
    
    # For Mahalanobis distance, compute covariance matrix
    if distance_name == 'mahalanobis':
        if pca_model is not None:
            # For PCA-projected features, use diagonal covariance matrix
            explained_var = pca_model['explained_variance']
            n_dims = len(query_feature.flatten())
            covariance_matrix = np.diag(explained_var[:n_dims])
            covariance_matrix = covariance_matrix + np.eye(n_dims) * 1e-6
        else:
            # For non-PCA features, compute full covariance matrix
            features_flat = np.array([f.flatten() for f in all_features])
            covariance_matrix = np.cov(features_flat.T)
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


def create_retrieval_montage(query_indices, all_features, image_paths, distance_func, 
                             distance_name, pca_model, top_k=10, n_queries=2):
    """
    Create a montage showing n query images and their top k retrieved results.
    Layout: n rows × (1 query + k results) = n rows × (k+1) columns
    
    Args:
        query_indices: List of query image indices
        all_features: Array of all feature vectors
        image_paths: List of all image paths
        distance_func: Distance function to use
        distance_name: Name of the distance metric
        pca_model: PCA model if using PCA descriptors
        top_k: Number of top results to display
        n_queries: Number of query images
    
    Returns:
        numpy.ndarray: Montage image canvas
    """
    tile_w, tile_h = 160, 120  # Size of each image tile
    
    # Create canvas: n rows × (k+1) columns
    canvas = np.zeros((n_queries * tile_h, (top_k + 1) * tile_w, 3), dtype=np.uint8)
    
    for row_idx, query_idx in enumerate(query_indices):
        query_feature = all_features[query_idx]
        query_path = image_paths[query_idx]
        query_filename = os.path.basename(query_path)
        query_class = get_class_id_from_filename(query_filename)
        
        # Compute distances
        distances = compute_distances(query_feature, all_features, distance_func, 
                                    distance_name, pca_model)
        
        # Exclude query from results
        ranked_results = [(d, i) for (d, i) in distances if i != query_idx]
        
        # Load and label query image
        query_img = load_and_resize_image(query_path, (tile_w, tile_h))
        query_label = f'Query (C={query_class})'
        query_img = add_label_to_image(query_img, query_label, position='bottom')
        
        # Place query image in first column
        y0, y1 = row_idx * tile_h, (row_idx + 1) * tile_h
        x0, x1 = 0, tile_w
        canvas[y0:y1, x0:x1] = query_img
        
        # Load and place top k results
        for col_idx in range(top_k):
            _, result_idx = ranked_results[col_idx]
            result_path = image_paths[result_idx]
            result_filename = os.path.basename(result_path)
            result_class = get_class_id_from_filename(result_filename)
            
            # Check if result is relevant (same class as query)
            is_relevant = (result_class == query_class)
            relevance_marker = "✓" if is_relevant else "✗"
            
            # Load result image
            result_img = load_and_resize_image(result_path, (tile_w, tile_h))
            
            # Add border color based on relevance (green for relevant, red for irrelevant)
            border_color = (0, 255, 0) if is_relevant else (0, 0, 255)  # Green or Red in BGR
            border_thickness = 3
            cv2.rectangle(result_img, (0, 0), (tile_w - 1, tile_h - 1), 
                         border_color, border_thickness)
            
            # Add label with rank, relevance marker, class, and distance
            distance = ranked_results[col_idx][0]
            label = f"#{col_idx+1} {relevance_marker} C={result_class} d={distance:.3f}"
            result_img = add_label_to_image(result_img, label, position='bottom')
            
            # Place in canvas
            x0, x1 = (col_idx + 1) * tile_w, (col_idx + 2) * tile_w
            canvas[y0:y1, x0:x1] = result_img
    
    return canvas


def visualize_retrieval_results(config, query_indices, save_path):
    """
    Generate visualization for a specific descriptor-distance combination.
    
    Args:
        config: Dictionary with descriptor, distance_func, distance_name, etc.
        query_indices: List of query image indices
        save_path: Path to save the visualization
    """
    descriptor_name = config['descriptor']
    distance_func = config['distance_func']
    distance_name = config['distance_name']
    display_name = config['display_name']
    
    print(f"\n📊 Processing: {display_name}")
    print("-" * 80)
    
    # Load descriptors
    all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
    print(f"  Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
    
    # Load PCA model if needed
    pca_model = None
    if distance_name == 'mahalanobis' and '_PCA' in descriptor_name:
        # Extract base descriptor name
        base_name = descriptor_name.split('_PCA')[0]
        pca_model_path = os.path.join(PCA_MODELS_FOLDER, f'{base_name}_pca.mat')
        
        if os.path.exists(pca_model_path):
            pca_model = load_pca_model(pca_model_path)
            print(f"  ✅ Loaded PCA model: {pca_model_path}")
        else:
            print(f"  ⚠️  PCA model not found: {pca_model_path}")
    
    # Create montage
    montage = create_retrieval_montage(
        query_indices=query_indices,
        all_features=all_features,
        image_paths=image_paths,
        distance_func=distance_func,
        distance_name=distance_name,
        pca_model=pca_model,
        top_k=TOP_K,
        n_queries=N_QUERIES
    )
    
    # Convert BGR to RGB for matplotlib
    montage_rgb = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
    
    # Create figure and display
    fig, ax = plt.subplots(figsize=(20, 6 * N_QUERIES))
    ax.imshow(montage_rgb)
    ax.axis('off')
    ax.set_title(display_name, fontsize=16, fontweight='bold', pad=20)
    
    # Add grid lines to separate query from results
    for row in range(N_QUERIES + 1):
        y = row * 120
        ax.axhline(y=y, color='red', linewidth=2, alpha=0.5)
    
    # Add vertical line after query column
    ax.axvline(x=160, color='red', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Visualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to generate retrieval visualizations.
    """
    print("="*80)
    print("VISUAL SEARCH RESULTS VISUALIZATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load descriptors from first config to get the image order
    # (all descriptors should have the same image order)
    first_config = DESCRIPTOR_CONFIGS[0]
    all_features_temp, image_paths_temp = load_descriptors(DESCRIPTOR_FOLDER, first_config['descriptor'])
    
    if len(image_paths_temp) < N_QUERIES:
        print(f"❌ Not enough images in dataset. Need at least {N_QUERIES}, found {len(image_paths_temp)}")
        return
    
    # Select specific query images from class 2 and class 13
    desired_classes = [2, 13]
    query_indices = []
    # Build class list from filenames
    filenames = [os.path.basename(p) for p in image_paths_temp]
    classes = [get_class_id_from_filename(fn) for fn in filenames]
    for desired_class in desired_classes:
        indices_of_class = [i for i, c in enumerate(classes) if c == desired_class]
        if indices_of_class:
            # Pick first index of the desired class
            query_indices.append(indices_of_class[0])
        else:
            print(f"⚠️  No images found for desired class {desired_class}. Selecting a random image instead.")
            # Fallback to random index if class not found
            np.random.seed(42)
            query_indices.append(np.random.randint(0, len(image_paths_temp)))
    # Ensure length matches N_QUERIES (in case config differs)
    query_indices = np.array(query_indices[:N_QUERIES])
    
    print(f"\n📷 Selected {N_QUERIES} query images from classes {desired_classes}:")
    for i, idx in enumerate(query_indices, 1):
        query_filename = os.path.basename(image_paths_temp[idx])
        query_class = get_class_id_from_filename(query_filename)
        print(f"  Query {i}: {query_filename} (index {idx})  -> class {query_class}")
    
    # Generate visualization for each descriptor-distance combination
    for config in DESCRIPTOR_CONFIGS:
        save_filename = f"retrieval_{config['filename']}.png"
        save_path = os.path.join(OUTPUT_FOLDER, save_filename)
        
        visualize_retrieval_results(config, query_indices, save_path)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\n📁 Generated files in {OUTPUT_FOLDER}:")
    for config in DESCRIPTOR_CONFIGS:
        save_filename = f"retrieval_{config['filename']}.png"
        print(f"   • {save_filename}")
    print("="*80)


if __name__ == "__main__":
    main()

