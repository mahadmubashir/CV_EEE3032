"""
Spatial Grid RGB Histogram Evaluation Script
EEE3032 - Computer Vision and Pattern Recognition

This script evaluates spatial grid descriptors with different grid sizes and distance metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cvpr_evaluate import (
    load_descriptors,
    evaluate_all_queries,
    plot_precision_recall_curve
)
from cvpr_compare import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
    mahalanobis_distance,
    chi_square_distance,
    bhattacharyya_distance
)

DESCRIPTOR_FOLDER = './descriptors'
DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
EXPERIMENTS_FOLDER = './experiments/spatial_grid_comparison'
TOP_K = 10

# Spatial grid descriptors to compare
SPATIAL_GRID_DESCRIPTORS = [
    'spatialGridRGBhisto_8bins_2x2',
    'spatialGridRGBhisto_8bins_4x4',
    'spatialGridRGBhisto_8bins_6x6',
    'spatialGridRGBhisto_8bins_8x8'
]

# Distance metrics to compare
DISTANCE_METRICS = {
    'euclidean': (euclidean_distance, 'Euclidean Distance'),
    'manhattan': (manhattan_distance, 'Manhattan Distance'),
    'chebyshev': (chebyshev_distance, 'Chebyshev Distance'),
    'minkowski': (lambda F1, F2: minkowski_distance(F1, F2, p=3), 'Minkowski Distance (p=3)'),
    'chi_square': (chi_square_distance, 'Chi-square Distance'),
    'bhattacharyya': (bhattacharyya_distance, 'Bhattacharyya Distance')
}


def plot_pr_curves_comparison(all_results, title, save_path):
    """
    Plot multiple PR curves on the same graph for comparison.
    
    Args:
        all_results: list of dicts, each with 'precisions', 'recalls', 'mAP', 'label'
        title: plot title
        save_path: path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for idx, result in enumerate(all_results):
        all_precisions = result['precisions']
        all_recalls = result['recalls']
        mAP = result['mAP']
        label = result['label']
        color = colors[idx % len(colors)]
        
        # Average precision and recall at each rank k across all queries
        top_k = len(all_precisions[0])
        averaged_precisions = []
        averaged_recalls = []
        
        for k in range(top_k):
            precisions_at_k = [p[k] for p in all_precisions]
            recalls_at_k = [r[k] for r in all_recalls]
            averaged_precisions.append(np.mean(precisions_at_k))
            averaged_recalls.append(np.mean(recalls_at_k))
        
        # Plot with label including mAP
        plt.plot(averaged_recalls, averaged_precisions, '-', linewidth=2, 
                color=color, label=f"{label} (mAP={mAP:.4f})")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # Set axis limits based on averaged data
    all_avg_recalls = []
    all_avg_precisions = []
    for result in all_results:
        all_precisions = result['precisions']
        all_recalls = result['recalls']
        top_k = len(all_precisions[0])
        
        for k in range(top_k):
            precisions_at_k = [p[k] for p in all_precisions]
            recalls_at_k = [r[k] for r in all_recalls]
            all_avg_precisions.append(np.mean(precisions_at_k))
            all_avg_recalls.append(np.mean(recalls_at_k))
    
    if all_avg_recalls and all_avg_precisions:
        max_recall = max(all_avg_recalls)
        max_precision = max(all_avg_precisions)
        x_max = max_recall * 1.05 if max_recall > 0 else 1.0
        y_max = max_precision * 1.05 if max_precision > 0 else 1.0
        plt.xlim([0, x_max])
        plt.ylim([0, y_max])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ PR curves comparison saved to: {save_path}")
    plt.close()


def plot_map_comparison_line(map_results, title, save_path):
    """
    Plot line graph comparing mAP across different grid sizes.
    
    Args:
        map_results: list of dicts with 'label' and 'mAP'
        title: plot title
        save_path: path to save the plot
    """
    labels = [r['label'] for r in map_results]
    maps = [r['mAP'] for r in map_results]
    
    # Extract numeric values for x-axis (grid sizes: 2x2, 4x4, 8x8)
    x_values = []
    for label in labels:
        # Extract grid size from label (e.g., "2x2" -> 2, "4x4" -> 4, "8x8" -> 8)
        try:
            # Try to extract first number from label like "2x2" -> 2
            num = int(label.split('x')[0])
            x_values.append(num)
        except:
            # Fallback: use index
            x_values.append(len(x_values) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot line graph
    plt.plot(x_values, maps, 'o-', linewidth=2.5, markersize=10, 
             color='steelblue', markerfacecolor='steelblue', 
             markeredgecolor='black', markeredgewidth=1.5)
    
    # Add value labels on data points
    for x, y, label in zip(x_values, maps, labels):
        plt.text(x, y, f'{y:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Grid Size (N×N)', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x_values, labels)
    plt.ylim([0, max(maps) * 1.2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ mAP comparison line graph saved to: {save_path}")
    plt.close()


def plot_summary_table(all_results, save_path):
    """
    Create a summary table showing mAP, Precision@10, and Recall@10 for all combinations.
    
    Args:
        all_results: list of dicts, each with 'descriptor', 'distance', 'mAP', 
                    'precision_at_10', 'recall_at_10'
        save_path: path to save the table
    """
    # Create table data
    table_data = []
    table_data.append(['Descriptor', 'Distance Metric', 'mAP', 'Precision@10', 'Recall@10'])
    
    for result in all_results:
        table_data.append([
            result['descriptor'],
            result['distance'],
            f"{result['mAP']:.4f}",
            f"{result['precision_at_10']:.4f}",
            f"{result['recall_at_10']:.4f}"
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(all_results) * 0.4 + 2)))
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.30, 0.25, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows (alternating colors)
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Spatial Grid RGB Histogram - Comprehensive Evaluation Summary\n'
                'All Descriptor-Distance Metric Combinations', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary table saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to run spatial grid comparison experiments.
    """
    print("="*80)
    print("SPATIAL GRID RGB HISTOGRAM EVALUATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
    
    # ============================================================================
    # Experiment 1: Compare distance metrics for spatialGridRGBhisto_8bins_4x4
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Comparing Distance Metrics (spatialGridRGBhisto_8bins_4x4)")
    print("="*80)
    
    descriptor_name = 'spatialGridRGBhisto_8bins_4x4'
    print(f"\n📊 Loading descriptors: {descriptor_name}")
    
    # Load descriptors
    all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
    print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
    
    experiment1_results = []
    
    for dist_name, (dist_func, dist_display) in DISTANCE_METRICS.items():
        print(f"\n📐 Evaluating with: {dist_display}")
        print("-" * 80)
        
        # Evaluate
        mean_precision_at_10, mean_recall_at_10, mean_average_precision, \
        all_precisions, all_recalls, confusion_matrix = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_function=dist_func,
            dataset_folder=DATASET_FOLDER,
            top_k=TOP_K,
            experiment_folder=None,  # Don't save individual plots
            distance_name=dist_name
        )
        
        print(f"\nResults for {dist_display}:")
        print(f"  Mean Precision@10: {mean_precision_at_10:.4f}")
        print(f"  Mean Recall@10: {mean_recall_at_10:.4f}")
        print(f"  mAP: {mean_average_precision:.4f}")
        
        # Store results
        experiment1_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': dist_display
        })
    
    # Plot PR curves comparison
    print("\n📈 Generating PR curves comparison for distance metrics...")
    pr_curves_path = os.path.join(EXPERIMENTS_FOLDER, 'spatial_grid_4x4_distance_metrics_pr_curves.png')
    plot_pr_curves_comparison(
        experiment1_results,
        title=f'PR Curves: Distance Metrics Comparison (Top {TOP_K}, spatialGridRGBhisto_8bins_4x4)',
        save_path=pr_curves_path
    )
    
    # ============================================================================
    # Experiment 2: Compare grid sizes using chi_square_distance
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Comparing Grid Sizes (Chi-square Distance)")
    print("="*80)
    
    experiment2_results = []
    map_comparison_data = []
    
    for descriptor_name in SPATIAL_GRID_DESCRIPTORS:
        print(f"\n📊 Evaluating: {descriptor_name}")
        print("-" * 80)
        
        # Load descriptors
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
        print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
        
        # Evaluate with chi_square_distance
        mean_precision_at_10, mean_recall_at_10, mean_average_precision, \
        all_precisions, all_recalls, confusion_matrix = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_function=chi_square_distance,
            dataset_folder=DATASET_FOLDER,
            top_k=TOP_K,
            experiment_folder=None,  # Don't save individual plots
            distance_name='chi_square'
        )
        
        print(f"\nResults for {descriptor_name}:")
        print(f"  Mean Precision@10: {mean_precision_at_10:.4f}")
        print(f"  Mean Recall@10: {mean_recall_at_10:.4f}")
        print(f"  mAP: {mean_average_precision:.4f}")
        
        # Store results
        experiment2_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': descriptor_name.replace('spatialGridRGBhisto_8bins_', '').replace('_', '×'),
            'descriptor_name': descriptor_name
        })
        
        map_comparison_data.append({
            'label': descriptor_name.replace('spatialGridRGBhisto_8bins_', '').replace('_', '×'),
            'mAP': mean_average_precision
        })
    
    # Plot mAP comparison line graph
    print("\n📊 Generating mAP comparison line graph...")
    map_chart_path = os.path.join(EXPERIMENTS_FOLDER, 'spatial_grid_grid_sizes_map_comparison.png')
    plot_map_comparison_line(
        map_comparison_data,
        title='Mean Average Precision (mAP) Comparison: Spatial Grid Sizes (Chi-square Distance)',
        save_path=map_chart_path
    )
    
    # ============================================================================
    # Experiment 3: Compare PR Curves for all grid sizes using chi_square_distance
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: Comparing PR Curves for All Grid Sizes (Chi-square Distance)")
    print("="*80)
    
    experiment3_results = []
    
    for descriptor_name in SPATIAL_GRID_DESCRIPTORS:
        print(f"\n📊 Evaluating: {descriptor_name}")
        print("-" * 80)
        
        # Load descriptors
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
        print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
        
        # Evaluate with chi_square_distance
        mean_precision_at_10, mean_recall_at_10, mean_average_precision, \
        all_precisions, all_recalls, confusion_matrix = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_function=chi_square_distance,
            dataset_folder=DATASET_FOLDER,
            top_k=TOP_K,
            experiment_folder=None,  # Don't save individual plots
            distance_name='chi_square'
        )
        
        print(f"\nResults for {descriptor_name}:")
        print(f"  Mean Precision@10: {mean_precision_at_10:.4f}")
        print(f"  Mean Recall@10: {mean_recall_at_10:.4f}")
        print(f"  mAP: {mean_average_precision:.4f}")
        
        # Store results
        experiment3_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': descriptor_name.replace('spatialGridRGBhisto_8bins_', '').replace('_', '×'),
            'descriptor_name': descriptor_name
        })
    
    # Plot PR curves comparison for all grid sizes
    print("\n📈 Generating PR curves comparison for all grid sizes...")
    pr_curves_path3 = os.path.join(EXPERIMENTS_FOLDER, 'spatial_grid_all_sizes_pr_curves.png')
    plot_pr_curves_comparison(
        experiment3_results,
        title=f'PR Curves: Spatial Grid Sizes Comparison (Top {TOP_K}, Chi-square Distance)',
        save_path=pr_curves_path3
    )
    
    # ============================================================================
    # Summary Table: Collect all results for comprehensive comparison
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE SUMMARY TABLE")
    print("="*80)
    
    all_summary_results = []
    
    # Add Experiment 1 results (Distance metrics with spatialGridRGBhisto_8bins_4x4)
    for result in experiment1_results:
        all_summary_results.append({
            'descriptor': 'spatialGridRGBhisto_8bins_4x4',
            'distance': result['label'],
            'mAP': result['mAP'],
            'precision_at_10': result['mean_precision_at_10'],
            'recall_at_10': result['mean_recall_at_10']
        })
    
    # Add Experiment 2 results (Grid sizes with Chi-square distance)
    for result in experiment2_results:
        all_summary_results.append({
            'descriptor': result['descriptor_name'],
            'distance': 'Chi-square Distance',
            'mAP': result['mAP'],
            'precision_at_10': result['mean_precision_at_10'],
            'recall_at_10': result['mean_recall_at_10']
        })
    
    # Generate summary table
    summary_table_path = os.path.join(EXPERIMENTS_FOLDER, 'spatial_grid_comprehensive_summary_table.png')
    plot_summary_table(all_summary_results, summary_table_path)
    
    # ============================================================================
    # Print Summary
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print("\n📐 Experiment 1 Results (Distance Metrics - spatialGridRGBhisto_8bins_4x4):")
    for result in experiment1_results:
        print(f"  {result['label']}: mAP = {result['mAP']:.4f}")
    
    print("\n🔲 Experiment 2 Results (Grid Sizes - Chi-square Distance):")
    for data in map_comparison_data:
        print(f"  {data['label']}: mAP = {data['mAP']:.4f}")
    
    print("\n📈 Experiment 3 Results (PR Curves - All Grid Sizes, Chi-square Distance):")
    for result in experiment3_results:
        print(f"  {result['label']}: mAP = {result['mAP']:.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print(f"\n📁 Results saved to: {EXPERIMENTS_FOLDER}")
    print("\nGenerated files:")
    print(f"  1. spatial_grid_4x4_distance_metrics_pr_curves.png")
    print(f"  2. spatial_grid_grid_sizes_map_comparison.png")
    print(f"  3. spatial_grid_all_sizes_pr_curves.png")
    print(f"  4. spatial_grid_comprehensive_summary_table.png")
    print("="*80)


if __name__ == "__main__":
    main()

