"""
GCH Feature Descriptor Comparison and Evaluation Script
EEE3032 - Computer Vision and Pattern Recognition

This script compares and evaluates Global Color Histogram (GCH) feature descriptors
with different quantization levels and distance metrics.
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
    chi_square_distance
)

DESCRIPTOR_FOLDER = './descriptors'
DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
EXPERIMENTS_FOLDER = './experiments/gch_comparison'
TOP_K = 10

# GCH descriptors to compare
GCH_DESCRIPTORS = [
    'globalRGBhisto_4bins',
    'globalRGBhisto_8bins',
    'globalRGBhisto_16bins',
    'globalRGBhisto_32bins'
]

# Distance metrics to compare
DISTANCE_METRICS = {
    'euclidean': (euclidean_distance, 'Euclidean Distance'),
    'manhattan': (manhattan_distance, 'Manhattan Distance'),
    'chebyshev': (chebyshev_distance, 'Chebyshev Distance'),
    'minkowski': (lambda F1, F2: minkowski_distance(F1, F2, p=3), 'Minkowski Distance (p=3)'),
    'chi_square': (chi_square_distance, 'Chi-square Distance')
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
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
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


def plot_map_comparison(map_results, title, save_path):
    """
    Plot line graph comparing mAP across different descriptors.
    
    Args:
        map_results: list of dicts with 'label' and 'mAP'
        title: plot title
        save_path: path to save the plot
    """
    labels = [r['label'] for r in map_results]
    maps = [r['mAP'] for r in map_results]
    
    # Extract numeric values for x-axis (bins: 4, 8, 16, 32)
    x_values = []
    for label in labels:
        # Extract number from label (e.g., "4 bins" -> 4)
        try:
            num = int(label.split()[0])
            x_values.append(num)
        except:
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
    
    plt.xlabel('GCH Descriptor (Quantization Level - Number of Bins)', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x_values, labels)
    plt.ylim([0, max(maps) * 1.2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ mAP comparison chart saved to: {save_path}")
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
    
    ax.set_title('GCH Feature Descriptors - Comprehensive Evaluation Summary\n'
                'All Descriptor-Distance Metric Combinations', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary table saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to run GCH comparison experiments.
    """
    print("="*60)
    print("GCH FEATURE DESCRIPTOR COMPARISON")
    print("="*60)
    
    # Create output directory
    os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
    
    # ============================================================================
    # Experiment 1: Compare GCH descriptors with different quantization levels
    # Using Chi-square distance
    # ============================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Comparing GCH Quantization Levels (Chi-square Distance)")
    print("="*60)
    
    experiment1_results = []
    map_comparison_data = []
    
    for descriptor_name in GCH_DESCRIPTORS:
        print(f"\n📊 Evaluating: {descriptor_name}")
        print("-" * 60)
        
        # Load descriptors
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
        print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
        
        # Evaluate
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
        experiment1_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': descriptor_name.replace('globalRGBhisto_', '').replace('bins', ' bins')
        })
        
        map_comparison_data.append({
            'label': descriptor_name.replace('globalRGBhisto_', '').replace('bins', ' bins'),
            'mAP': mean_average_precision
        })
    
    # Plot PR curves comparison
    print("\n📈 Generating PR curves comparison...")
    pr_curves_path = os.path.join(EXPERIMENTS_FOLDER, 'gch_quantization_comparison_pr_curves.png')
    plot_pr_curves_comparison(
        experiment1_results,
        title=f'PR Curves: GCH Quantization Levels Comparison (Top {TOP_K}, Chi-square Distance)',
        save_path=pr_curves_path
    )
    
    # Plot mAP comparison
    print("\n📊 Generating mAP comparison chart...")
    map_chart_path = os.path.join(EXPERIMENTS_FOLDER, 'gch_quantization_map_comparison.png')
    plot_map_comparison(
        map_comparison_data,
        title='Mean Average Precision (mAP) Comparison: GCH Quantization Levels',
        save_path=map_chart_path
    )
    
    # ============================================================================
    # Experiment 2: Compare different distance metrics with globalRGBhisto_8bins
    # ============================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Comparing Distance Metrics (globalRGBhisto_8bins)")
    print("="*60)
    
    descriptor_name = 'globalRGBhisto_8bins'
    print(f"\n📊 Loading descriptors: {descriptor_name}")
    
    # Load descriptors
    all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
    print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
    
    experiment2_results = []
    
    for dist_name, (dist_func, dist_display) in DISTANCE_METRICS.items():
        print(f"\n📐 Evaluating with: {dist_display}")
        print("-" * 60)
        
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
        experiment2_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': dist_display
        })
    
    # Plot PR curves comparison
    print("\n📈 Generating PR curves comparison for distance metrics...")
    pr_curves_path2 = os.path.join(EXPERIMENTS_FOLDER, 'gch_distance_metrics_comparison_pr_curves.png')
    plot_pr_curves_comparison(
        experiment2_results,
        title=f'PR Curves: Distance Metrics Comparison (Top {TOP_K}, globalRGBhisto_8bins)',
        save_path=pr_curves_path2
    )
    
    # ============================================================================
    # Summary Table: Collect all results for comprehensive comparison
    # ============================================================================
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE SUMMARY TABLE")
    print("="*60)
    
    all_summary_results = []
    
    # Add Experiment 1 results (GCH quantization levels with Chi-square distance)
    for result in experiment1_results:
        descriptor_label = result['label']
        all_summary_results.append({
            'descriptor': f'globalRGBhisto_{descriptor_label.replace(" bins", "bins")}',
            'distance': 'Chi-square Distance',
            'mAP': result['mAP'],
            'precision_at_10': result['mean_precision_at_10'],
            'recall_at_10': result['mean_recall_at_10']
        })
    
    # Add Experiment 2 results (Distance metrics with globalRGBhisto_8bins)
    for result in experiment2_results:
        all_summary_results.append({
            'descriptor': 'globalRGBhisto_8bins',
            'distance': result['label'],
            'mAP': result['mAP'],
            'precision_at_10': result['mean_precision_at_10'],
            'recall_at_10': result['mean_recall_at_10']
        })
    
    # Generate summary table
    summary_table_path = os.path.join(EXPERIMENTS_FOLDER, 'gch_comprehensive_summary_table.png')
    plot_summary_table(all_summary_results, summary_table_path)
    
    # ============================================================================
    # Print Summary
    # ============================================================================
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print("\n📊 Experiment 1 Results (GCH Quantization Levels - Chi-square Distance):")
    for data in map_comparison_data:
        print(f"  {data['label']}: mAP = {data['mAP']:.4f}")
    
    print("\n📐 Experiment 2 Results (Distance Metrics - globalRGBhisto_8bins):")
    for result in experiment2_results:
        print(f"  {result['label']}: mAP = {result['mAP']:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS GENERATED")
    print("="*60)
    print(f"\n📁 Results saved to: {EXPERIMENTS_FOLDER}")
    print("\nGenerated files:")
    print(f"  1. gch_quantization_comparison_pr_curves.png")
    print(f"  2. gch_quantization_map_comparison.png")
    print(f"  3. gch_distance_metrics_comparison_pr_curves.png")
    print(f"  4. gch_comprehensive_summary_table.png")
    print("="*60)


if __name__ == "__main__":
    main()

