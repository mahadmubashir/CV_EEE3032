"""
PCA Descriptor Evaluation Script
EEE3032 - Computer Vision and Pattern Recognition

This script evaluates PCA-reduced descriptors using Mahalanobis distance.
It compares different PCA configurations for:
1. globalRGBhisto_8bins (PCA4, PCA8, PCA10, PCA16)
2. spatialGridRGBhisto_8bins_4x4 (PCA64, PCA96, PCA128, PCA256)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cvpr_evaluate import (
    load_descriptors,
    evaluate_all_queries,
    plot_precision_recall_curve
)
from cvpr_compare import mahalanobis_distance

DESCRIPTOR_FOLDER = './descriptors'
DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
EXPERIMENTS_FOLDER = './experiments/pca_evaluation'
TOP_K = 10

# PCA descriptors for globalRGBhisto_8bins
GCH_PCA_DESCRIPTORS = [
    'globalRGBhisto_8bins_PCA10',
    'globalRGBhisto_8bins_PCA16',
    'globalRGBhisto_8bins_PCA18',
    'globalRGBhisto_8bins_PCA20'
]

# PCA descriptors for spatialGridRGBhisto_8bins_4x4
SPATIAL_PCA_DESCRIPTORS = [
    'spatialGridRGBhisto_8bins_4x4_PCA12',
    'spatialGridRGBhisto_8bins_4x4_PCA16',
    'spatialGridRGBhisto_8bins_4x4_PCA24',
    'spatialGridRGBhisto_8bins_4x4_PCA32'
]


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
    Plot line graph comparing mAP across different PCA configurations.
    
    Args:
        map_results: list of dicts with 'label' and 'mAP'
        title: plot title
        save_path: path to save the plot
    """
    labels = [r['label'] for r in map_results]
    maps = [r['mAP'] for r in map_results]
    
    # Extract numeric values for x-axis (PCA components: 4, 8, 10, 16 or 64, 96, 128, 256)
    x_values = []
    for label in labels:
        # Extract number from label (e.g., "PCA4" -> 4, "PCA64" -> 64)
        try:
            # Try to extract number after "PCA"
            num_str = label.replace('PCA', '')
            num = int(num_str)
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
    
    plt.xlabel('Number of PCA Components', fontsize=12)
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
    
    ax.set_title('PCA Descriptors - Comprehensive Evaluation Summary\n'
                'All Descriptor-Distance Metric Combinations (Mahalanobis Distance)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary table saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to run PCA descriptor evaluation experiments.
    """
    print("="*80)
    print("PCA DESCRIPTOR EVALUATION (MAHALANOBIS DISTANCE)")
    print("="*80)
    
    # Create output directory
    os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
    
    # ============================================================================
    # Experiment 1: Evaluate globalRGBhisto_8bins PCA descriptors
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Evaluating globalRGBhisto_8bins PCA Descriptors")
    print("="*80)
    
    gch_pca_results = []
    gch_map_data = []
    
    for descriptor_name in GCH_PCA_DESCRIPTORS:
        print(f"\n📊 Evaluating: {descriptor_name}")
        print("-" * 80)
        
        # Load descriptors
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
        print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
        
        # Evaluate with Mahalanobis distance
        mean_precision_at_10, mean_recall_at_10, mean_average_precision, \
        all_precisions, all_recalls, confusion_matrix = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_function=mahalanobis_distance,
            dataset_folder=DATASET_FOLDER,
            top_k=TOP_K,
            experiment_folder=None,  # Don't save individual plots
            distance_name='mahalanobis'
        )
        
        print(f"\nResults for {descriptor_name}:")
        print(f"  Mean Precision@10: {mean_precision_at_10:.4f}")
        print(f"  Mean Recall@10: {mean_recall_at_10:.4f}")
        print(f"  mAP: {mean_average_precision:.4f}")
        
        # Extract PCA component number for label
        pca_comp = descriptor_name.replace('globalRGBhisto_8bins_PCA', '')
        label = f"PCA{pca_comp}"
        
        # Store results
        gch_pca_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': label,
            'descriptor_name': descriptor_name
        })
        
        gch_map_data.append({
            'label': label,
            'mAP': mean_average_precision
        })
    
    # Graph 1: PR Curves for globalRGBhisto_8bins PCA descriptors
    print("\n📈 Generating Graph 1: PR Curves for globalRGBhisto_8bins PCA...")
    pr_curves_path1 = os.path.join(EXPERIMENTS_FOLDER, 'gch_pca_pr_curves.png')
    plot_pr_curves_comparison(
        gch_pca_results,
        title=f'PR Curves: globalRGBhisto_8bins PCA Descriptors (Top {TOP_K}, Mahalanobis Distance)',
        save_path=pr_curves_path1
    )
    
    # Graph 2: Line graph comparing mAPs for globalRGBhisto_8bins PCA
    print("\n📊 Generating Graph 2: mAP Comparison for globalRGBhisto_8bins PCA...")
    map_chart_path1 = os.path.join(EXPERIMENTS_FOLDER, 'gch_pca_map_comparison.png')
    plot_map_comparison_line(
        gch_map_data,
        title='Mean Average Precision (mAP) Comparison: globalRGBhisto_8bins PCA Components\n(Mahalanobis Distance)',
        save_path=map_chart_path1
    )
    
    # ============================================================================
    # Experiment 2: Evaluate spatialGridRGBhisto_8bins_4x4 PCA descriptors
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Evaluating spatialGridRGBhisto_8bins_4x4 PCA Descriptors")
    print("="*80)
    
    spatial_pca_results = []
    spatial_map_data = []
    
    for descriptor_name in SPATIAL_PCA_DESCRIPTORS:
        print(f"\n📊 Evaluating: {descriptor_name}")
        print("-" * 80)
        
        # Load descriptors
        all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
        print(f"Loaded {len(image_paths)} descriptors, dimension: {all_features.shape[1]}")
        
        # Evaluate with Mahalanobis distance
        mean_precision_at_10, mean_recall_at_10, mean_average_precision, \
        all_precisions, all_recalls, confusion_matrix = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_function=mahalanobis_distance,
            dataset_folder=DATASET_FOLDER,
            top_k=TOP_K,
            experiment_folder=None,  # Don't save individual plots
            distance_name='mahalanobis'
        )
        
        print(f"\nResults for {descriptor_name}:")
        print(f"  Mean Precision@10: {mean_precision_at_10:.4f}")
        print(f"  Mean Recall@10: {mean_recall_at_10:.4f}")
        print(f"  mAP: {mean_average_precision:.4f}")
        
        # Extract PCA component number for label
        pca_comp = descriptor_name.replace('spatialGridRGBhisto_8bins_4x4_PCA', '')
        label = f"PCA{pca_comp}"
        
        # Store results
        spatial_pca_results.append({
            'precisions': all_precisions,
            'recalls': all_recalls,
            'mAP': mean_average_precision,
            'mean_precision_at_10': mean_precision_at_10,
            'mean_recall_at_10': mean_recall_at_10,
            'label': label,
            'descriptor_name': descriptor_name
        })
        
        spatial_map_data.append({
            'label': label,
            'mAP': mean_average_precision
        })
    
    # Graph 3: PR Curves for spatialGridRGBhisto_8bins_4x4 PCA descriptors
    print("\n📈 Generating Graph 3: PR Curves for spatialGridRGBhisto_8bins_4x4 PCA...")
    pr_curves_path2 = os.path.join(EXPERIMENTS_FOLDER, 'spatial_pca_pr_curves.png')
    plot_pr_curves_comparison(
        spatial_pca_results,
        title=f'PR Curves: spatialGridRGBhisto_8bins_4x4 PCA Descriptors (Top {TOP_K}, Mahalanobis Distance)',
        save_path=pr_curves_path2
    )
    
    # Graph 4: Line graph comparing mAPs for spatialGridRGBhisto_8bins_4x4 PCA
    print("\n📊 Generating Graph 4: mAP Comparison for spatialGridRGBhisto_8bins_4x4 PCA...")
    map_chart_path2 = os.path.join(EXPERIMENTS_FOLDER, 'spatial_pca_map_comparison.png')
    plot_map_comparison_line(
        spatial_map_data,
        title='Mean Average Precision (mAP) Comparison: spatialGridRGBhisto_8bins_4x4 PCA Components\n(Mahalanobis Distance)',
        save_path=map_chart_path2
    )
    
    # ============================================================================
    # Summary Table: Collect all results for comprehensive comparison
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE SUMMARY TABLE")
    print("="*80)
    
    all_summary_results = []
    
    # Add Experiment 1 results (globalRGBhisto_8bins PCA descriptors with Mahalanobis distance)
    for result in gch_pca_results:
        all_summary_results.append({
            'descriptor': result['descriptor_name'],
            'distance': 'Mahalanobis Distance',
            'mAP': result['mAP'],
            'precision_at_10': result['mean_precision_at_10'],
            'recall_at_10': result['mean_recall_at_10']
        })
    
    # Add Experiment 2 results (spatialGridRGBhisto_8bins_4x4 PCA descriptors with Mahalanobis distance)
    for result in spatial_pca_results:
        all_summary_results.append({
            'descriptor': result['descriptor_name'],
            'distance': 'Mahalanobis Distance',
            'mAP': result['mAP'],
            'precision_at_10': result['mean_precision_at_10'],
            'recall_at_10': result['mean_recall_at_10']
        })
    
    # Generate summary table
    summary_table_path = os.path.join(EXPERIMENTS_FOLDER, 'pca_comprehensive_summary_table.png')
    plot_summary_table(all_summary_results, summary_table_path)
    
    # ============================================================================
    # Print Summary
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print("\n📊 globalRGBhisto_8bins PCA Results (Mahalanobis Distance):")
    for data in gch_map_data:
        print(f"  {data['label']}: mAP = {data['mAP']:.4f}")
    
    print("\n📊 spatialGridRGBhisto_8bins_4x4 PCA Results (Mahalanobis Distance):")
    for data in spatial_map_data:
        print(f"  {data['label']}: mAP = {data['mAP']:.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print(f"\n📁 Results saved to: {EXPERIMENTS_FOLDER}")
    print("\nGenerated files:")
    print(f"  1. gch_pca_pr_curves.png - PR Curves for globalRGBhisto_8bins PCA")
    print(f"  2. gch_pca_map_comparison.png - mAP comparison for globalRGBhisto_8bins PCA")
    print(f"  3. spatial_pca_pr_curves.png - PR Curves for spatialGridRGBhisto_8bins_4x4 PCA")
    print(f"  4. spatial_pca_map_comparison.png - mAP comparison for spatialGridRGBhisto_8bins_4x4 PCA")
    print(f"  5. pca_comprehensive_summary_table.png - Comprehensive summary table")
    print("="*80)


if __name__ == "__main__":
    main()

