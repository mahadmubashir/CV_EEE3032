"""
PCA Dimensionality Reduction Visualization Script
EEE3032 - Computer Vision and Pattern Recognition

This script visualizes how PCA reduces dimensionality for:
1. globalRGBhisto_8bins
2. spatialGridRGBhisto_8bins_4x4
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
from cvpr_evaluate import load_descriptors
from cvpr_pca import load_pca_model, project_descriptors

DESCRIPTOR_FOLDER = './descriptors'
PCA_MODELS_FOLDER = './descriptors/pca_models'
OUTPUT_FOLDER = './experiments/pca_visualizations'

# Descriptors to visualize
DESCRIPTORS_TO_VISUALIZE = [
    {
        'name': 'globalRGBhisto_8bins',
        'display_name': 'Global RGB Histogram (8 bins)',
        'pca_configs': [
            {'components': 4, 'name': 'globalRGBhisto_8bins_PCA4'},
            {'components': 8, 'name': 'globalRGBhisto_8bins_PCA8'},
            {'components': 10, 'name': 'globalRGBhisto_8bins_PCA10'},
            {'components': 16, 'name': 'globalRGBhisto_8bins_PCA16'},
        ]
    },
    {
        'name': 'spatialGridRGBhisto_8bins_4x4',
        'display_name': 'Spatial Grid RGB Histogram (8 bins, 4×4 grid)',
        'pca_configs': [
            {'components': 64, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA64'},
            {'components': 96, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA96'},
            {'components': 128, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA128'},
            {'components': 256, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA256'},
        ]
    }
]


def visualize_pca_reduction(descriptor_info, save_path):
    """
    Create a comprehensive visualization of PCA dimensionality reduction.
    
    Args:
        descriptor_info: dict with 'name', 'display_name', and 'pca_configs'
        save_path: path to save the visualization
    """
    descriptor_name = descriptor_info['name']
    display_name = descriptor_info['display_name']
    pca_configs = descriptor_info['pca_configs']
    
    print(f"\n📊 Processing: {descriptor_name}")
    
    # Load original descriptors
    all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, descriptor_name)
    original_dim = all_features.shape[1]
    num_images = all_features.shape[0]
    
    print(f"  Original dimension: {original_dim}")
    print(f"  Number of images: {num_images}")
    
    # Load PCA model
    pca_model_path = os.path.join(PCA_MODELS_FOLDER, f"{descriptor_name}_pca.mat")
    if not os.path.exists(pca_model_path):
        print(f"  ⚠️  PCA model not found: {pca_model_path}")
        return
    
    pca_model = load_pca_model(pca_model_path)
    print(f"  ✅ PCA model loaded")
    
    # Get variance data
    cumulative_var_ratio = pca_model['cumulative_variance_ratio']
    reduced_dims = [config['components'] for config in pca_configs]
    labels = [f"PCA{config['components']}" for config in pca_configs]
    
    # Create figure with 2 subplots stacked vertically
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[1, 1.2])
    
    # ========================================================================
    # Plot 1: Variance Preserved for Each PCA Configuration
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    variance_preserved = []
    for config in pca_configs:
        n_comp = config['components']
        if n_comp <= len(cumulative_var_ratio):
            var = cumulative_var_ratio[n_comp - 1] * 100  # Convert to percentage
        else:
            var = 100.0  # All variance preserved if using all components
        variance_preserved.append(var)
    
    bars = ax1.bar(range(len(pca_configs)), variance_preserved,
                   color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, var, label in zip(bars, variance_preserved, labels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.2f}%\n({label})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('PCA Configuration', fontsize=12)
    ax1.set_ylabel('Variance Preserved (%)', fontsize=12)
    ax1.set_title('Variance Preserved by Each PCA Configuration', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(pca_configs)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=95, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='95% Threshold')
    ax1.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='90% Threshold')
    ax1.legend(fontsize=10)
    
    # ========================================================================
    # Plot 2: Dimension Comparison Table
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Configuration', 'Original Dim', 'Reduced Dim', 'Compression', 'Variance Preserved'])
    
    for config in pca_configs:
        n_comp = config['components']
        compression = original_dim / n_comp
        if n_comp <= len(cumulative_var_ratio):
            var = cumulative_var_ratio[n_comp - 1] * 100
        else:
            var = 100.0
        table_data.append([
            config['name'],
            f'{original_dim}',
            f'{n_comp}',
            f'{compression:.2f}×',
            f'{var:.2f}%'
        ])
    
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.4, 0.12, 0.12, 0.12, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax2.set_title('PCA Dimensionality Reduction Summary', fontsize=13, fontweight='bold', pad=15)
    
    plt.suptitle(f'PCA Visualization: {display_name}', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Visualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to generate PCA visualizations.
    """
    print("="*80)
    print("PCA DIMENSIONALITY REDUCTION VISUALIZATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Generate visualization for each descriptor
    for descriptor_info in DESCRIPTORS_TO_VISUALIZE:
        descriptor_name = descriptor_info['name']
        save_filename = f"pca_visualization_{descriptor_name}.png"
        save_path = os.path.join(OUTPUT_FOLDER, save_filename)
        
        visualize_pca_reduction(descriptor_info, save_path)
    
    print("\n" + "="*80)
    print("✅ ALL PCA VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\n📁 Generated files in {OUTPUT_FOLDER}:")
    for descriptor_info in DESCRIPTORS_TO_VISUALIZE:
        descriptor_name = descriptor_info['name']
        save_filename = f"pca_visualization_{descriptor_name}.png"
        print(f"   • {save_filename}")
    print("="*80)


if __name__ == "__main__":
    main()

