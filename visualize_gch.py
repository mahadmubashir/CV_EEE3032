"""
Visualization Script for Global Color Histogram (GCH) Feature Descriptors
EEE3032 - Computer Vision and Pattern Recognition

This script generates visualizations showing:
1. Query image
2. GCH feature descriptor calculation for different bin counts (4, 8, 16)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from extractRandom import extractRGBHistogram
from random import randint

DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
OUTPUT_FOLDER = './experiments/gch_visualizations'

# Bin configurations to visualize
BIN_CONFIGS = [4, 8, 16]

def visualize_gch_for_bins(img, num_bins, output_path):
    """
    Visualize Global Color Histogram for a given number of bins.
    
    Args:
        img: Input image (normalized 0-1)
        num_bins: Number of bins per channel
        output_path: Path to save the visualization
    """
    # Compute GCH descriptor
    descriptor = extractRGBHistogram(img, num_bins=num_bins)
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Original Image
    ax1 = plt.subplot(2, 3, 1)
    img_display = (img * 255).astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    ax1.imshow(img_display)
    ax1.set_title(f'Query Image\nSize: {img_width}×{img_height} pixels', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: RGB Channels Separated
    ax2 = plt.subplot(2, 3, 2)
    r_channel = img[:, :, 2]  # Red channel (BGR format)
    g_channel = img[:, :, 1]  # Green channel
    b_channel = img[:, :, 0]  # Blue channel
    
    rgb_channels = np.zeros((img_height, img_width * 3, 3))
    rgb_channels[:, :img_width, 0] = r_channel  # Red
    rgb_channels[:, img_width:2*img_width, 1] = g_channel  # Green
    rgb_channels[:, 2*img_width:, 2] = b_channel  # Blue
    
    ax2.imshow(rgb_channels)
    ax2.set_title('RGB Channels Separated\n(R | G | B)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Plot 3: Histogram visualization (3D bar chart style)
    ax3 = plt.subplot(2, 3, 3)
    descriptor_flat = descriptor.flatten()
    n_bins_total = len(descriptor_flat)
    x_positions = np.arange(n_bins_total)
    
    bars = ax3.bar(x_positions, descriptor_flat, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Histogram Bin Index', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'GCH Histogram ({num_bins} bins/channel)\nTotal dimension: {n_bins_total} (3×{num_bins})', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add text annotation showing descriptor dimension
    ax3.text(0.02, 0.98, f'Descriptor dimension: {n_bins_total}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4-6: Individual RGB channel histograms
    # GCH structure: R bins (0 to num_bins-1), G bins (num_bins to 2*num_bins-1), B bins (2*num_bins to 3*num_bins-1)
    bins_per_channel = num_bins
    hist_r = descriptor_flat[0:bins_per_channel]
    hist_g = descriptor_flat[bins_per_channel:2*bins_per_channel]
    hist_b = descriptor_flat[2*bins_per_channel:3*bins_per_channel]
    
    # Plot 4: Individual RGB channel histograms side by side
    ax4 = plt.subplot(2, 3, 4)
    x_r = np.arange(bins_per_channel)
    x_g = np.arange(bins_per_channel) + bins_per_channel
    x_b = np.arange(bins_per_channel) + 2*bins_per_channel
    
    width = 0.8
    ax4.bar(x_r, hist_r, width, label='Red', color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.bar(x_g, hist_g, width, label='Green', color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.bar(x_b, hist_b, width, label='Blue', color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Bin Index', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title(f'RGB Channel Histograms\n({num_bins} bins per channel)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add vertical lines to separate channels
    ax4.axvline(x=bins_per_channel - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=2*bins_per_channel - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 5: Cumulative histogram
    ax5 = plt.subplot(2, 3, 5)
    cumulative = np.cumsum(descriptor_flat)
    cumulative_normalized = cumulative / cumulative[-1]  # Normalize to 0-1
    ax5.plot(x_positions, cumulative_normalized, 'r-', linewidth=2)
    ax5.fill_between(x_positions, cumulative_normalized, alpha=0.3, color='red')
    ax5.set_xlabel('Histogram Bin Index', fontsize=11)
    ax5.set_ylabel('Cumulative Frequency (Normalized)', fontsize=11)
    ax5.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # Plot 6: Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    GCH Descriptor Statistics
    {'='*40}
    
    Bin Configuration:
    • Bins per channel: {num_bins}
    • Total bins: {n_bins_total}
    • Descriptor shape: {descriptor.shape}
    
    Histogram Statistics:
    • Min value: {descriptor_flat.min():.6f}
    • Max value: {descriptor_flat.max():.6f}
    • Mean value: {descriptor_flat.mean():.6f}
    • Std deviation: {descriptor_flat.std():.6f}
    • Sum: {descriptor_flat.sum():.6f}
    
    Non-zero bins: {np.count_nonzero(descriptor_flat)}
    Zero bins: {np.count_nonzero(descriptor_flat == 0)}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Global Color Histogram (GCH) Visualization - {num_bins} Bins per Channel', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()


def main():
    """
    Main function to generate GCH visualizations for different bin counts.
    """
    print("="*60)
    print("GCH FEATURE DESCRIPTOR VISUALIZATION")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get a sample image from the dataset
    image_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.bmp')]
    if not image_files:
        print(f"❌ No images found in {DATASET_FOLDER}")
        return
    
    # Use a random image as query
    query_filename = image_files[randint(0, len(image_files) - 1)]
    query_path = os.path.join(DATASET_FOLDER, query_filename)
    
    print(f"\n📷 Loading query image: {query_filename}")
    img = cv2.imread(query_path).astype(np.float64) / 255.0  # Normalize to 0-1
    img_height, img_width = img.shape[:2]
    print(f"   Image size: {img_width}×{img_height} pixels")
    print(f"   Image shape: {img.shape}")
    
    # Generate visualizations for each bin configuration
    print(f"\n🎨 Generating GCH visualizations...")
    for num_bins in BIN_CONFIGS:
        print(f"\n   Processing {num_bins} bins...")
        
        # Compute descriptor to get dimension info
        descriptor = extractRGBHistogram(img, num_bins=num_bins)
        descriptor_dim = descriptor.size
        print(f"      Descriptor dimension: {descriptor_dim}")
        
        # Generate visualization
        output_path = os.path.join(OUTPUT_FOLDER, f'gch_{num_bins}bins_{query_filename.replace(".bmp", "")}.png')
        visualize_gch_for_bins(img, num_bins, output_path)
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\n📁 All visualizations saved to: {OUTPUT_FOLDER}")
    print("\nGenerated files:")
    for num_bins in BIN_CONFIGS:
        filename = f'gch_{num_bins}bins_{query_filename.replace(".bmp", "")}.png'
        print(f"   - {filename}")
    print("="*60)


if __name__ == "__main__":
    main()

