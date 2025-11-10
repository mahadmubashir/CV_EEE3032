"""
Spatial Grid RGB Histogram Visualization Script
EEE3032 - Computer Vision and Pattern Recognition

This script visualizes how spatial grid descriptors work by:
1. Selecting a random query image
2. Dividing it into grid cells (2x2, 4x4, 8x8)
3. Showing each grid cell and its computed GCH histogram
4. Creating separate visualizations for each grid size
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from extractRandom import extractSpatialGrid, extractRGBHistogram

DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
OUTPUT_FOLDER = './experiments/spatial_grid_visualizations'

# Grid configurations to visualize
GRID_CONFIGS = [
    {'rows': 2, 'cols': 2, 'name': '2x2'},
    {'rows': 4, 'cols': 4, 'name': '4x4'},
    {'rows': 8, 'cols': 8, 'name': '8x8'}
]

NUM_BINS = 8  # Using 8 bins as per user's requirement


def visualize_spatial_grid(image_path, grid_rows, grid_cols, num_bins, save_path):
    """
    Create a comprehensive visualization of spatial grid descriptor computation.
    
    Args:
        image_path: Path to the query image
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        num_bins: Number of bins per color channel
        save_path: Path to save the visualization
    """
    # Load and normalize image
    img_original = cv2.imread(image_path)
    img_normalized = img_original.astype(np.float64) / 255.0
    img_display = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    height, width = img_display.shape[:2]
    total_cells = grid_rows * grid_cols
    
    # Compute cell dimensions (ensure integers)
    cell_height = int(height // grid_rows)
    cell_width = int(width // grid_cols)
    
    # Create figure with subplots
    # Layout: Original image + grid overlay at top
    # Then grid cells with their histograms below
    # For each cell, we show: cell image + its RGB histogram
    fig = plt.figure(figsize=(20, 8 + grid_rows * 3))
    gs = gridspec.GridSpec(grid_rows * 2 + 2, grid_cols + 1, figure=fig, 
                          hspace=0.4, wspace=0.3, height_ratios=[1, 1] + [1, 0.8] * grid_rows)
    
    # ========================================================================
    # Plot 1: Original Image with Grid Overlay
    # ========================================================================
    ax_original = fig.add_subplot(gs[0, :grid_cols])
    ax_original.imshow(img_display)
    ax_original.set_title(f'Query Image: {os.path.basename(image_path)}\n'
                         f'Size: {width}×{height} pixels | Grid: {grid_rows}×{grid_cols} = {total_cells} cells',
                         fontsize=14, fontweight='bold')
    ax_original.axis('off')
    
    # Draw grid lines on the image
    img_with_grid = img_display.copy()
    for i in range(grid_rows + 1):
        y = int(i * cell_height)
        cv2.line(img_with_grid, (0, y), (width, y), (255, 0, 0), 2)
    for j in range(grid_cols + 1):
        x = int(j * cell_width)
        cv2.line(img_with_grid, (x, 0), (x, height), (255, 0, 0), 2)
    
    ax_grid = fig.add_subplot(gs[1, :grid_cols])
    ax_grid.imshow(img_with_grid)
    ax_grid.set_title(f'Image with {grid_rows}×{grid_cols} Grid Overlay',
                     fontsize=12, fontweight='bold')
    ax_grid.axis('off')
    
    # ========================================================================
    # Extract and visualize each grid cell
    # ========================================================================
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    cell_histograms = []
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Compute cell boundaries (ensure integers)
            y_start = int(row * cell_height)
            y_end = int((row + 1) * cell_height) if row < grid_rows - 1 else int(height)
            x_start = int(col * cell_width)
            x_end = int((col + 1) * cell_width) if col < grid_cols - 1 else int(width)
            
            # Ensure valid boundaries (at least 1 pixel)
            y_start = max(0, min(y_start, height - 1))
            y_end = max(y_start + 1, min(y_end, height))
            x_start = max(0, min(x_start, width - 1))
            x_end = max(x_start + 1, min(x_end, width))
            
            # Extract grid cell
            cell = img_uint8[y_start:y_end, x_start:x_end, :]
            
            # Check if cell is valid, create placeholder if empty
            if cell.size == 0 or len(cell.shape) < 3 or cell.shape[0] == 0 or cell.shape[1] == 0:
                # Create a small placeholder cell if empty
                cell = np.zeros((1, 1, 3), dtype=np.uint8)
                print(f"Warning: Empty cell at ({row}, {col}), using placeholder")
            
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            
            # Compute histogram for each color channel
            hist_r = np.histogram(cell[:, :, 2], bins=num_bins, range=(0, 256))[0]  # Red
            hist_g = np.histogram(cell[:, :, 1], bins=num_bins, range=(0, 256))[0]  # Green
            hist_b = np.histogram(cell[:, :, 0], bins=num_bins, range=(0, 256))[0]  # Blue
            
            # Normalize histograms
            hist_r = hist_r.astype(np.float64) / max(np.sum(hist_r), 1)
            hist_g = hist_g.astype(np.float64) / max(np.sum(hist_g), 1)
            hist_b = hist_b.astype(np.float64) / max(np.sum(hist_b), 1)
            
            cell_histograms.append({'r': hist_r, 'g': hist_g, 'b': hist_b})
            
            # Plot cell image (row 0 for images, row 1 for histograms)
            ax_cell = fig.add_subplot(gs[row * 2 + 2, col])
            ax_cell.imshow(cell_rgb)
            ax_cell.set_title(f'Cell ({row},{col})\n{cell_rgb.shape[1]}×{cell_rgb.shape[0]}px',
                             fontsize=9, fontweight='bold')
            ax_cell.axis('off')
            
            # Plot histogram for this cell (below the cell image)
            ax_hist = fig.add_subplot(gs[row * 2 + 3, col])
            bin_centers = np.arange(num_bins)
            width = 0.25
            ax_hist.bar(bin_centers - width, hist_r, width, label='R', color='red', alpha=0.7)
            ax_hist.bar(bin_centers, hist_g, width, label='G', color='green', alpha=0.7)
            ax_hist.bar(bin_centers + width, hist_b, width, label='B', color='blue', alpha=0.7)
            ax_hist.set_xlabel('Bin', fontsize=8)
            ax_hist.set_ylabel('Freq', fontsize=8)
            ax_hist.set_title(f'GCH ({num_bins} bins)', fontsize=8, fontweight='bold')
            ax_hist.tick_params(labelsize=7)
            ax_hist.grid(True, alpha=0.3)
            if row == 0 and col == 0:  # Show legend only once
                ax_hist.legend(fontsize=7, loc='upper right')
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    ax_summary = fig.add_subplot(gs[1, grid_cols])
    ax_summary.axis('off')
    
    # Compute full descriptor
    descriptor = extractSpatialGrid(img_normalized, grid_rows=grid_rows, 
                                   grid_cols=grid_cols, num_bins=num_bins)
    
    summary_text = f"""
SPATIAL GRID DESCRIPTOR SUMMARY

Grid Configuration:
  • Grid Size: {grid_rows}×{grid_cols} = {total_cells} cells
  • Bins per channel: {num_bins}
  • Channels per cell: 3 (R, G, B)
  • Numbers per cell: {num_bins} × 3 = {num_bins * 3}
  
Descriptor Dimensions:
  • Total descriptor size: {descriptor.shape[1]} numbers
  • Calculation: {total_cells} cells × {num_bins * 3} numbers = {descriptor.shape[1]}
  
Cell Dimensions:
  • Image size: {width}×{height} pixels
  • Cell size: ~{cell_width}×{cell_height} pixels
  
Descriptor Structure:
  [Cell(0,0)_R | Cell(0,0)_G | Cell(0,0)_B |
   Cell(0,1)_R | Cell(0,1)_G | Cell(0,1)_B |
   ... | Cell({grid_rows-1},{grid_cols-1})_R | Cell({grid_rows-1},{grid_cols-1})_G | Cell({grid_rows-1},{grid_cols-1})_B]
  
Descriptor Statistics:
  • Min value: {descriptor.min():.4f}
  • Max value: {descriptor.max():.4f}
  • Mean value: {descriptor.mean():.4f}
  • Std deviation: {descriptor.std():.4f}
    """
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Spatial Grid RGB Histogram Visualization: {grid_rows}×{grid_cols} Grid',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Spatial grid visualization ({grid_rows}×{grid_cols}) saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to generate spatial grid visualizations.
    """
    print("="*80)
    print("SPATIAL GRID RGB HISTOGRAM VISUALIZATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.bmp')]
    
    if not image_files:
        print(f"❌ No images found in {DATASET_FOLDER}")
        return
    
    # Select a random query image
    np.random.seed(42)  # For reproducibility
    query_filename = np.random.choice(image_files)
    query_path = os.path.join(DATASET_FOLDER, query_filename)
    
    print(f"\n📷 Selected query image: {query_filename}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"\n🔲 Generating visualizations for {len(GRID_CONFIGS)} grid configurations...")
    
    # Generate visualization for each grid configuration
    for config in GRID_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Processing {config['name']} grid ({config['rows']}×{config['cols']})...")
        print("="*80)
        
        save_filename = f"spatial_grid_{config['name']}_{os.path.basename(query_filename).replace('.bmp', '.png')}"
        save_path = os.path.join(OUTPUT_FOLDER, save_filename)
        
        visualize_spatial_grid(
            image_path=query_path,
            grid_rows=config['rows'],
            grid_cols=config['cols'],
            num_bins=NUM_BINS,
            save_path=save_path
        )
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\n📁 Generated files in {OUTPUT_FOLDER}:")
    for config in GRID_CONFIGS:
        save_filename = f"spatial_grid_{config['name']}_{os.path.basename(query_filename).replace('.bmp', '.png')}"
        print(f"   • {save_filename}")
    print("="*80)


if __name__ == "__main__":
    main()

