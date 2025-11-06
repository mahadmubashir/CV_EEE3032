"""
Step-by-Step Explanation: Spatial Grid RGB Histogram
This script demonstrates how spatial gridding works with exact shapes
"""

import cv2
import numpy as np
import os
from extractRandom import extractSpatialGrid

print("="*80)
print("STEP-BY-STEP EXPLANATION: SPATIAL GRID RGB HISTOGRAM")
print("="*80)

# ============================================================================
# STEP 1: READING THE IMAGE (Same as before)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: READING THE IMAGE")
print("="*80)

dataset_folder = './MSRC_ObjCategImageDatabase_v2/Images'
image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.bmp')]
example_image = image_files[0]
img_path = os.path.join(dataset_folder, example_image)

img_original = cv2.imread(img_path)
img_normalized = img_original.astype(np.float64) / 255.0

print(f"\n📷 Image: {example_image}")
print(f"📐 Image shape: {img_normalized.shape}")
print(f"   - Height: {img_normalized.shape[0]} pixels")
print(f"   - Width: {img_normalized.shape[1]} pixels")
print(f"   - Channels: {img_normalized.shape[2]} (B, G, R)")

# ============================================================================
# STEP 2: UNDERSTANDING SPATIAL GRIDDING CONCEPT
# ============================================================================
print("\n" + "="*80)
print("STEP 2: UNDERSTANDING SPATIAL GRIDDING")
print("="*80)

GRID_ROWS = 3
GRID_COLS = 3
NUM_BINS = 8

height, width = img_normalized.shape[:2]
cell_height = height // GRID_ROWS
cell_width = width // GRID_COLS

print(f"\n🔲 GRID CONFIGURATION:")
print(f"   - Grid size: {GRID_ROWS} rows × {GRID_COLS} columns = {GRID_ROWS * GRID_COLS} cells")
print(f"   - Image size: {height} × {width} pixels")
print(f"   - Cell size: ~{cell_height} × {cell_width} pixels per cell")

print("\n💡 KEY DIFFERENCE FROM GLOBAL HISTOGRAM:")
print("   Global Histogram: Counts colors across the ENTIRE image")
print("   Spatial Grid: Divides image into cells, counts colors in EACH cell separately")
print("   This preserves SPATIAL INFORMATION (where colors appear)")

print("\n📐 VISUAL REPRESENTATION OF GRID:")
print("   ┌─────────┬─────────┬─────────┐")
print("   │  Cell   │  Cell   │  Cell   │")
print("   │  (0,0)  │  (0,1)  │  (0,2)  │")
print("   ├─────────┼─────────┼─────────┤")
print("   │  Cell   │  Cell   │  Cell   │")
print("   │  (1,0)  │  (1,1)  │  (1,2)  │")
print("   ├─────────┼─────────┼─────────┤")
print("   │  Cell   │  Cell   │  Cell   │")
print("   │  (2,0)  │  (2,1)  │  (2,2)  │")
print("   └─────────┴─────────┴─────────┘")
print(f"   Total: {GRID_ROWS * GRID_COLS} cells")

# ============================================================================
# STEP 3: DIVIDING IMAGE INTO GRID CELLS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DIVIDING IMAGE INTO GRID CELLS")
print("="*80)

print(f"\n📐 CELL BOUNDARIES:")
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        y_start = row * cell_height
        y_end = (row + 1) * cell_height if row < GRID_ROWS - 1 else height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width if col < GRID_COLS - 1 else width
        
        cell_height_actual = y_end - y_start
        cell_width_actual = x_end - x_start
        
        print(f"   Cell ({row},{col}): rows [{y_start}:{y_end}], cols [{x_start}:{x_end}]")
        print(f"                    Size: {cell_height_actual} × {cell_width_actual} pixels")

print("\n💡 EXPLANATION:")
print("   Each cell is a rectangular region of the image")
print("   We extract each cell and compute a histogram separately")
print("   This tells us WHERE colors appear, not just WHAT colors")

# ============================================================================
# STEP 4: COMPUTING HISTOGRAM FOR EACH CELL
# ============================================================================
print("\n" + "="*80)
print("STEP 4: COMPUTING HISTOGRAM FOR EACH GRID CELL")
print("="*80)

print(f"\n🔍 PROCESS FOR EACH CELL:")
print(f"   1. Extract cell region from image")
print(f"   2. Compute RGB histogram with {NUM_BINS} bins per channel")
print(f"   3. Normalize histogram (sums to 1)")
print(f"   4. Result: {NUM_BINS * 3} numbers per cell")

print(f"\n📊 HISTOGRAM PER CELL:")
print(f"   - Bins per channel: {NUM_BINS}")
print(f"   - Channels: R, G, B (3 channels)")
print(f"   - Numbers per cell: {NUM_BINS} × 3 = {NUM_BINS * 3}")

# Extract actual cells to show
print(f"\n📐 ACTUAL CELL SIZES:")
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        y_start = row * cell_height
        y_end = (row + 1) * cell_height if row < GRID_ROWS - 1 else height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width if col < GRID_COLS - 1 else width
        
        cell = img_normalized[y_start:y_end, x_start:x_end, :]
        print(f"   Cell ({row},{col}): shape {cell.shape} = {cell.shape[0] * cell.shape[1]} pixels")

# ============================================================================
# STEP 5: CONCATENATING ALL CELL HISTOGRAMS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CONCATENATING ALL CELL HISTOGRAMS")
print("="*80)

spatial_descriptor = extractSpatialGrid(img_normalized, grid_rows=GRID_ROWS, 
                                        grid_cols=GRID_COLS, num_bins=NUM_BINS)

print(f"\n📐 SPATIAL GRID DESCRIPTOR SHAPE: {spatial_descriptor.shape}")

total_cells = GRID_ROWS * GRID_COLS
numbers_per_cell = NUM_BINS * 3
total_numbers = total_cells * numbers_per_cell

print(f"\n🔢 BREAKDOWN:")
print(f"   - Number of cells: {total_cells}")
print(f"   - Numbers per cell: {numbers_per_cell}")
print(f"   - Total descriptor length: {total_cells} × {numbers_per_cell} = {total_numbers}")

print(f"\n📋 DESCRIPTOR STRUCTURE:")
print(f"   [Cell(0,0)_R, Cell(0,0)_G, Cell(0,0)_B,")
print(f"    Cell(0,1)_R, Cell(0,1)_G, Cell(0,1)_B,")
print(f"    Cell(0,2)_R, Cell(0,2)_G, Cell(0,2)_B,")
print(f"    Cell(1,0)_R, Cell(1,0)_G, Cell(1,0)_B,")
print(f"    ...")
print(f"    Cell(2,2)_R, Cell(2,2)_G, Cell(2,2)_B]")

print(f"\n   Total: {total_numbers} numbers in a single row vector")

# Show actual breakdown
print(f"\n🔍 ACTUAL VALUES BREAKDOWN:")
for cell_idx in range(min(3, total_cells)):  # Show first 3 cells
    start_idx = cell_idx * numbers_per_cell
    end_idx = start_idx + numbers_per_cell
    cell_values = spatial_descriptor[0, start_idx:end_idx]
    row = cell_idx // GRID_COLS
    col = cell_idx % GRID_COLS
    print(f"   Cell ({row},{col}): {cell_values[:3]} ... (first 3 values shown)")

# ============================================================================
# STEP 6: COMPARISON WITH GLOBAL HISTOGRAM
# ============================================================================
print("\n" + "="*80)
print("STEP 6: COMPARISON WITH GLOBAL HISTOGRAM")
print("="*80)

from extractRandom import extractRGBHistogram
global_descriptor = extractRGBHistogram(img_normalized, num_bins=NUM_BINS)

print(f"\n📊 COMPARISON:")
print(f"   Global RGB Histogram:")
print(f"   - Shape: {global_descriptor.shape}")
print(f"   - Length: {global_descriptor.shape[1]} numbers")
print(f"   - Represents: Overall color distribution (ignores location)")

print(f"\n   Spatial Grid RGB Histogram:")
print(f"   - Shape: {spatial_descriptor.shape}")
print(f"   - Length: {spatial_descriptor.shape[1]} numbers")
print(f"   - Represents: Color distribution in each spatial region")

print(f"\n   Difference:")
print(f"   - Global: {global_descriptor.shape[1]} numbers")
print(f"   - Spatial Grid: {spatial_descriptor.shape[1]} numbers")
print(f"   - Ratio: {spatial_descriptor.shape[1] / global_descriptor.shape[1]:.1f}x more information")

print("\n💡 WHY SPATIAL GRID IS BETTER:")
print("   Example: Two images with same colors but different layouts")
print("   - Image A: Red on left, Blue on right")
print("   - Image B: Blue on left, Red on right")
print("   - Global histogram: SAME (both have 50% red, 50% blue)")
print("   - Spatial grid: DIFFERENT (cell positions differ)")
print("   → Spatial grid can distinguish these images!")

# ============================================================================
# STEP 7: COLLECTING ALL SPATIAL GRID DESCRIPTORS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: COLLECTING DESCRIPTORS FROM ALL IMAGES")
print("="*80)

print(f"\n📦 For all 591 images:")
print(f"   - Each image: ({spatial_descriptor.shape[0]}, {spatial_descriptor.shape[1]})")
print(f"   - All images: (591, {spatial_descriptor.shape[1]})")

print(f"\n📐 FINAL DESCRIPTOR MATRIX:")
print(f"   ┌────────────────────────────────────────────────────────────┐")
print(f"   │ [Cell(0,0)...Cell(2,2)]  ← Image 1  │ Row 0")
print(f"   │ [Cell(0,0)...Cell(2,2)]  ← Image 2  │ Row 1")
print(f"   │ [Cell(0,0)...Cell(2,2)]  ← Image 3  │ Row 2")
print(f"   │              ...                    │")
print(f"   │ [Cell(0,0)...Cell(2,2)]  ← Image 591│ Row 590")
print(f"   └────────────────────────────────────────────────────────────┘")
print(f"         ↑                              ↑")
print(f"      Column 0                    Column {spatial_descriptor.shape[1]-1}")

# ============================================================================
# STEP 8: PCA ON SPATIAL GRID DESCRIPTORS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: APPLYING PCA TO SPATIAL GRID DESCRIPTORS")
print("="*80)

print(f"\n📐 INPUT TO PCA:")
print(f"   - Shape: (591, {spatial_descriptor.shape[1]})")
print(f"   - Original dimension: {spatial_descriptor.shape[1]}")

print(f"\n🔧 PCA PROCESS:")
print(f"   - Computes covariance matrix: ({spatial_descriptor.shape[1]}, {spatial_descriptor.shape[1]})")
print(f"   - Finds principal components")
print(f"   - Components shape: ({spatial_descriptor.shape[1]}, {spatial_descriptor.shape[1]})")

print(f"\n📉 PCA PROJECTION (Example: PCA 20):")
print(f"   - Original: (591, {spatial_descriptor.shape[1]})")
print(f"   - Projected: (591, 20)")
print(f"   - Reduction: {spatial_descriptor.shape[1]} → 20 dimensions")
reduction_pct = (1 - 20 / spatial_descriptor.shape[1]) * 100
print(f"   - Compression: {reduction_pct:.1f}% reduction")

print("\n💡 EXPLANATION:")
print("   Same PCA process as global histogram, but:")
print(f"   - Starting dimension is {spatial_descriptor.shape[1]} instead of 24")
print("   - More dimensions = more information, but also more redundancy")
print("   - PCA finds the most important patterns across all cells")

# ============================================================================
# VISUAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUAL SUMMARY: SPATIAL GRID PIPELINE")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    SPATIAL GRID PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

1. IMAGE READING
   ┌─────────────┐
   │ Image File  │ → (213, 320, 3)
   └─────────────┘
         │
         ▼ Normalize
   (213, 320, 3) - Values [0, 1]

2. DIVIDE INTO GRID
   ┌─────────────┐
   │ Normalized  │ → Divide into 3×3 = 9 cells
   │   Image     │
   └─────────────┘
         │
         ▼
   ┌─────┬─────┬─────┐
   │ C₀₀ │ C₀₁ │ C₀₂ │  Each cell: ~71×107 pixels
   ├─────┼─────┼─────┤
   │ C₁₀ │ C₁₁ │ C₁₂ │
   ├─────┼─────┼─────┤
   │ C₂₀ │ C₂₁ │ C₂₂ │
   └─────┴─────┴─────┘

3. COMPUTE HISTOGRAM FOR EACH CELL
   ┌─────┐  ┌─────┐  ┌─────┐
   │ C₀₀ │  │ C₀₁ │  │ C₀₂ │
   │Hist │  │Hist │  │Hist │  Each histogram: 24 numbers
   │(24) │  │(24) │  │(24) │  (8 bins × 3 channels)
   └─────┘  └─────┘  └─────┘
   ┌─────┐  ┌─────┐  ┌─────┐
   │ C₁₀ │  │ C₁₁ │  │ C₁₂ │
   │Hist │  │Hist │  │Hist │
   │(24) │  │(24) │  │(24) │
   └─────┘  └─────┘  └─────┘
   ┌─────┐  ┌─────┐  ┌─────┐
   │ C₂₀ │  │ C₂₁ │  │ C₂₂ │
   │Hist │  │Hist │  │Hist │
   │(24) │  │(24) │  │(24) │
   └─────┘  └─────┘  └─────┘

4. CONCATENATE ALL CELL HISTOGRAMS
   [C₀₀_hist | C₀₁_hist | C₀₂_hist | C₁₀_hist | ... | C₂₂_hist]
   
   Shape: (1, 216)
   9 cells × 24 numbers = 216 total numbers

5. COLLECT ALL IMAGES
   ┌──────────────────────────────────────┐
   │ [C₀₀...C₂₂]  ← Image 1  │ Row 0
   │ [C₀₀...C₂₂]  ← Image 2  │ Row 1
   │ [C₀₀...C₂₂]  ← Image 3  │ Row 2
   │      ...                │
   │ [C₀₀...C₂₂]  ← Image 591│ Row 590
   └──────────────────────────────────────┘
   
   Shape: (591, 216)

6. PCA PROJECTION
   ┌──────────────────────────────────────┐
   │ All         │ → PCA computation
   │ Descriptors │ → Project to 20 dims
   │ (591, 216)  │
   └──────────────────────────────────────┘
         │
         ▼
   (591, 20) - Reduced dimension descriptors

┌─────────────────────────────────────────────────────────────────────────┐
│                    KEY DIFFERENCES                                        │
└─────────────────────────────────────────────────────────────────────────┘

GLOBAL HISTOGRAM:
  • Shape: (1, 24)
  • Represents: "What colors are in the image?"
  • Ignores: Where colors appear
  • Example: "50% red, 50% blue" (could be anywhere)

SPATIAL GRID HISTOGRAM:
  • Shape: (1, 216) for 3×3 grid
  • Represents: "What colors are in each region?"
  • Preserves: Spatial location information
  • Example: "Top-left: mostly red, Bottom-right: mostly blue"

BENEFITS OF SPATIAL GRID:
  ✓ Can distinguish images with same colors but different layouts
  ✓ Captures spatial relationships
  ✓ More discriminative for visual search
  ✗ Larger descriptor size (216 vs 24)
  ✗ More computation required

WHEN TO USE WHICH:
  • Global: Simple, fast, good for color-based search
  • Spatial Grid: Better accuracy, captures spatial patterns
""")

print("\n" + "="*80)
print("END OF SPATIAL GRID EXPLANATION")
print("="*80)

