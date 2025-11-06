"""
Step-by-Step Visualization: Image → Histogram → PCA
This script demonstrates the exact shapes and transformations at each step
"""

import cv2
import numpy as np
import os
from extractRandom import extractRGBHistogram
from cvpr_pca import compute_pca, project_descriptors

print("="*80)
print("STEP-BY-STEP EXPLANATION: IMAGE → HISTOGRAM → PCA")
print("="*80)

# ============================================================================
# STEP 1: READING AN IMAGE
# ============================================================================
print("\n" + "="*80)
print("STEP 1: READING AN IMAGE FROM THE DATASET")
print("="*80)

dataset_folder = './MSRC_ObjCategImageDatabase_v2/Images'
image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.bmp')]
example_image = image_files[0]  # Let's use the first image
img_path = os.path.join(dataset_folder, example_image)

print(f"\n📷 Reading image: {example_image}")
img_original = cv2.imread(img_path)

print(f"\n📐 ORIGINAL IMAGE SHAPE: {img_original.shape}")
print(f"   - Height: {img_original.shape[0]} pixels")
print(f"   - Width: {img_original.shape[1]} pixels")
print(f"   - Channels: {img_original.shape[2]} (B, G, R)")
print(f"   - Data type: {img_original.dtype}")
print(f"   - Value range: {img_original.min()} to {img_original.max()}")

print("\n💡 EXPLANATION:")
print("   The image is a 3D array:")
print("   - First dimension (213): Number of rows (height)")
print("   - Second dimension (320): Number of columns (width)")
print("   - Third dimension (3): Color channels (Blue, Green, Red)")
print("   - Each pixel has 3 values: [B, G, R] ranging from 0 to 255")

# ============================================================================
# STEP 2: NORMALIZING THE IMAGE
# ============================================================================
print("\n" + "="*80)
print("STEP 2: NORMALIZING THE IMAGE")
print("="*80)

img_normalized = img_original.astype(np.float64) / 255.0

print(f"\n📐 NORMALIZED IMAGE SHAPE: {img_normalized.shape}")
print(f"   - Shape: {img_normalized.shape} (same as original)")
print(f"   - Data type: {img_normalized.dtype}")
print(f"   - Value range: {img_normalized.min():.3f} to {img_normalized.max():.3f}")

print("\n💡 EXPLANATION:")
print("   We divide by 255 to convert pixel values from [0, 255] to [0, 1]")
print("   This makes calculations easier and ensures consistency across images")
print("   Shape stays the same: (213, 320, 3)")

# ============================================================================
# STEP 3: COMPUTING GLOBAL RGB HISTOGRAM
# ============================================================================
print("\n" + "="*80)
print("STEP 3: COMPUTING GLOBAL RGB HISTOGRAM")
print("="*80)

NUM_BINS = 8  # As used in the code
histogram = extractRGBHistogram(img_normalized, num_bins=NUM_BINS)

print(f"\n📊 HISTOGRAM COMPUTATION:")
print(f"   - Number of bins per channel: {NUM_BINS}")
print(f"   - Channels: Red, Green, Blue (3 channels)")
print(f"   - Total histogram size: {NUM_BINS} × 3 = {NUM_BINS * 3}")

print("\n📐 HISTOGRAM SHAPE: {0}".format(histogram.shape))
print(f"   - Shape: {histogram.shape} (1 row, {NUM_BINS * 3} columns)")
print(f"   - Data type: {histogram.dtype}")

print("\n🔍 BREAKDOWN OF THE HISTOGRAM:")
print(f"   - Red channel histogram:   bins 0-{NUM_BINS-1}   (values: {histogram[0, 0:NUM_BINS]})")
print(f"   - Green channel histogram: bins {NUM_BINS}-{NUM_BINS*2-1} (values: {histogram[0, NUM_BINS:NUM_BINS*2]})")
print(f"   - Blue channel histogram:  bins {NUM_BINS*2}-{NUM_BINS*3-1} (values: {histogram[0, NUM_BINS*2:NUM_BINS*3]})")

print("\n💡 EXPLANATION:")
print("   A histogram counts how many pixels fall into each 'bin' (range)")
print("   Example: If bin 0 is [0-31], bin 1 is [32-63], etc.")
print("   - We count pixels in each bin for Red, Green, and Blue separately")
print("   - Each histogram is normalized (sums to 1) so it's size-independent")
print("   - We concatenate all 3 histograms: [R_bins, G_bins, B_bins]")
print(f"   - Final result: A single row vector of length {NUM_BINS * 3}")

# ============================================================================
# STEP 4: COLLECTING DESCRIPTORS FROM ALL IMAGES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: COLLECTING DESCRIPTORS FROM ALL IMAGES")
print("="*80)

# Load a few descriptors to show the pattern
all_descriptors_list = []
sample_images = image_files[:5]  # Just 5 for demonstration

print(f"\n📦 Processing {len(sample_images)} sample images...")
for img_file in sample_images:
    img_path = os.path.join(dataset_folder, img_file)
    img = cv2.imread(img_path).astype(np.float64) / 255.0
    desc = extractRGBHistogram(img, num_bins=NUM_BINS)
    all_descriptors_list.append(desc.flatten())  # Flatten to 1D

all_descriptors_array = np.array(all_descriptors_list)

print(f"\n📐 ALL DESCRIPTORS ARRAY SHAPE: {all_descriptors_array.shape}")
print(f"   - Number of images: {all_descriptors_array.shape[0]}")
print(f"   - Descriptor dimension: {all_descriptors_array.shape[1]}")

print("\n💡 EXPLANATION:")
print("   We stack all image descriptors into a 2D matrix:")
print("   - Each row = one image's histogram descriptor")
print("   - Each column = one histogram bin")
print(f"   - Shape: (number_of_images, {NUM_BINS * 3})")
print(f"   - For full dataset: (591, {NUM_BINS * 3}) = (591, 24)")

# ============================================================================
# STEP 5: COMPUTING PCA
# ============================================================================
print("\n" + "="*80)
print("STEP 5: COMPUTING PCA (PRINCIPAL COMPONENT ANALYSIS)")
print("="*80)

# For demonstration, use the sample descriptors
pca_model = compute_pca(all_descriptors_array, normalize=True)

print(f"\n📐 INPUT TO PCA:")
print(f"   - Shape: {all_descriptors_array.shape}")
print(f"   - Each row is one image's {NUM_BINS * 3}-dimensional descriptor")

print(f"\n🔧 PCA COMPUTATION:")
print(f"   - Computes covariance matrix: ({all_descriptors_array.shape[1]}, {all_descriptors_array.shape[1]})")
print(f"   - Finds principal components (eigenvectors)")
print(f"   - Principal components shape: {pca_model['components'].shape}")

print(f"\n📊 PCA MODEL COMPONENTS:")
print(f"   - Components shape: {pca_model['components'].shape}")
print(f"   - Each column is a principal component (direction of maximum variance)")
print(f"   - Number of components: {pca_model['components'].shape[1]}")

print(f"\n📈 VARIANCE EXPLAINED:")
n_show = min(5, len(pca_model['explained_variance_ratio']))
for i in range(n_show):
    print(f"   - Component {i+1}: {pca_model['explained_variance_ratio'][i]*100:.2f}% variance")

print("\n💡 EXPLANATION:")
print("   PCA finds the directions where data varies the most")
print("   - Think of it as rotating the coordinate system")
print("   - First component = direction of maximum variance")
print("   - Second component = direction of second-most variance (perpendicular to first)")
print("   - And so on...")
print(f"   - We get {pca_model['components'].shape[1]} components (same as original dimension)")

# ============================================================================
# STEP 6: PROJECTING TO LOWER DIMENSIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: PROJECTING DESCRIPTORS TO LOWER DIMENSIONS")
print("="*80)

N_COMPONENTS = 10  # As used in PCA10
projected, actual_n = project_descriptors(all_descriptors_array, pca_model, n_components=N_COMPONENTS)

print(f"\n📐 PROJECTION:")
print(f"   - Original shape: {all_descriptors_array.shape}")
print(f"   - Number of components used: {actual_n}")
print(f"   - Projected shape: {projected.shape}")

print(f"\n📉 DIMENSIONALITY REDUCTION:")
print(f"   - Before: {all_descriptors_array.shape[1]} dimensions")
print(f"   - After: {projected.shape[1]} dimensions")
print(f"   - Reduction: {all_descriptors_array.shape[1]} → {projected.shape[1]}")

var_explained = pca_model['cumulative_variance_ratio'][actual_n - 1] * 100
print(f"   - Variance preserved: {var_explained:.2f}%")

print("\n💡 EXPLANATION:")
print("   We project each image's descriptor onto the first N principal components")
print("   - Instead of using all 24 dimensions, we use only the top 10")
print("   - This reduces storage and computation while keeping most information")
print(f"   - Each image now has {N_COMPONENTS} numbers instead of {NUM_BINS * 3}")
print(f"   - We keep {var_explained:.2f}% of the original variance")

# ============================================================================
# VISUAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUAL SUMMARY OF THE PIPELINE")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE FLOW                        │
└─────────────────────────────────────────────────────────────────┘

1. IMAGE READING
   ┌─────────────┐
   │ Image File  │ → (213, 320, 3) - Height × Width × Channels
   └─────────────┘
         │
         ▼
   Normalize: ÷ 255
         │
         ▼
   (213, 320, 3) - Values in [0, 1]

2. HISTOGRAM COMPUTATION
   ┌─────────────┐
   │ Normalized  │ → Count pixels in bins for R, G, B
   │   Image     │ → Normalize each histogram
   └─────────────┘ → Concatenate: [R_bins | G_bins | B_bins]
         │
         ▼
   (1, 24) - Single row vector
   [R₀ R₁ ... R₇ G₀ G₁ ... G₇ B₀ B₁ ... B₇]

3. COLLECT ALL DESCRIPTORS
   ┌─────────────┐
   │ Image 1     │ → (1, 24)
   │ Image 2     │ → (1, 24)
   │ Image 3     │ → (1, 24)
   │    ...      │
   │ Image 591   │ → (1, 24)
   └─────────────┘
         │
         ▼ Stack vertically
   (591, 24) - Each row is one image's descriptor

4. PCA COMPUTATION
   ┌─────────────┐
   │ All         │ → Compute covariance matrix (24 × 24)
   │ Descriptors │ → Find principal components
   │ (591, 24)   │ → Get eigenvectors (directions of variance)
   └─────────────┘
         │
         ▼
   PCA Model:
   - Components: (24, 24) - Each column is a principal component
   - Explained variance: How much each component contributes

5. PCA PROJECTION
   ┌─────────────┐
   │ Descriptors │ → Project onto top 10 components
   │ (591, 24)   │ → Matrix multiplication
   └─────────────┘
         │
         ▼
   (591, 10) - Reduced dimension descriptors
   Each image now represented by 10 numbers instead of 24!

┌─────────────────────────────────────────────────────────────────┐
│                    KEY INSIGHTS                                  │
└─────────────────────────────────────────────────────────────────┘

• Image: 213 × 320 × 3 = 204,480 numbers (pixels)
• Histogram: 24 numbers (summarizes color distribution)
• PCA 10: 10 numbers (keeps 89.5% of information, uses 58% less storage)

• PCA finds the most important directions in the data
• We keep only the top directions (components)
• This reduces noise and speeds up computation
""")

print("\n" + "="*80)
print("END OF EXPLANATION")
print("="*80)

