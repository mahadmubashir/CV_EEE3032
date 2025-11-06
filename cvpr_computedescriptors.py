import os
import shutil
import numpy as np
import cv2
import scipy.io as sio
from extractRandom import extractRGBHistogram, extractSpatialGrid
from cvpr_pca import compute_pca, project_descriptors, save_pca_model, print_pca_summary

DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = './descriptors'

# Define subfolders for different descriptor types
GLOBAL_RGB_SUBFOLDER = 'globalRGBhisto'
SPATIAL_GRID_SUBFOLDER = 'spatialGridRGBhisto'
PCA_MODELS_FOLDER = 'pca_models'

# Spatial grid parameters
GRID_ROWS = 4
GRID_COLS = 4
NUM_BINS = 4  # Bins per channel for both global and spatial grid

# PCA parameters - dimensions to experiment with
# PCA_COMPONENTS_LIST = [10, 20, 50, 100]  # Number of components to keep
PCA_COMPONENTS_LIST = [4]

# ============================================================================
# Clean up old descriptors before computing new ones
# ============================================================================
print("="*60)
print("CLEANING UP OLD DESCRIPTORS")
print("="*60)

if os.path.exists(OUT_FOLDER):
    print(f"Removing existing folder: {OUT_FOLDER}")
    shutil.rmtree(OUT_FOLDER)
    print("✅ Old descriptors removed")

# Create fresh output directories
print(f"\nCreating fresh output directories...")
os.makedirs(os.path.join(OUT_FOLDER, GLOBAL_RGB_SUBFOLDER), exist_ok=True)
os.makedirs(os.path.join(OUT_FOLDER, SPATIAL_GRID_SUBFOLDER), exist_ok=True)
os.makedirs(os.path.join(OUT_FOLDER, PCA_MODELS_FOLDER), exist_ok=True)
print("✅ Output directories created")
print("="*60)

print("="*60)
print("Computing Feature Descriptors")
print("="*60)
print(f"Global RGB Histogram: {GLOBAL_RGB_SUBFOLDER}")
print(f"Spatial Grid RGB Histogram: {SPATIAL_GRID_SUBFOLDER} (grid: {GRID_ROWS}x{GRID_COLS}, bins: {NUM_BINS})")
print("="*60)

# Iterate through all BMP files in the dataset folder
image_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith(".bmp")]
total_images = len(image_files)

# Store all descriptors for PCA computation
all_global_descriptors = []
all_spatial_descriptors = []
image_filenames = []

for idx, filename in enumerate(image_files, 1):
    print(f"[{idx}/{total_images}] Processing {filename}")
    
    img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
    img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
    
    # === Compute Global RGB Histogram Descriptor ===
    global_descriptor = extractRGBHistogram(img, num_bins=NUM_BINS)
    global_out_path = os.path.join(OUT_FOLDER, GLOBAL_RGB_SUBFOLDER, filename.replace('.bmp', '.mat'))
    sio.savemat(global_out_path, {'F': global_descriptor})
    all_global_descriptors.append(global_descriptor.flatten())
    
    # === Compute Spatial Grid RGB Histogram Descriptor ===
    spatial_descriptor = extractSpatialGrid(img, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, num_bins=NUM_BINS)
    spatial_out_path = os.path.join(OUT_FOLDER, SPATIAL_GRID_SUBFOLDER, filename.replace('.bmp', '.mat'))
    sio.savemat(spatial_out_path, {'F': spatial_descriptor})
    all_spatial_descriptors.append(spatial_descriptor.flatten())
    
    image_filenames.append(filename)

print("\n" + "="*60)
print("Descriptor computation complete!")
print(f"Global RGB histograms saved to: {GLOBAL_RGB_SUBFOLDER}")
print(f"Spatial grid histograms saved to: {SPATIAL_GRID_SUBFOLDER}")
print("="*60)

# ============================================================================
# PCA Computation and Projection
# ============================================================================

print("\n" + "="*60)
print("COMPUTING PCA MODELS")
print("="*60)

# Convert to numpy arrays
print("\n📊 Converting descriptor lists to numpy arrays...")
all_global_descriptors = np.array(all_global_descriptors)
all_spatial_descriptors = np.array(all_spatial_descriptors)

print(f"\n✅ Descriptor Arrays Created:")
print(f"   Global RGB Histogram descriptors shape: {all_global_descriptors.shape}")
print(f"     → {all_global_descriptors.shape[0]} images × {all_global_descriptors.shape[1]} dimensions per image")
print(f"     → Each image descriptor: ({all_global_descriptors.shape[1]},) = {all_global_descriptors.shape[1]} numbers")
print(f"\n   Spatial Grid descriptors shape: {all_spatial_descriptors.shape}")
print(f"     → {all_spatial_descriptors.shape[0]} images × {all_spatial_descriptors.shape[1]} dimensions per image")
print(f"     → Each image descriptor: ({all_spatial_descriptors.shape[1]},) = {all_spatial_descriptors.shape[1]} numbers")

# Show example descriptors
print(f"\n🔍 Example Descriptor (First Image):")
print(f"   Global RGB Histogram (first 5 values): {all_global_descriptors[0, :5]}")
print(f"   Spatial Grid (first 5 values): {all_spatial_descriptors[0, :5]}")

# === Compute PCA for Global RGB Histogram ===
print("\n" + "="*60)
print("COMPUTING PCA FOR GLOBAL RGB HISTOGRAM")
print("="*60)

print(f"\n📐 INPUT TO PCA:")
print(f"   Descriptor matrix shape: {all_global_descriptors.shape}")
print(f"   - Rows (images): {all_global_descriptors.shape[0]}")
print(f"   - Columns (features): {all_global_descriptors.shape[1]}")
print(f"   - Each row = one image's {all_global_descriptors.shape[1]}-dimensional descriptor")

print(f"\n🔧 PCA COMPUTATION PROCESS:")
print(f"   Step 1: Computing mean of each feature dimension...")
mean_global = np.mean(all_global_descriptors, axis=0)
print(f"     Mean vector shape: {mean_global.shape}")
print(f"     Mean values (first 5): {mean_global[:5]}")

print(f"\n   Step 2: Centering data (subtracting mean)...")
centered_global = all_global_descriptors - mean_global
print(f"     Centered data shape: {centered_global.shape} (same as input)")

print(f"\n   Step 3: Computing covariance matrix...")
print(f"     Covariance matrix will be: ({all_global_descriptors.shape[1]}, {all_global_descriptors.shape[1]})")
print(f"     This shows relationships between {all_global_descriptors.shape[1]} feature dimensions")

print(f"\n   Step 4: Finding principal components (eigenvectors)...")
pca_global = compute_pca(all_global_descriptors, normalize=True)

print(f"\n✅ PCA MODEL CREATED:")
print(f"   Principal components shape: {pca_global['components'].shape}")
print(f"     → {pca_global['components'].shape[0]} rows × {pca_global['components'].shape[1]} columns")
print(f"     → Each column = one principal component (direction vector)")
print(f"     → Each component has {pca_global['components'].shape[0]} values")

print(f"\n   Explained variance shape: {pca_global['explained_variance_ratio'].shape}")
print(f"     → {len(pca_global['explained_variance_ratio'])} components")
print(f"     → First 5 components explain:")
for i in range(min(5, len(pca_global['explained_variance_ratio']))):
    print(f"       Component {i+1}: {pca_global['explained_variance_ratio'][i]*100:.2f}%")

print_pca_summary(pca_global, PCA_COMPONENTS_LIST)

# Save PCA model
pca_global_path = os.path.join(OUT_FOLDER, PCA_MODELS_FOLDER, 'globalRGBhisto_pca.mat')
save_pca_model(pca_global, pca_global_path)

# Project and save PCA-reduced descriptors
print(f"\n" + "-"*60)
print(f"PROJECTING GLOBAL RGB HISTOGRAM TO LOWER DIMENSIONS")
print("-"*60)

for n_comp in PCA_COMPONENTS_LIST:
    if n_comp <= all_global_descriptors.shape[1]:
        print(f"\n🔄 PROJECTION TO {n_comp} DIMENSIONS:")
        print(f"   Input shape: {all_global_descriptors.shape}")
        print(f"   Using first {n_comp} principal components")
        
        # Show example before projection
        example_before = all_global_descriptors[0]
        print(f"   Example descriptor BEFORE PCA:")
        print(f"     Shape: {example_before.shape}")
        print(f"     First 5 values: {example_before[:5]}")
        print(f"     Last 5 values: {example_before[-5:]}")
        
        projected, actual_n_comp = project_descriptors(all_global_descriptors, pca_global, n_components=n_comp)
        
        print(f"\n   ✅ PROJECTION COMPLETE:")
        print(f"     Output shape: {projected.shape}")
        print(f"     - {projected.shape[0]} images × {projected.shape[1]} dimensions")
        print(f"     - Reduction: {all_global_descriptors.shape[1]} → {projected.shape[1]} dimensions")
        reduction_pct = (1 - projected.shape[1] / all_global_descriptors.shape[1]) * 100
        print(f"     - Compression: {reduction_pct:.1f}% reduction")
        
        # Show example after projection
        example_after = projected[0]
        print(f"\n   Example descriptor AFTER PCA ({n_comp} components):")
        print(f"     Shape: {example_after.shape}")
        print(f"     All {n_comp} values: {example_after}")
        
        var_explained = pca_global['cumulative_variance_ratio'][actual_n_comp - 1] * 100
        print(f"\n   📊 Variance preserved: {var_explained:.2f}%")
        print(f"   💾 Saving to: {GLOBAL_RGB_SUBFOLDER}_PCA{n_comp}/")
        
        # Create subfolder for this PCA dimension
        pca_subfolder = f'{GLOBAL_RGB_SUBFOLDER}_PCA{n_comp}'
        pca_folder_path = os.path.join(OUT_FOLDER, pca_subfolder)
        os.makedirs(pca_folder_path, exist_ok=True)
        
        # Save projected descriptors
        for i, filename in enumerate(image_filenames):
            projected_desc = projected[i].reshape(1, -1)
            pca_out_path = os.path.join(pca_folder_path, filename.replace('.bmp', '.mat'))
            sio.savemat(pca_out_path, {'F': projected_desc})
        
        print(f"   ✅ Saved {len(image_filenames)} projected descriptors")

# === Compute PCA for Spatial Grid ===
print("\n" + "="*60)
print("COMPUTING PCA FOR SPATIAL GRID RGB HISTOGRAM")
print("="*60)

print(f"\n📐 INPUT TO PCA:")
print(f"   Descriptor matrix shape: {all_spatial_descriptors.shape}")
print(f"   - Rows (images): {all_spatial_descriptors.shape[0]}")
print(f"   - Columns (features): {all_spatial_descriptors.shape[1]}")
print(f"   - Each row = one image's {all_spatial_descriptors.shape[1]}-dimensional descriptor")
print(f"   - This is {all_spatial_descriptors.shape[1] / all_global_descriptors.shape[1]:.1f}x larger than global histogram!")

print(f"\n🔧 PCA COMPUTATION PROCESS:")
print(f"   Step 1: Computing mean of each feature dimension...")
mean_spatial = np.mean(all_spatial_descriptors, axis=0)
print(f"     Mean vector shape: {mean_spatial.shape}")
print(f"     Mean values (first 5): {mean_spatial[:5]}")

print(f"\n   Step 2: Centering data (subtracting mean)...")
centered_spatial = all_spatial_descriptors - mean_spatial
print(f"     Centered data shape: {centered_spatial.shape} (same as input)")

print(f"\n   Step 3: Computing covariance matrix...")
print(f"     Covariance matrix will be: ({all_spatial_descriptors.shape[1]}, {all_spatial_descriptors.shape[1]})")
print(f"     This is a {all_spatial_descriptors.shape[1]}×{all_spatial_descriptors.shape[1]} matrix!")
print(f"     Shows relationships between {all_spatial_descriptors.shape[1]} feature dimensions")

print(f"\n   Step 4: Finding principal components (eigenvectors)...")
pca_spatial = compute_pca(all_spatial_descriptors, normalize=True)

print(f"\n✅ PCA MODEL CREATED:")
print(f"   Principal components shape: {pca_spatial['components'].shape}")
print(f"     → {pca_spatial['components'].shape[0]} rows × {pca_spatial['components'].shape[1]} columns")
print(f"     → Each column = one principal component (direction vector)")
print(f"     → Each component has {pca_spatial['components'].shape[0]} values")

print(f"\n   Explained variance shape: {pca_spatial['explained_variance_ratio'].shape}")
print(f"     → {len(pca_spatial['explained_variance_ratio'])} components")
print(f"     → First 5 components explain:")
for i in range(min(5, len(pca_spatial['explained_variance_ratio']))):
    print(f"       Component {i+1}: {pca_spatial['explained_variance_ratio'][i]*100:.2f}%")

print_pca_summary(pca_spatial, PCA_COMPONENTS_LIST)

# Save PCA model
pca_spatial_path = os.path.join(OUT_FOLDER, PCA_MODELS_FOLDER, 'spatialGridRGBhisto_pca.mat')
save_pca_model(pca_spatial, pca_spatial_path)

# Project and save PCA-reduced descriptors
print(f"\n" + "-"*60)
print(f"PROJECTING SPATIAL GRID RGB HISTOGRAM TO LOWER DIMENSIONS")
print("-"*60)

for n_comp in PCA_COMPONENTS_LIST:
    if n_comp <= all_spatial_descriptors.shape[1]:
        print(f"\n🔄 PROJECTION TO {n_comp} DIMENSIONS:")
        print(f"   Input shape: {all_spatial_descriptors.shape}")
        print(f"   Using first {n_comp} principal components")
        
        # Show example before projection
        example_before = all_spatial_descriptors[0]
        print(f"   Example descriptor BEFORE PCA:")
        print(f"     Shape: {example_before.shape}")
        print(f"     First 5 values: {example_before[:5]}")
        print(f"     Values at positions 50-54: {example_before[50:55]}")
        print(f"     Last 5 values: {example_before[-5:]}")
        
        projected, actual_n_comp = project_descriptors(all_spatial_descriptors, pca_spatial, n_components=n_comp)
        
        print(f"\n   ✅ PROJECTION COMPLETE:")
        print(f"     Output shape: {projected.shape}")
        print(f"     - {projected.shape[0]} images × {projected.shape[1]} dimensions")
        print(f"     - Reduction: {all_spatial_descriptors.shape[1]} → {projected.shape[1]} dimensions")
        reduction_pct = (1 - projected.shape[1] / all_spatial_descriptors.shape[1]) * 100
        print(f"     - Compression: {reduction_pct:.1f}% reduction")
        
        # Show example after projection
        example_after = projected[0]
        print(f"\n   Example descriptor AFTER PCA ({n_comp} components):")
        print(f"     Shape: {example_after.shape}")
        print(f"     All {n_comp} values: {example_after}")
        
        var_explained = pca_spatial['cumulative_variance_ratio'][actual_n_comp - 1] * 100
        print(f"\n   📊 Variance preserved: {var_explained:.2f}%")
        print(f"   💾 Saving to: {SPATIAL_GRID_SUBFOLDER}_PCA{n_comp}/")
        
        # Create subfolder for this PCA dimension
        pca_subfolder = f'{SPATIAL_GRID_SUBFOLDER}_PCA{n_comp}'
        pca_folder_path = os.path.join(OUT_FOLDER, pca_subfolder)
        os.makedirs(pca_folder_path, exist_ok=True)
        
        # Save projected descriptors
        for i, filename in enumerate(image_filenames):
            projected_desc = projected[i].reshape(1, -1)
            pca_out_path = os.path.join(pca_folder_path, filename.replace('.bmp', '.mat'))
            sio.savemat(pca_out_path, {'F': projected_desc})
        
        print(f"   ✅ Saved {len(image_filenames)} projected descriptors")

print("\n" + "="*60)
print("PCA COMPUTATION COMPLETE!")
print("="*60)

print("\n📊 FINAL SUMMARY:")
print("="*60)
print("\nGLOBAL RGB HISTOGRAM:")
print(f"   Original descriptors: (591, {all_global_descriptors.shape[1]})")
print(f"   Each image: {all_global_descriptors.shape[1]} numbers")
for n_comp in PCA_COMPONENTS_LIST:
    if n_comp <= all_global_descriptors.shape[1]:
        var_explained = pca_global['cumulative_variance_ratio'][n_comp - 1] * 100
        print(f"   PCA {n_comp}: (591, {n_comp}) - {var_explained:.2f}% variance preserved")

print("\nSPATIAL GRID RGB HISTOGRAM:")
print(f"   Original descriptors: (591, {all_spatial_descriptors.shape[1]})")
print(f"   Each image: {all_spatial_descriptors.shape[1]} numbers")
for n_comp in PCA_COMPONENTS_LIST:
    if n_comp <= all_spatial_descriptors.shape[1]:
        var_explained = pca_spatial['cumulative_variance_ratio'][n_comp - 1] * 100
        print(f"   PCA {n_comp}: (591, {n_comp}) - {var_explained:.2f}% variance preserved")

print("\n" + "="*60)
print(f"✅ PCA models saved to: {PCA_MODELS_FOLDER}/")
print(f"✅ PCA-projected descriptors saved to: *_PCA* subfolders")
print("="*60)

