import os
import shutil
import numpy as np
import cv2
import scipy.io as sio
from extractRandom import extractRGBHistogram, extractSpatialGrid
from cvpr_pca import compute_pca, project_descriptors, save_pca_model, print_pca_summary

DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = './descriptors'
PCA_MODELS_FOLDER = 'pca_models'

# Descriptor configurations to compute
DESCRIPTOR_CONFIGS = [
    # Global RGB Histograms
    {'type': 'global', 'bins': 4, 'name': 'globalRGBhisto_4bins'},
    {'type': 'global', 'bins': 8, 'name': 'globalRGBhisto_8bins'},
    {'type': 'global', 'bins': 16, 'name': 'globalRGBhisto_16bins'},
    {'type': 'global', 'bins': 32, 'name': 'globalRGBhisto_32bins'},
    
    # Spatial Grid RGB Histograms
    {'type': 'spatial', 'bins': 8, 'rows': 2, 'cols': 2, 'name': 'spatialGridRGBhisto_8bins_2x2'},
    {'type': 'spatial', 'bins': 8, 'rows': 4, 'cols': 4, 'name': 'spatialGridRGBhisto_8bins_4x4'},
    {'type': 'spatial', 'bins': 8, 'rows': 6, 'cols': 6, 'name': 'spatialGridRGBhisto_8bins_6x6'},
    {'type': 'spatial', 'bins': 8, 'rows': 8, 'cols': 8, 'name': 'spatialGridRGBhisto_8bins_8x8'},
]

# PCA configurations
PCA_CONFIGS = [
    # PCA for globalRGBhisto_8bins
    {'base': 'globalRGBhisto_8bins', 'components': 10, 'name': 'globalRGBhisto_8bins_PCA10'},
    {'base': 'globalRGBhisto_8bins', 'components': 16, 'name': 'globalRGBhisto_8bins_PCA16'},
    {'base': 'globalRGBhisto_8bins', 'components': 18, 'name': 'globalRGBhisto_8bins_PCA18'},
    {'base': 'globalRGBhisto_8bins', 'components': 20, 'name': 'globalRGBhisto_8bins_PCA20'},
    
    # PCA for spatialGridRGBhisto_8bins_4x4
    {'base': 'spatialGridRGBhisto_8bins_4x4', 'components': 12, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA12'},
    {'base': 'spatialGridRGBhisto_8bins_4x4', 'components': 16, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA16'},
    {'base': 'spatialGridRGBhisto_8bins_4x4', 'components': 24, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA24'},
    {'base': 'spatialGridRGBhisto_8bins_4x4', 'components': 32, 'name': 'spatialGridRGBhisto_8bins_4x4_PCA32'}
]

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
os.makedirs(os.path.join(OUT_FOLDER, PCA_MODELS_FOLDER), exist_ok=True)
for config in DESCRIPTOR_CONFIGS:
    os.makedirs(os.path.join(OUT_FOLDER, config['name']), exist_ok=True)
print("✅ Output directories created")
print("="*60)

print("="*60)
print("Computing Feature Descriptors")
print("="*60)
print(f"Total configurations: {len(DESCRIPTOR_CONFIGS)}")
for i, config in enumerate(DESCRIPTOR_CONFIGS, 1):
    if config['type'] == 'global':
        print(f"  {i}. {config['name']} ({config['bins']} bins)")
    else:
        print(f"  {i}. {config['name']} ({config['bins']} bins, {config['rows']}x{config['cols']} grid)")
print("="*60)

# Iterate through all BMP files in the dataset folder
image_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith(".bmp")]
total_images = len(image_files)
image_filenames = []

# Store descriptors for each configuration
all_descriptors = {config['name']: [] for config in DESCRIPTOR_CONFIGS}

for idx, filename in enumerate(image_files, 1):
    print(f"[{idx}/{total_images}] Processing {filename}")
    
    img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
    img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
    
    # Compute each descriptor configuration
    for config in DESCRIPTOR_CONFIGS:
        if config['type'] == 'global':
            descriptor = extractRGBHistogram(img, num_bins=config['bins'])
            if idx == 1:  # Print dimensions for first image only
                print(f"  {config['name']}: descriptor shape = {descriptor.shape}, "
                      f"dimension = {descriptor.size} (num_bins={config['bins']})")
        else:  # spatial
            descriptor = extractSpatialGrid(img, grid_rows=config['rows'], 
                                          grid_cols=config['cols'], num_bins=config['bins'])
            if idx == 1:  # Print dimensions for first image only
                print(f"  {config['name']}: descriptor shape = {descriptor.shape}, "
                      f"dimension = {descriptor.size} (grid={config['rows']}x{config['cols']}, num_bins={config['bins']})")
        
        # Save descriptor
        out_path = os.path.join(OUT_FOLDER, config['name'], filename.replace('.bmp', '.mat'))
        sio.savemat(out_path, {'F': descriptor})
        
        # Store for PCA computation
        all_descriptors[config['name']].append(descriptor.flatten())
    
    image_filenames.append(filename)

print("\n" + "="*60)
print("Descriptor computation complete!")
for config in DESCRIPTOR_CONFIGS:
    print(f"✅ {config['name']}: {total_images} descriptors saved")
print("="*60)

# ============================================================================
# PCA Computation and Projection
# ============================================================================

if PCA_CONFIGS:
    print("\n" + "="*60)
    print("COMPUTING PCA MODELS")
    print("="*60)
    
    # Convert descriptor lists to numpy arrays
    for config_name in all_descriptors:
        all_descriptors[config_name] = np.array(all_descriptors[config_name])
    
    # Compute PCA for each configuration
    for pca_config in PCA_CONFIGS:
        base_name = pca_config['base']
        n_components = pca_config['components']
        pca_name = pca_config['name']
        
        if base_name not in all_descriptors:
            print(f"⚠️  Warning: Base descriptor '{base_name}' not found. Skipping PCA.")
            continue
        
        descriptors = all_descriptors[base_name]
        
        print(f"\n{'='*60}")
        print(f"COMPUTING PCA FOR: {base_name}")
        print(f"Target: {pca_name} ({n_components} components)")
        print("="*60)
        
        print(f"Input descriptor shape: {descriptors.shape}")
        print(f"  - {descriptors.shape[0]} images × {descriptors.shape[1]} dimensions")
        
        # Compute PCA
        pca_model = compute_pca(descriptors, normalize=True)
        
        # Save PCA model
        pca_model_path = os.path.join(OUT_FOLDER, PCA_MODELS_FOLDER, f'{base_name}_pca.mat')
        save_pca_model(pca_model, pca_model_path)
        print(f"✅ PCA model saved to: {pca_model_path}")
        
        # Project descriptors
        projected, actual_n_comp = project_descriptors(descriptors, pca_model, n_components=n_components)
        
        print(f"✅ Projected descriptor shape: {projected.shape}")
        var_explained = pca_model['cumulative_variance_ratio'][actual_n_comp - 1] * 100
        print(f"✅ Variance preserved: {var_explained:.2f}%")
        
        # Create output folder
        pca_folder_path = os.path.join(OUT_FOLDER, pca_name)
        os.makedirs(pca_folder_path, exist_ok=True)
        
        # Save projected descriptors
        for i, filename in enumerate(image_filenames):
            projected_desc = projected[i].reshape(1, -1)
            pca_out_path = os.path.join(pca_folder_path, filename.replace('.bmp', '.mat'))
            sio.savemat(pca_out_path, {'F': projected_desc})
        
        print(f"✅ Saved {len(image_filenames)} projected descriptors to: {pca_name}/")
    
    print("\n" + "="*60)
    print("PCA COMPUTATION COMPLETE!")
    print("="*60)

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\n✅ All descriptors computed and saved:")
for config in DESCRIPTOR_CONFIGS:
    print(f"   - {config['name']}")
for pca_config in PCA_CONFIGS:
    print(f"   - {pca_config['name']}")
print("="*60)

