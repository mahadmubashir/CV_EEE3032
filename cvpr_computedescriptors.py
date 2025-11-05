import os
import numpy as np
import cv2
import scipy.io as sio
from extractRandom import extractRGBHistogram, extractSpatialGrid

DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = './descriptors'

# Define subfolders for different descriptor types
GLOBAL_RGB_SUBFOLDER = 'globalRGBhisto'
SPATIAL_GRID_SUBFOLDER = 'spatialGridRGBhisto'

# Spatial grid parameters
GRID_ROWS = 3
GRID_COLS = 3
NUM_BINS = 8  # Bins per channel for both global and spatial grid

# Ensure output directories exist
os.makedirs(os.path.join(OUT_FOLDER, GLOBAL_RGB_SUBFOLDER), exist_ok=True)
os.makedirs(os.path.join(OUT_FOLDER, SPATIAL_GRID_SUBFOLDER), exist_ok=True)

print("="*60)
print("Computing Feature Descriptors")
print("="*60)
print(f"Global RGB Histogram: {GLOBAL_RGB_SUBFOLDER}")
print(f"Spatial Grid RGB Histogram: {SPATIAL_GRID_SUBFOLDER} (grid: {GRID_ROWS}x{GRID_COLS}, bins: {NUM_BINS})")
print("="*60)

# Iterate through all BMP files in the dataset folder
image_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith(".bmp")]
total_images = len(image_files)

for idx, filename in enumerate(image_files, 1):
    print(f"[{idx}/{total_images}] Processing {filename}")
    
    img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
    img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
    
    # === Compute Global RGB Histogram Descriptor ===
    global_descriptor = extractRGBHistogram(img, num_bins=NUM_BINS)
    global_out_path = os.path.join(OUT_FOLDER, GLOBAL_RGB_SUBFOLDER, filename.replace('.bmp', '.mat'))
    sio.savemat(global_out_path, {'F': global_descriptor})
    
    # === Compute Spatial Grid RGB Histogram Descriptor ===
    spatial_descriptor = extractSpatialGrid(img, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, num_bins=NUM_BINS)
    spatial_out_path = os.path.join(OUT_FOLDER, SPATIAL_GRID_SUBFOLDER, filename.replace('.bmp', '.mat'))
    sio.savemat(spatial_out_path, {'F': spatial_descriptor})

print("\n" + "="*60)
print("Descriptor computation complete!")
print(f"Global RGB histograms saved to: {GLOBAL_RGB_SUBFOLDER}")
print(f"Spatial grid histograms saved to: {SPATIAL_GRID_SUBFOLDER}")
print("="*60)

