# CV_EEE3032 – Visual Search System (EEE3032 Coursework)

A complete Python implementation of an image visual search system for the MSRC_ObjCategImageDatabase_v2 dataset. It includes descriptor computation (Global RGB Histogram and Spatial Grid), optional PCA dimensionality reduction, multiple distance metrics, visual retrieval demos, and evaluation utilities (PR curves, mAP, confusion matrix).

## Project Structure

- `cvpr_computedescriptors.py`: Iterates over the dataset and computes descriptors.
  - Global RGB histograms with 4/8/16/32 bins
  - Spatial grid RGB histograms (2x2, 4x4, 6x6, 8x8)
  - Saves `.mat` feature files to `descriptors/<descriptor_name>/`
  - Generates PCA models and projected descriptors for configured variants
- `cvpr_evaluate.py`: Evaluation framework.
  - Loads descriptors, computes distances, evaluates all queries
  - Precision@K, Recall@K, AP, mAP
  - Mean PR curve (non-interpolated averaging), confusion matrix
- `cvpr_visualsearch.py`: Interactive visual search and montage visualization.
  - Dynamically loads descriptors, lets you choose distance metric and `top_k`
  - Displays query and ranked results in a montage
- `cvpr_compare.py`: Distance metrics.
  - Euclidean, Manhattan, Minkowski, Chebyshev, Mahalanobis, Chi-square
- `cvpr_pca.py`: PCA utilities.
  - Compute, save, and load PCA models; project descriptors
- `extractRandom.py`: Feature extraction functions used by descriptor pipeline.
  - `extractRGBHistogram(img, num_bins)` and `extractSpatialGrid(img, grid_rows, grid_cols, num_bins)`

Folders:
- `descriptors/`: All generated feature descriptors and `pca_models/`.
- `experiments/`: All generated plots (PR curves, confusion matrices, visualizations).
- `MSRC_ObjCategImageDatabase_v2/`: Dataset (Images + GroundTruth).
- `scripts/`: Standalone scripts for experiments and visualizations:
  - `experiment_gch.py`: Compare GCH quantization and distance metrics
  - `experiment_spatial_grid.py`: Compare spatial grids and metrics
  - `experiment_pca.py`: Evaluate PCA descriptors (Mahalanobis)
  - `visualize_dataset.py`: Example images + class-count table
  - `visualize_gch.py`, `visualize_spatial_grid.py`: Descriptor visualizations
  - `visualize_pca.py`: PCA effect visualization
  - `visualize_retrieval_results.py`: Retrieval montages for selected configs

## Requirements

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows (PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
```

If OpenCV GUI backends are not needed, the current `opencv-python` wheel is sufficient.

## Paths

- Dataset images folder (used across scripts):
  - `./MSRC_ObjCategImageDatabase_v2/Images`
- Descriptors output folder:
  - `./descriptors`
- Experiments output folder:
  - `./experiments`

Ensure these folders exist and contain the expected data before running scripts.

## Typical Workflow

1) Compute descriptors
```bash
python cvpr_computedescriptors.py
```
This will populate `descriptors/` with:
- Global RGB histogram descriptors: `globalRGBhisto_<bins>bins`
- Spatial grid descriptors: `spatialGridRGBhisto_8bins_<grid>`
- PCA models and projected descriptors as configured

2) Evaluate a descriptor
```bash
python cvpr_evaluate.py
```
Follow prompts to pick a descriptor and distance metric. Outputs:
- `Mean Precision@10`, `Mean Recall@10`, `mAP`
- PR curve (green line), confusion matrix (top-1), saved under `experiments/<descriptor_metric>/`

3) Run experiment scripts (optional)
- GCH comparison:
```bash
python scripts/experiment_gch.py
```
- Spatial grid comparison:
```bash
python scripts/experiment_spatial_grid.py
```
- PCA evaluation (Mahalanobis):
```bash
python scripts/experiment_pca.py
```
Outputs are saved under `experiments/` in their respective subfolders.

4) Visualize retrieval results (optional)
```bash
python scripts/visualize_retrieval_results.py
```
Generates montages for predefined descriptor-distance combos under `experiments/retrieval_visualizations/`.

5) Visualize dataset classes (optional)
```bash
python scripts/visualize_dataset.py
```
Saves two PNGs: example images from 5 classes and a class-count table.

## Notes
- The MSRC dataset should contain 591 images across 20 classes. Filenames are formatted like `12_33_s.bmp`, where `12` is the class ID.
- Large artifacts under `descriptors/` and `experiments/` are intentionally not tracked by version control.

