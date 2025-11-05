
import numpy as np

def extractRGBHistogram(img, num_bins: int = 4):
    # # Generate a random row vector with 30 elements
    # F = np.random.rand(1, 30)
    # # Returns a row [rand rand .... rand] representing an image descriptor
    # # computed from image 'img'
    # # Note img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
    """
    Extract a global RGB histogram descriptor from an image.
    
    This function computes a color histogram that represents the distribution
    of red, green, and blue pixel values in the image. This is a simple
    but effective way to capture the overall color characteristics of an image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image as a 3D array (height, width, channels)
        Expected to be normalized RGB image with values in range [0,1]
    num_bins : int
        Number of bins per color channel (default: 32)
    
    Returns:
    --------
    F : numpy.ndarray
        Feature descriptor as a row vector of shape (1, 3*num_bins)
    """
    
    # Step 1: Define histogram parameters
    # Use 'num_bins' per color channel; configurable for experiments
    if num_bins < 1:
        num_bins = 1
    
    # Step 2: Convert image to 0-255 range for histogram computation
    # OpenCV expects integer values for histogram computation
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Step 3: Compute histogram for each color channel
    # We'll compute separate histograms for R, G, B channels
    hist_r = np.histogram(img_uint8[:, :, 2], bins=num_bins, range=(0, 256))[0]  # Red channel
    hist_g = np.histogram(img_uint8[:, :, 1], bins=num_bins, range=(0, 256))[0]  # Green channel  
    hist_b = np.histogram(img_uint8[:, :, 0], bins=num_bins, range=(0, 256))[0]  # Blue channel
    
    # Step 4: Normalize histograms to make them comparable across images
    # This ensures that images of different sizes have comparable histograms
    hist_r = hist_r.astype(np.float64) / np.sum(hist_r)
    hist_g = hist_g.astype(np.float64) / np.sum(hist_g)
    hist_b = hist_b.astype(np.float64) / np.sum(hist_b)
    
    # Step 5: Concatenate all histograms into a single feature vector
    # Length is 3 * num_bins
    F = np.concatenate([hist_r, hist_g, hist_b])
    
    # Step 6: Reshape to row vector format (1, 3*num_bins)
    F = F.reshape(1, -1)
    
    return F


def extractSpatialGrid(img, grid_rows=3, grid_cols=3, num_bins=8):
    """
    Extract a spatial grid RGB histogram descriptor from an image.
    
    This function divides the image into a grid of cells and computes separate
    RGB histograms for each cell. This captures both color distribution and
    spatial layout information, making it more discriminative than global histograms.
    
    The image is divided into grid_rows x grid_cols cells, and for each cell,
    a separate RGB histogram is computed. All histograms are concatenated to
    form the final descriptor.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image as a 3D array (height, width, channels)
        Expected to be normalized RGB image with values in range [0,1]
    grid_rows : int
        Number of rows in the spatial grid (default: 3)
    grid_cols : int
        Number of columns in the spatial grid (default: 3)
    num_bins : int
        Number of bins per color channel for each grid cell (default: 8)
    
    Returns:
    --------
    F : numpy.ndarray
        Feature descriptor as a row vector of shape (1, grid_rows*grid_cols*3*num_bins)
        Features are ordered row-wise: cell[0,0], cell[0,1], ..., cell[grid_rows-1,grid_cols-1]
    """
    # Validate inputs
    if grid_rows < 1:
        grid_rows = 1
    if grid_cols < 1:
        grid_cols = 1
    if num_bins < 1:
        num_bins = 1
    
    # Convert image to 0-255 range for histogram computation
    img_uint8 = (img * 255).astype(np.uint8)
    
    height, width = img_uint8.shape[:2]
    
    # Compute cell dimensions
    cell_height = height // grid_rows
    cell_width = width // grid_cols
    
    # Store histograms for all grid cells
    all_histograms = []
    
    # Iterate through each grid cell
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Compute cell boundaries
            y_start = row * cell_height
            y_end = (row + 1) * cell_height if row < grid_rows - 1 else height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width if col < grid_cols - 1 else width
            
            # Extract grid cell
            cell = img_uint8[y_start:y_end, x_start:x_end, :]
            
            # Skip if cell is empty
            if cell.size == 0:
                # Create zero histogram if cell is empty
                zero_hist = np.zeros(num_bins, dtype=np.float64)
                all_histograms.extend([zero_hist, zero_hist, zero_hist])
                continue
            
            # Compute histogram for each color channel in this cell
            hist_r = np.histogram(cell[:, :, 2], bins=num_bins, range=(0, 256))[0]  # Red channel
            hist_g = np.histogram(cell[:, :, 1], bins=num_bins, range=(0, 256))[0]  # Green channel
            hist_b = np.histogram(cell[:, :, 0], bins=num_bins, range=(0, 256))[0]  # Blue channel
            
            # Normalize histograms for this cell
            hist_r = hist_r.astype(np.float64) / max(np.sum(hist_r), 1)
            hist_g = hist_g.astype(np.float64) / max(np.sum(hist_g), 1)
            hist_b = hist_b.astype(np.float64) / max(np.sum(hist_b), 1)
            
            # Append cell histograms (R, G, B order)
            all_histograms.extend([hist_r, hist_g, hist_b])
    
    # Concatenate all grid cell histograms into single feature vector
    F = np.concatenate(all_histograms)
    
    # Reshape to row vector format
    F = F.reshape(1, -1)
    
    return F
