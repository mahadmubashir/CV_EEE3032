
import numpy as np

def extractRandom(img):
    # # Generate a random row vector with 30 elements
    # F = np.random.rand(1, 30)
    # # Returns a row [rand rand .... rand] representing an image descriptor
    # # computed from image 'img'
    # # Note img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
    """
    Extract a global RGB histogram descriptor from an image.
    
    This function computes a color histogram that represents the distribution
    of red, green, and blue pixel values in the image. This is a simple
    but effective wa2y to capture the overall color characteristics of an image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image as a 3D array (height, width, channels)
        Expected to be normalized RGB image with values in range [0,1]
    
    Returns:
    --------
    F : numpy.ndarray
        Feature descriptor as a 1D array representing the RGB histogram
    """
    
    # Step 1: Define histogram parameters
    # We'll use 8 bins per color channel (8^3 = 512 total bins)
    # This gives us a good balance between detail and computational efficiency
    num_bins = 8
    
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
    # This creates a 24-dimensional feature vector (8+8+8)
    F = np.concatenate([hist_r, hist_g, hist_b])
    
    # Step 6: Reshape to row vector format (1, 24) for consistency with original code
    F = F.reshape(1, -1)
    
    return F
