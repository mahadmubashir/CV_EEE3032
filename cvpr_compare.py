
import numpy as np

def cvpr_compare(F1, F2):
    # # This function should compare F1 to F2 - i.e. compute the distance
    # # between the two descriptors
    # # For now it just returns a random number
    # dst = np.random.rand()

    """
    Compute the Euclidean distance between two feature descriptors.
    
    This function calculates how similar two images are based on their
    RGB histogram descriptors. A smaller distance means more similar images.
    
    Parameters:
    -----------
    F1 : numpy.ndarray
        First feature descriptor (histogram) as a 1D array
    F2 : numpy.ndarray  
        Second feature descriptor (histogram) as a 1D array
    
    Returns:
    --------
    dst : float
        Euclidean distance between the two descriptors
        Lower values indicate more similar images
    """
    
    # Step 1: Ensure both descriptors are 1D arrays
    # This handles cases where descriptors might be 2D (1, 24) or 1D (24,)
    F1_flat = F1.flatten()
    F2_flat = F2.flatten()
    
    # Step 2: Check that both descriptors have the same length
    if len(F1_flat) != len(F2_flat):
        raise ValueError(f"Descriptor dimensions don't match: {len(F1_flat)} vs {len(F2_flat)}")
    
    # Step 3: Compute Euclidean distance
    # Formula: sqrt(sum((F1[i] - F2[i])^2))
    # This measures the "straight-line distance" between two points in feature space
    diff = F1_flat - F2_flat
    dst = np.sqrt(np.sum(diff ** 2))
    
    return dst


