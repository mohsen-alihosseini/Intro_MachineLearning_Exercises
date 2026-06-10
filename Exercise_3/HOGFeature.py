'''
Histogram of Oriented Gradients utilities for exercise 3.
'''

import numpy as np
import cv2

# do not import more modules!
# You may use cv2.Sobel for the derivatives.
# Compute magnitudes, orientations, histogram binning, and block normalization yourself with NumPy.
# Do not use cv2.HOGDescriptor or any other ready-made HOG implementation.


def computeGradients(img):
    """
    Compute gradient magnitudes and unsigned orientations in degrees.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    # TODO: compute Sobel derivatives, magnitudes, and orientations.
    # Allowed: cv2.Sobel for the x/y derivatives and NumPy for the remaining computations.
    # Not allowed: any ready-made HOG or feature extraction implementation.
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    orientation = np.arctan2(grad_y, grad_x) * (180.0 / np.pi) # Compute angles in radians and map to degrees
    orientation = orientation % 180.0 # Normalize to unsigned range [0, 180)
    return magnitude, orientation


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    # TODO: divide the image into cells and accumulate magnitudes into bins.
    # Use NumPy indexing/loops to implement the histogram accumulation yourself.
    # Do not call a library routine that directly computes cell histograms for HOG.
    img_h, img_w = magnitude.shape
    cells_h=img_h // cell_size
    cells_w=img_w // cell_size

    histograms= np.zeros((cells_h, cells_w, num_bins), dtype=np.float32)
    bin_width= 180.0 / num_bins

    # Iterative calculation over structural cell regions
    for i in range(cells_h):
        for j in range(cells_w):
            # Isolate cell patch bounds
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ori = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # Map pixel angles directly onto histogram bin indices
            bin_indices = (cell_ori / bin_width).astype(np.int32)
            np.clip(bin_indices, 0, num_bins - 1, out=bin_indices)
            
            # Vectorized accumulation using bincount inside the local cells
            histograms[i, j, :] = np.bincount(bin_indices.ravel(), weights=cell_mag.ravel(), minlength=num_bins)

    return histograms


def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    # TODO: compute the final descriptor from your own cell histograms.
    # Implement the block normalization and concatenation yourself with NumPy.
    # Do not use cv2.HOGDescriptor, skimage.feature.hog, or similar helpers.
    
    
    magnitude, orientation = computeGradients(img) # frist Compute pixel gradients
    
    
    cell_hists = buildCellHistograms(magnitude, orientation, cell_size, num_bins) # then Accumulate spatial cell histograms
    
    cells_h, cells_w, _ = cell_hists.shape
    
    # Calculate dimensions for overlapping sliding descriptor blocks
    blocks_h = cells_h - block_size + 1
    blocks_w = cells_w - block_size + 1
    
    hog_features = []

    # Apply standard L2-norm block contrast pooling over neighborhood strides
    for i in range(blocks_h):
        for j in range(blocks_w):
            # Isolate current localized block cluster (e.g. 2x2 cells)
            block_data = cell_hists[i:i+block_size, j:j+block_size, :]
            block_vector = block_data.flatten()
            
            # L2 normalization with a micro epsilon regularizer to secure division safety
            norm = np.sqrt(np.sum(block_vector ** 2) + eps ** 2)
            normalized_block = block_vector / norm
            
            hog_features.append(normalized_block)

    # Return a unified, flattened continuous feature array
    if len(hog_features) == 0:
        return np.array([], dtype=np.float32)
        
    return np.concatenate(hog_features)
