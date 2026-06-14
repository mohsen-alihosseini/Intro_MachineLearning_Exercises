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

    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    grad = np.sqrt(x * x + y * y).astype(np.float32)

    rad = np.arctan2(y, x)  # range (-pi, pi]
    deg = np.degrees(rad)  # range (-180, 180]

    orient = np.mod(deg, 180.0).astype(np.float32)

    return grad, orient


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    H, W = magnitude.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    # Crop to integer number of cells
    Hc = n_cells_y * cell_size
    Wc = n_cells_x * cell_size
    mag = magnitude[:Hc, :Wc]
    ang = orientation[:Hc, :Wc]

    # Prepare histogram array
    hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)

    # Bin centers and width
    bin_width = 180.0 // num_bins  # degrees per bin
    # map orientations [0,180) to bin_idx in [0, n_bins)
    bin_idx = ang // bin_width  # float index

    # Split image into cells and accumulate
    for cy in range(n_cells_y):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(n_cells_x):
            x0 = cx * cell_size
            x1 = x0 + cell_size

            cell_mag = mag[y0:y1, x0:x1].ravel()
            cell_binf = bin_idx[y0:y1, x0:x1].ravel()

            # For each pixel in the cell, distribute its magnitude to two nearest bins
            # lower bin index
            lower = np.floor(cell_binf).astype(np.int32)  # in [0, n_bins-1]
            upper = (lower + 1) % num_bins
            # fractional part
            frac = cell_binf - lower

            # accumulate
            # For speed use bincount per bin with weights for lower and upper parts
            # accumulate lower contributions
            if cell_mag.size == 0:
                continue
            lower_contrib = (1.0 - frac) * cell_mag
            upper_contrib = frac * cell_mag

            # accumulate into hist[cy, cx, :]
            # use np.add.at for repeated indices
            np.add.at(hist[cy, cx], lower, lower_contrib)
            np.add.at(hist[cy, cx], upper, upper_contrib)
    return hist


def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # magnitude and unsigned orientation [0,180)
    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    ang_rad = np.arctan2(gy, gx)
    ang_deg = np.degrees(ang_rad)
    orientation = np.mod(ang_deg, 180.0).astype(np.float32)

    H, W = magnitude.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    Hc = n_cells_y * cell_size
    Wc = n_cells_x * cell_size
    mag = magnitude[:Hc, :Wc]
    ang = orientation[:Hc, :Wc]

    # Build per-cell histograms
    hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    bin_width = 180.0 / num_bins
    bin_idx = ang / bin_width  # float index in [0, num_bins)

    for cy in range(n_cells_y):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(n_cells_x):
            x0 = cx * cell_size
            x1 = x0 + cell_size

            cell_mag = mag[y0:y1, x0:x1].ravel()
            cell_binf = bin_idx[y0:y1, x0:x1].ravel()

            if cell_mag.size == 0:
                continue

            lower = np.floor(cell_binf).astype(np.int32)
            upper = (lower + 1) % num_bins
            frac = cell_binf - lower

            lower_contrib = (1.0 - frac) * cell_mag
            upper_contrib = frac * cell_mag

            np.add.at(hist[cy, cx], lower, lower_contrib)
            np.add.at(hist[cy, cx], upper, upper_contrib)

    # Build blocks, normalize with L2-Hys, concatenate
    by = bx = block_size
    sy = sx = 1  # stride in cells (overlap = block_size-1 cells)
    n_blocks_y = 1 + (n_cells_y - by) // sy
    n_blocks_x = 1 + (n_cells_x - bx) // sx
    if n_blocks_y <= 0 or n_blocks_x <= 0:
        return np.array([], dtype=np.float32)

    block_vec_len = by * bx * num_bins
    blocks = np.zeros((n_blocks_y, n_blocks_x, block_vec_len), dtype=np.float32)
    clip_val = 0.2

    for by_i in range(n_blocks_y):
        y0 = by_i * sy
        for bx_i in range(n_blocks_x):
            x0 = bx_i * sx
            block_cells = hist[y0:y0 + by, x0:x0 + bx, :]
            v = block_cells.ravel().astype(np.float32)

            # L2 normalization
            norm = np.sqrt(np.sum(v * v) + eps * eps)
            if norm > 0:
                v = v / norm
            # clip (L2-Hys)
            v = np.minimum(v, clip_val)
            # renormalize
            norm2 = np.sqrt(np.sum(v * v) + eps * eps)
            if norm2 > 0:
                v = v / norm2
            blocks[by_i, bx_i, :] = v

    descriptor = blocks.ravel().astype(np.float32)
    return descriptor
