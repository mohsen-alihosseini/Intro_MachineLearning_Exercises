'''
Created on 20.06.2025

@author: Linda Schneider
'''

import numpy as np
import cv2

# do not import more modules!
# Use OpenCV only for basic image operations such as resizing and thresholding.
# Use NumPy for the bounding box computation and for centering the symbol.
# Do not use contour detection or connected components here.


def simpleAlignment(img, size=128):
    """
    Align a grayscale symbol by centering its foreground on a fixed canvas.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    # Step 1: Resize the input image to a fixed square size.
    # Allowed: cv2.resize.

    resized = cv2.resize(img, (size, size))

    # Step 2: Binarize the resized image with Otsu thresholding.
    # Allowed: cv2.threshold with Otsu.

    threshhold = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Step 3: Find the bounding box of the foreground with NumPy only.
    # Hint: The symbols are dark, the background is bright.
    # Allowed: NumPy operations such as argwhere, min, max, slicing.
    # Not allowed: cv2.findContours, connectedComponents, or similar high-level localization helpers.

    fg = resized < threshhold
    y, x = np.argwhere(fg)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    # Step 4: Crop the grayscale region of interest from the resized image.
    # Use NumPy slicing.

    crop = resized[ymin:ymax + 1, xmin:xmax + 1]

    # Step 5: Resize the cropped region such that it fits into half the canvas.
    # Allowed: cv2.resize.

    h_max = w_max = size // 2

    h, w = crop.shape
    # if empty
    if h == 0 or w == 0:
        return crop

    # compute scale to fit within (max_w, max_h) while preserving aspect ratio
    scale = min(1.0, min(w_max / w, h_max / h))  # <=1: don't upscale
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    # Step 6: Place the resized symbol in the center of a blank canvas.
    # Use NumPy indexing and array assignment for centering.

    canvas = np.full((h_max, w_max), background, dtype=resized.dtype)
