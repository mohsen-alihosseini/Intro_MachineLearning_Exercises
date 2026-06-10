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
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


    # Step 2: Binarize the resized image with Otsu thresholding.
    # Allowed: cv2.threshold with Otsu.
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Find the bounding box of the foreground with NumPy only.
    # Hint: The symbols are dark, the background is bright.
    # Allowed: NumPy operations such as argwhere, min, max, slicing.
    # Not allowed: cv2.findContours, connectedComponents, or similar high-level localization helpers.
    foreground_pts = np.argwhere(thresh == 255)
    
    if len(foreground_pts) == 0:
        # Fallback if the image is completely blank/empty
        return np.full((size, size), 255, dtype=np.uint8)

    ymin,xmin=foreground_pts.min(axis=0)
    ymax,xmax=foreground_pts.max(axis=0)

    # Step 4: Crop the grayscale region of interest from the resized image.
    # Use NumPy slicing.
    cropped=resized[ymin:ymax+1, xmin:xmax+1]

    # Step 5: Resize the cropped region such that it fits into half the canvas.
    # Allowed: cv2.resize.
    half_size = size // 2
    h, w = cropped.shape
    scale=min(half_size / h, half_size / w)
    new_w=max(1, int(w * scale))
    new_h=max(1, int(h * scale))
    
    resized_symbol=cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)


    # Step 6: Place the resized symbol in the center of a blank canvas.
    # Use NumPy indexing and array assignment for centering.
    canvas = np.full((size, size), 255, dtype=np.uint8)
    start_y = (size - new_h) // 2
    start_x = (size - new_w) // 2
    
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_symbol

    return canvas
