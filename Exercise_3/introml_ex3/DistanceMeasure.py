'''
Distance measures for exercise 3.
'''

import numpy as np

# do not import more modules!
# Implement the formulas yourself using NumPy operations.
# Do not use external distance or metric libraries.


def mseDistance(imgA, imgB):
    """
    Compute the mean squared error between two equally sized grayscale images.
    """
    if imgA.shape != imgB.shape:
        raise ValueError("Images must have the same shape.")

    af = imgA.astype(np.float32)
    bf = imgB.astype(np.float32)

    diff = af - bf
    sq = diff**2
    return np.mean(sq)


def euclideanDistance(featureA, featureB):
    """
    Compute the Euclidean distance between two feature vectors.
    """
    if featureA.shape != featureB.shape:
        raise ValueError("Feature vectors must have the same shape.")

    diff = featureA - featureB
    sq = diff**2
    s = np.sum(sq)
    return float(np.sqrt(s))
