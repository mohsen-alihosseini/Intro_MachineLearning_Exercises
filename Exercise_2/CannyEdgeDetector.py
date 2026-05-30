import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
        # Create Gaussian kernel
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    
    total = 0
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            total += kernel[i, j]
    
    # Normalize so kernel sums to 1
    kernel = kernel / total
    
    # Apply convolution using scipy's convolve
    filtered = convolve(img_in.astype(np.float64), kernel, mode='constant', cval=0.0)
    
    # Convert back to int as required by tests
    filtered = filtered.astype(int)
    
    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
 # Sobel kernels (3x3)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    # Apply convolution
    gx = convolve(img_in.astype(np.float64), sobel_x, mode='constant', cval=0.0)
    gy = convolve(img_in.astype(np.float64), sobel_y, mode='constant', cval=0.0)
    
    # FIX: Negate gy to match test expectations because convolution flips the kernel
    gy = -gy
    
    # Convert back to int as required by tests
    gx = gx.astype(int)
    gy = gy.astype(int)
    
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # Calculate gradient magnitude
    g = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)
    
    # Calculate gradient direction (in radians)
    theta = np.arctan2(gy.astype(np.float64), gx.astype(np.float64))
    
    # Convert magnitude to int as required by tests
    g = g.astype(int)
    
    return g, theta
    


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # Convert radians to degrees
    degrees = np.degrees(angle)
    
    # Normalize to [0, 180)
    degrees = degrees % 180
    
    # Find nearest angle: 0, 45, 90, 135
    if degrees < 22.5:
        return 0
    elif degrees < 67.5:
        return 45
    elif degrees < 112.5:
        return 90
    elif degrees < 157.5:
        return 135
    else:
        return 0  # 180° is same as 0°


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    # Initialize output with zeros
    max_sup = np.zeros_like(g)
    
    # Get image dimensions
    h, w = g.shape
    
    # For each pixel, apply non-maximum suppression
    for i in range(1, h - 1):  # Skip border pixels
        for j in range(1, w - 1):
            # Get the quantized angle (0, 45, 90, 135)
            angle = convertAngle(theta[i, j])
            
            # Compare with neighbors based on gradient direction
            if angle == 0:  # Horizontal gradient - compare left and right
                if g[i, j] >= g[i, j - 1] and g[i, j] >= g[i, j + 1]:
                    max_sup[i, j] = g[i, j]
                else:
                    max_sup[i, j] = 0
                    
            elif angle == 45:  # Diagonal (top-right to bottom-left)
                if g[i, j] >= g[i - 1, j + 1] and g[i, j] >= g[i + 1, j - 1]:
                    max_sup[i, j] = g[i, j]
                else:
                    max_sup[i, j] = 0
                    
            elif angle == 90:  # Vertical gradient - compare top and bottom
                if g[i, j] >= g[i - 1, j] and g[i, j] >= g[i + 1, j]:
                    max_sup[i, j] = g[i, j]
                else:
                    max_sup[i, j] = 0
                    
            elif angle == 135:  # Diagonal (top-left to bottom-right)
                if g[i, j] >= g[i - 1, j - 1] and g[i, j] >= g[i + 1, j + 1]:
                    max_sup[i, j] = g[i, j]
                else:
                    max_sup[i, j] = 0
    
    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    h, w = max_sup.shape
    
    # Step 1: Classify pixels
    # 0 = weak (between low and high), 1 = strong (above high), 2 = suppressed (below low)
    classified = np.zeros((h, w), dtype=int)
    
    for i in range(h):
        for j in range(w):
            if max_sup[i, j] > t_high:
                classified[i, j] = 2  # Strong edge
            elif max_sup[i, j] > t_low:
                classified[i, j] = 1  # Weak edge
            else:
                classified[i, j] = 0  # Suppressed
    
    # Step 2: Hysteresis - strong edges keep weak neighbors
    result = np.zeros((h, w), dtype=np.uint8)
    
    # For each pixel, if it's strong, set to 255 and also set weak neighbors to 255
    for i in range(h):
        for j in range(w):
            if classified[i, j] == 2:  # Strong edge
                result[i, j] = 255
                
                # Check all 8 neighbors (if within bounds)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if classified[ni, nj] == 1:  # Weak neighbor
                                result[ni, nj] = 255
    
    return result


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
