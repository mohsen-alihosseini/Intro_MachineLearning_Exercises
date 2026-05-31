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
    kernel = np.zeros((ksize, ksize), dtype=np.float64)

    center = ksize // 2

    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[y, x] = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    # Filter image
    filtered = convolve(img_in, kernel)
    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve(img_in, sobel_x)
    gy = convolve(img_in, sobel_y)
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    g = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # convert to [0,180)
    angle = angle % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        return 0
    elif 22.5 <= angle < 67.5:
        return 45
    elif 67.5 <= angle < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    h, w = g.shape
    result = np.zeros_like(g)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            angle = np.degrees(theta[y, x])
            angle = angle % 180
            angle = convertAngle(angle)
            current = g[y, x]
            if angle == 0:
                n1 = g[y, x - 1]
                n2 = g[y, x + 1]
            elif angle == 45:
                n1 = g[y - 1, x + 1]
                n2 = g[y + 1, x - 1]
            elif angle == 90:
                n1 = g[y - 1, x]
                n2 = g[y + 1, x]
            else:  # 135
                n1 = g[y - 1, x - 1]
                n2 = g[y + 1, x + 1]
            if current >= n1 and current >= n2:
                result[y, x] = current
    return result


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
    threshimg = np.zeros((h, w), dtype=np.uint8)

    # Classification
    threshimg[max_sup <= t_low] = 0
    weak = (max_sup > t_low) & (max_sup <= t_high)
    threshimg[weak] = 1
    strong = max_sup > t_high
    threshimg[strong] = 2
    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if threshimg[y, x] == 2:
                result[y, x] = 255
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny = y + dy
                        nx = x + dx
                        if threshimg[ny, nx] >= 1:
                            result[ny, nx] = 255
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
