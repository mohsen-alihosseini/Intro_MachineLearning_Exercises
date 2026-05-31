from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    k = np.zeros((ksize, ksize), dtype=np.float64)
    center = ksize // 2
    # fill kernel with Gaussian values
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            k[j, i] = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    k /= np.sum(k)
    return k


def slow_convolve(arr, k):
    h, w = arr.shape
    kh, kw = k.shape
    pad_y = kh // 2
    pad_x = kw // 2

    padded = np.zeros((h + 2 * pad_y, w + 2 * pad_x), dtype=np.float64)
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = arr
    result = np.zeros((h, w), dtype=np.float64)
    # convolution
    for i in range(h):
        for j in range(w):
            value = 0.0
            for u in range(kh):
                for v in range(kw):
                    value += k[u, v] * padded[i + u, j + v]
            result[i, j] = value
    return result


if __name__ == '__main__':
    k = make_kernel(3, 1)  # todo: find better parameters

    # TODO: chose the image you prefer
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    blurred = slow_convolve(im, k)
    unsharp_mask = im - blurred
    # Add mask back to original image
    result = im + unsharp_mask
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    Image.fromarray(result).save("sharpened.png")
