from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt



def make_kernel(ksize, sigma):
    # return  # implement the Gaussian kernel here
    #Create a Gaussian kernel of size ksize x ksize with given sigma
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    
    total = 0
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            # Gaussian formula
            kernel[i, j] = (1.0 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            total += kernel[i, j]
    
    # Normalize so that the kernel sums to 1
    kernel = kernel / total
    return kernel


def slow_convolve(arr, k):
    # return  # implement the convolution with padding here
    if len(arr.shape) == 3:
        # Process each channel separately
        result_channels = []
        for channel in range(arr.shape[2]):
            result_channels.append(slow_convolve_2d(arr[:, :, channel], k))
        return np.stack(result_channels, axis=2)
    else:
        # Handle grayscale (2D arrays)
        return slow_convolve_2d(arr, k)
    
#here created convolv for each channel that used in slow channel
def slow_convolve_2d(arr, k):
    """
    Convolve a 2D array arr with a kernel k using zero-padding.
    Output size is same as input size.
    """
    # Get dimensions
    img_h, img_w = arr.shape
    k_h, k_w = k.shape
    
    # Calculate padding sizes
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # Pad the input image with zeros
    padded = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
    padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = arr
    
    # Create output array
    output = np.zeros((img_h, img_w))
    
    # Flip the kernel (convolution requires kernel flip)
    k_flipped = np.flip(k)
    
    # Perform convolution
    for i in range(img_h):
        for j in range(img_w):
            # Extract the region from padded image
            region = padded[i:i + k_h, j:j + k_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * k_flipped)
    
    return output

    


if __name__ == '__main__':
    # k = make_kernel(3, 1)   # todo: find better parameters
    # kernel size 5 with sigma 1.0 (good for sharpening)
    k = make_kernel(10, 1.0)
    
    # TODO: chose the image you prefer
    im = np.array(Image.open('./data/input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    # Step 1: Blur the image (convolve with Gaussian kernel)
    im_float = im.astype(np.float64)

    # Step 1: Blur the image (convolve with Gaussian kernel)
    blurred = slow_convolve(im_float, k)

    # Step 2: Compute unsharp mask = input - blurred
    unsharp_mask = im_float - blurred

    # Step 3: Add the mask back to the original image
    # result = input + (input - blurred) = 2*input - blurred
    result = im_float + unsharp_mask

    # Step 4: Clip values to the range [0, 255]
    result = np.clip(result, 0, 255)

    # Step 5: Convert the array to np.uint8
    result = result.astype(np.uint8)

    # Step 6: Save the result
    result_img = Image.fromarray(result)
    result_img.save('sharpened_result.png')




    # Display both original and sharpened images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(im)
    axes[0].set_title(f'Original Image\n{im.shape}')
    axes[0].axis('off')
    
    # Sharpened image
    axes[1].imshow(result)
    axes[1].set_title(f'Sharpened Image\nksize={k}, sigma={k}')
    axes[1].axis('off')
    
    # Difference (magnified for visibility)
    diff = np.abs(im.astype(np.float64) - result.astype(np.float64))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Difference (edges enhanced)\nMax diff: {diff.max():.1f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSharpening complete! Result saved as 'sharpened_result.png'")
    print(f"Original image range: [{im.min()}, {im.max()}]")
    print(f"Sharpened image range: [{result.min()}, {result.max()}]")