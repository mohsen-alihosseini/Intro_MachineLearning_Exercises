import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    # The function receives a padded image (pad_image) and the current pixel of our padded image.
    # ToDo: Return the surrounding area around that center pixel with the given size (window_size).
    # ToDo: Use slicing.

    radius = window_size // 2
    # Slice the matrix around the given center row and column indices
    row_start = center_row - radius
    row_end = center_row + radius + 1
    col_start = center_col - radius
    col_end = center_col + radius + 1
    return padded_image[row_start:row_end, col_start:col_end]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    # Pad the image with zeros.
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply erosion on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    
    output = np.zeros_like(image)

    # ToDo: Iterate over the provided image and perform erosion around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    padding_size = se_size // 2
    padded = pad_image(image, padding_size)

    h, w = image.shape
    # Iterate through the original image dimensions
    for i in range(h):
        for j in range(w):
            # Map original image (i, j) to padded image coordinates
            center_row = i + padding_size
            center_col = j + padding_size
            
            # Extract neighborhood region
            region = extract_region(padded, center_row, center_col, se_size)
            
            # Check where structuring element is active (where SE is 1)
            # If all those matching spots in the image neighborhood are also 1, preserve the pixel
            if np.all(region[structuring_element == 1] == 1):
                output[i, j] = 1
                
    return output

    return output


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply dilation on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    output = np.zeros_like(image)
    padding_size = se_size // 2
    padded = pad_image(image, padding_size)

    # ToDo: Iterate over the provided image and perform dilation around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    h, w = image.shape
    # Iterate through original image dimensions
    for i in range(h):
        for j in range(w):
            center_row = i + padding_size
            center_col = j + padding_size
            
            # Extract neighborhood region
            region = extract_region(padded, center_row, center_col, se_size)
            
            # If any active pixel of SE hits a foreground pixel (1) in the region
            if np.any(region[structuring_element == 1] == 1):
                output[i, j] = 1

    return output


def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform opening (erosion followed by dilation).
    result = input_image.copy()

    # Repeat Erosion phase
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)
        
    # Repeat Dilation phase
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)

    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform closing (dilation followed by erosion).
    result = input_image.copy()
    # Repeat Dilation phase
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)
        
    # Repeat Erosion phase
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)

    return result


def load_binary(filepath: str) -> np.ndarray:
    # Load the image and binarize it again with a simple threshold.
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)  # type: ignore
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    # Save the binary image.
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Paths.
    raw_erosion_image_path = 'data/erosion_image_raw.png'
    raw_dilation_image_path = 'data/dilation_image_raw.png'
    erosion_out_path = 'data/erosion_output.png'
    dilation_out_path = 'data/dilation_output.png'

    # Load images.
    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    # Structuring element.
    SE = np.ones((5, 5), dtype=np.uint8)

    # Erosion.
    # ToDo: Perform erosion multiple times until the circles separate from each other.
    # eroded = erode_binary(erosion_input, SE)
    eroded = erosion_input.copy()
    erosion_iterations = 3  # Adjusted target iterations
    for _ in range(erosion_iterations):
        eroded = erode_binary(eroded, SE)
    save_binary(eroded, erosion_out_path)
    show_image(eroded, "Erosion Output")

    # Dilation.
    # ToDo: Perform dilation multiple times until the hole closes.
    # dilated = dilate_binary(dilation_input, SE)
    dilated = dilation_input.copy()
    dilation_iterations = 4  # Adjusted target iterations
    for _ in range(dilation_iterations):
        dilated = dilate_binary(dilated, SE)
    save_binary(dilated, dilation_out_path)
    show_image(dilated, "Dilation Output")
