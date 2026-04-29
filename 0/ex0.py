import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"
# IMAGE_PATH: str = "data/gray.jpg"


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        # ToDo: Save the colour type and load the image using CV2.
        self._colour_type: str = colour_type
        self._image: np.ndarray = cv2.imread(image_path)


        

    def get_image_data(self) -> tuple[np.ndarray, str]:
        """
        Return the image data (image and colour scheme).

        Returns:
            tuple(np.ndarray, str): Loaded image and current colour scheme.
        """
        return self._image, self._colour_type

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # ToDo: Show the image depending on the colour type.
        if self._colour_type == "Gray":
            plt.imshow(self._image, cmap='gray')
        else:
            plt.imshow(self._image)
        plt.axis('off')
        plt.show()
        

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        # ToDo: Save the image.
        if self._colour_type == "RGB":
            # Convert back to BGR for cv2 saving
            image_to_save = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(total_image_path, image_to_save)
        else:
            cv2.imwrite(total_image_path, self._image)

    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # ToDo: Perform the colour conversion.
        self._image = self._image[:, :, [2, 1, 0]]

        # ToDo: Update the colour type.
        if self._colour_type == "RGB":
            self._colour_type = "BGR"
        else:
            self._colour_type = "RGB"

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
        clip_min (int): Minimum image colour intensity.
        clip_max (int): Maximum image colour intensity.
        """
        # ToDo: Clip the image values to the given values.
        self._image[self._image < clip_min] = clip_min
        self._image[self._image > clip_max] = clip_max

    def convert_to_grayscale(self, method: str = "lightness"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")

        if method == "lightness":
            # Lightness = (max(R,G,B) + min(R,G,B)) / 2
            max_val = np.max(self._image, axis=2)
            min_val = np.min(self._image, axis=2)
            self._image = ((max_val + min_val) / 2).astype(np.uint8)

        if method == "average":
          # Average = (R + G + B) / 3
          self._image = np.mean(self._image, axis=2).astype(np.uint8)

        if method == "luminosity":
            if self._colour_type == "RGB":
                self._image = (0.21 * self._image[:, :, 0] + 
                            0.72 * self._image[:, :, 1] + 
                            0.07 * self._image[:, :, 2]).astype(np.uint8)
            else:  # BGR
                self._image = (0.07 * self._image[:, :, 0] +  # B channel (0.07)
                            0.72 * self._image[:, :, 1] +  # G channel (0.72)
                            0.21 * self._image[:, :, 2]).astype(np.uint8)  # R channel (0.21)

        # ToDo: Update the colour type.
        self._colour_type = "Gray"

    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        # ToDo: Rotate the image depending on the given rotation value.
        rotations = (degrees // 90) % 4  # Normalize to 0-3 rotations

        # rotations == 0: no change
        if rotations == 1:  # 90 degrees clockwise
        # Rotate 90° clockwise = transpose then reverse rows
            self._image = np.transpose(self._image, (1, 0, 2)) if len(self._image.shape) == 3 else np.transpose(self._image)
            self._image = self._image[::-1, ...]
        elif rotations == 2:  # 180 degrees
            self._image = self._image[::-1, ::-1]
        elif rotations == 3:  # 270 degrees clockwise (or 90 counter-clockwise)
            self._image = np.transpose(self._image, (1, 0, 2)) if len(self._image.shape) == 3 else np.transpose(self._image)
            self._image = self._image[:, ::-1]
        # rotations == 0: no change

    def flip_image(self, flip_value: int):
        """
        Flip an image either horizontally (0), vertically (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # ToDo: Flip the image using indexing.
        if flip_value == 0:  # Horizontal flip
            self._image = self._image[:, ::-1]
        elif flip_value == 1:  # Vertical flip
            self._image = self._image[::-1, :]
        elif flip_value == 2:  # Both flips
            self._image = self._image[::-1, ::-1]

    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        # ToDo: Check that the given parameters are valid!
        if new_height <= 0 or new_width <= 0:
            raise ValueError("Height and width must be positive!")
        if new_height > self._image.shape[0] or new_width > self._image.shape[1]:
            raise ValueError("Given Cropped dimensions are bigger than original image!!!")


        # ToDo: Crop the image around the center.
        orig_height, orig_width = self._image.shape[0], self._image.shape[1]
        # Calculate starting hight width based on new hight width
        start_y = (orig_height - new_height) // 2
        start_x = (orig_width - new_width) // 2
    
        # Crop the image
        self._image = self._image[start_y:start_y + new_height, start_x:start_x + new_width]


    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # ToDo: Resize the image. Research the available options in CV2.
        self._image = cv2.resize(self._image, (new_width, new_height))

if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
