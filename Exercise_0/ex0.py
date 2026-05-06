import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


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
        cv2.imshow("Image", self._image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        # ToDo: Save the image.
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
        # das Bild ist ja [Hoehe, Breite, [Kanal 1, 2, 3 (, Alpha)]] gespeichert
        self._image = self._image[:, :, [2, 1, 0]]

        # ToDo: Update the colour type.
        if self._colour_type == "BGR":
            self._colour_type = "RGB"
        else:
            self._colour_type = "BGR"

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
        # indiziert jeden Wert bloß das Feld ist verdammt 3D, deswegen dauert's so lange, bis ich's mir vorstelle GRRRRR
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
            gray = (self._image.max(axis=2) + self._image.min(axis=2)) / 2
            self._image[:, :, 0] = gray
            self._image[:, :, 1] = gray
            self._image[:, :, 2] = gray

        if method == "average":
            gray = self._image[:, :, 0] / 3 + self._image[:, :, 1] / 3 + self._image[:, :, 2] / 3
            self._image[:, :, 0] = gray
            self._image[:, :, 1] = gray
            self._image[:, :, 2] = gray

        if method == "luminosity":
            self._image[:, :, 0] = self._image[:, :, 0] * 0.114
            self._image[:, :, 1] = self._image[:, :, 0] * 0.587
            self._image[:, :, 2] = self._image[:, :, 0] * 0.299

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
        degrees = degrees // 90 % 4
        if degrees == 1:
            self._image = np.transpose(self._image)[:, ::-1]
        elif degrees == 2:
            self._image = self._image[::-1, ::-1]
        elif degrees == 3:
            self._image = np.transpose(self._image)[::-1, :]

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
        if flip_value == 0:
            self._image = self._image[::-1, :]
        elif flip_value == 1:
            self._image = self._image[:, ::-1]
        else:
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
        # i.g. if it's 0 (around the center) it means the central pixel itself
        # if it's at least 1 it's the neighbour pixels to the central one
        if new_width < 0 or new_height < 0:
            raise ValueError("The provided new width and height must be greater at least 0 around the center pixel!")

        # ToDo: Crop the image around the center.
        center_x = self._image.shape[0] // 2
        center_y = self._image.shape[1] // 2
        if new_width == 0 and new_height == 0:
            self._image = self._image[center_x, center_y]
        else:
            self._image = self._image[center_x - new_width: center_x + new_width, center_y - new_height: center_y + new_height]

    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # ToDo: Resize the image. Research the available options in CV2.
        x = self._image.shape[0]
        y = self._image.shape[1]

        if x < 1 or y < 1:
            raise ValueError("The provided new width and height must be greater than 0!")

        if x < new_width and y < new_height:
            self._image = cv2.resize(self._image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif x > new_width and y > new_height:
            self._image = cv2.resize(self._image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            self._image = cv2.resize(self._image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
