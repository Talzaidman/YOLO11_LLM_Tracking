import cv2
import numpy as np

target_width = 1388
target_height = 1038

def cropped2sameplace(full_image, cropped, dx, dy):
    """
    Overlays a cropped image onto a full image at a specified position.

    Args:
        full_image (numpy.ndarray): The full background image.
        cropped (numpy.ndarray): The cropped image to overlay.
        dx (int): X-coordinate of the top-left corner of the overlay.
        dy (int): Y-coordinate of the top-left corner of the overlay.

    Returns:
        numpy.ndarray: The resulting image with the overlay. Returns None on error.
    """
    if cropped is None:
        print("Error: Input cropped image is None.")
        return None
    cropped_height, cropped_width = cropped.shape[:2]
    cropped_roi = cropped
    if dx < 0 or dy < 0 or dx + cropped_width > full_image.shape[1] or dy + cropped_height > full_image.shape[0]:
        print("Error: Cropped image exceeds the boundaries of the full image.")
        return full_image
    full_image[dy:dy+cropped_height, dx:dx + cropped_width] = cropped_roi

    return full_image