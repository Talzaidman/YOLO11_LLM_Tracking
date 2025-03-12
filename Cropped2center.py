import cv2
cam_width = 1000
cam_height = 1000

def cropped2center(cropped):
    height, width, channels = cropped.shape
    if height < cam_height or width < cam_width:
        # Calculate padding
        top = (cam_height - height) // 2
        bottom = cam_height - height - top
        left = (cam_width - width) // 2
        right = cam_width - width - left
        # Add padding
        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
    return cropped