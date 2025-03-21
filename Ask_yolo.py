import cv2
from Cropped2center import cropped2center

cam_width, cam_height = 1388, 1038

def asking_yolo(model, frame, chosen_class):
    target_found = False
    height, width, channels = frame.shape
    results = model(frame, imgsz=max(height, width))
    for result in results:  # Iterate over results
        for box in result.boxes.data:  # Iterate over detected boxes
            # Extract bounding box information
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()  # Convert to NumPy array
            # Check if the detected class matches the target class
            if result.names[int(cls)] == chosen_class:
                target_found = True
                return results, x1, y1, x2, y2, target_found
    x1, y1, x2, y2 = 0, 0, cam_width, cam_height
    return results, x1, y1, x2, y2, target_found
