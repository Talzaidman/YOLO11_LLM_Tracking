import cv2
from Cropped2center import cropped2center

cam_width, cam_height = 1000, 1000
STRIDE = 5

def asking_yolo(model, frame, chosen_class, past_coor, frame_num):
    target_found = False
    height, width, channels = frame.shape
    results = model(frame, imgsz=max(height, width))
    for result in results:  # Iterate over results
        for box in result.boxes.data:  # Iterate over detected boxes
            # Extract bounding box information
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()  # Convert to NumPy array

            # Check if the detected class matches the target class
            if result.names[int(cls)] == chosen_class:
                # Ensure bounding box coordinates are within frame dimensions
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Crop the frame based on the bounding box
                cropped = frame[y1:y2, x1:x2]
                cv2.imshow("Full Frame", cropped2center(cropped))

                target_found = True
                return results, x1, y1, x2, y2, target_found
    x1, y1, x2, y2 = None, None, None, None
    return results, x1, y1, x2, y2, target_found
