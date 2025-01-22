import cv2


cam_width, cam_height = 640, 480


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
                # Ensure bounding box coordinates are within frame dimensions
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Crop the frame based on the bounding box
                cropped = frame[y1:y2, x1:x2]
                height, width, channels = cropped.shape
                if height < cam_height or width < cam_width:
                    # Calculate padding
                    top = (cam_height - height) // 2
                    bottom = cam_height - height - top
                    left = (cam_width - width) // 2
                    right = cam_width - width - left
                    # Add padding
                    cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # Display the cropped image
                cv2.imshow("Full Frame", cropped)

                target_found = True
                return results, x1, y1, x2, y2, target_found
    x1, y1, x2, y2 = None, None, None, None
    return results, x1, y1, x2, y2, target_found
