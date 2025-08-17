import cv2
import numpy as np
import time
from ultralytics import YOLO


def main():
    """Main function to capture video and detect objects with YOLOv8"""
    # Load YOLOv8 model
    model = YOLO("yolov8x.pt")

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set frame dimensions (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting object detection. Press 'q' to quit.")

    # Get the names of the classes
    names = model.names

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform detection with YOLOv8
        results = model(frame)

        # Process and display the results
        result_frame = frame.copy()

        # Get detection results
        boxes = results[0].boxes

        # Draw bounding boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Only show detections with confidence > 0.5
            if conf > 0.5:
                # Generate a color based on class
                color = (int(hash(names[cls]) % 256),
                         int(hash(names[cls] * 2) % 256),
                         int(hash(names[cls] * 3) % 256))

                # Draw rectangle
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{names[cls]} {conf:.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



        # Display result
        cv2.imshow("YOLOv8 Object Detection", result_frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()