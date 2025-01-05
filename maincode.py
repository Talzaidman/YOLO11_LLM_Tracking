from ultralytics import YOLO
import cv2
from text2command import get_gpt_command

# Load a model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
# results = model(r"C:\Users\zaita\Downloads\WDW-DS-Marketplace-Co-Op-Pastel-Stoney-Clover-Lane-Disney-Parks-Collection-Release-Crowds.jpg")
# results[0].show()

# Open the default camera (0 for the first camera, 1 for the second, etc.)
cap = cv2.VideoCapture(0)


chosen_class = ""

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection on the current frame
    results = model(frame)

    # Flag to check if a target class is detected
    target_found = False

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

                # Display the cropped image
                cv2.imshow("Full Frame", cropped)
                target_found = True

    if not target_found:
        # Display the full annotated frame if no target class is found
        annotated_frame = results[0].plot()
        cv2.imshow("Full Frame", annotated_frame)

    # Press 'q' to quit or 'c' to change class
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit the program
        break
    elif key == ord('c'):  # Change the class when 'c' is pressed
        new_class = input("message program: ")
        dic_json_answer = get_gpt_command(new_class)
        chosen_class = dic_json_answer["class"]
        print(f"Chosen class updated to: {chosen_class}")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()