from ultralytics import YOLO
import cv2
from text2command import get_gpt_command
from Ask_yolo import asking_yolo

# Load a model
model = YOLO("yolo11s.pt")

# Open the default camera (0 for the first camera, 1 for the second, etc.)
cap = cv2.VideoCapture(0)

chosen_class = "person"

target_found = False
padding = 30
cam_width, cam_height = 640, 480

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape

    if not ret:
        print("Error: Failed to capture frame.")
        break

    if target_found:
        target_found = False
        results, x1, y1, x2, y2, target_found = asking_yolo(model, frame[max(0, y1 - padding): min(frame.shape[0], y2 + padding),
                                                          max(0, x1 - padding):min(frame.shape[1], x2 + padding)],
                                                   chosen_class)
    if not target_found:
        target_found = False
        results, x1, y1, x2, y2, target_found = asking_yolo(model,
                                                     frame,
                                                     chosen_class)

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
