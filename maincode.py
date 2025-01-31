from ultralytics import YOLO
import cv2
from text2command import get_gpt_command
from Ask_yolo import asking_yolo
import time
import matplotlib.pyplot as plt
import numpy as np

# Load a model
model = YOLO("yolo11n.pt")

# Open the default camera (0 for the first camera, 1 for the second, etc.)
cap = cv2.VideoCapture(0)

chosen_class = "remote"

target_found = False
padding = 30
cam_width, cam_height = 640, 480

fps_ls = []
x1, y1, x2, y2 = 0, 0, 0, 0

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    start_time = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape
    if not ret:
        print("Error: Failed to capture frame.")
        break

    if target_found:
        target_found = False
        # Store the original coordinates for coordinate transformation
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)

        # Crop the frame
        cropped_frame = frame[max(0, y1 - padding): min(frame.shape[0], y2 + padding),
                        max(0, x1 - padding): min(frame.shape[1], x2 + padding)]

        # Process the cropped frame
        results, rel_x1, rel_y1, rel_x2, rel_y2, target_found = asking_yolo(model,
                                                                            cropped_frame,
                                                                            chosen_class)

        # If object found in cropped frame, transform coordinates back to full frame
        if target_found:
            x1 = rel_x1 + crop_x1
            y1 = rel_y1 + crop_y1
            x2 = rel_x2 + crop_x1
            y2 = rel_y2 + crop_y1

    if not target_found:
        target_found = False
        results, x1, y1, x2, y2, target_found = asking_yolo(model,
                                                     frame,
                                                     chosen_class)
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

    end_time = time.time()
    execution_time = 1/(end_time - start_time)
    fps_ls.append(execution_time)


fps_ls = np.convolve(fps_ls, np.ones(10)/10, mode='valid')
# Create the plot
plt.plot(list(range(len(fps_ls[5:]))), fps_ls[5:], label="fps", color="blue", linestyle="-", linewidth=2)

# Add labels and title
plt.xlabel("frame number")
plt.ylabel("fps")

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
