from ultralytics import YOLO
import cv2
from text2command import get_gpt_command
from Ask_yolo import asking_yolo
import time
import matplotlib.pyplot as plt
import numpy as np
from vimba import *
from Cropped2center import cropped2center

# Load a model
model = YOLO("yolo11s.pt")
cam_width = 1388
cam_height = 1038
STRIDE = 15
target_found = False
padding = 30
fps_ls = []
x1, y1, x2, y2 = 0, 0, 0, 0
chosen_class = "person"
delta_x1 = 0
delta_y1 = 0

# Open the default camera (0 for the first camera, 1 for the second, etc.)
# cap = cv2.VideoCapture(0)

def frame_handler(camera, frame):
    global target_found, x1, y1, x2, y2, chosen_class, fps_ls, delta_x1, delta_y1
    camera.queue_frame(frame)
    start_time = time.time()
    numpy_array = frame.as_numpy_ndarray()
    frame = cv2.cvtColor(numpy_array, cv2.COLOR_BayerRG2BGR)
    frame2show = cropped2center(frame)
    frame_small = cv2.resize(frame2show, (frame2show.shape[1] // 2, frame2show.shape[0] // 2))
    cv2.imshow("Full Frame", frame_small)

    # ret, frame = cap.read()
    if len(fps_ls) % STRIDE == 0:
        target_found = False

        results, x1, y1, x2, y2, target_found = asking_yolo(model,frame,chosen_class)

        if target_found:
            x1 += delta_x1
            y1 += delta_y1
            x2 += delta_x1
            y2 += delta_y1
            cam.get_feature_by_name("Height").set(min(y2 - y1 + 2*padding,cam_height- max(0, y1 - padding)))
            cam.get_feature_by_name("Width").set(min(x2-x1 + 2*padding,cam_width- max(0, x1 - padding)))
            cam.get_feature_by_name("OffsetY").set(max(0, y1 - padding))
            cam.get_feature_by_name("OffsetX").set(max(0, x1 - padding))
            delta_x1 = max(0, x1 - padding)
            delta_y1 = max(0, y1 - padding)
        else:
            cam.get_feature_by_name("Height").set(cam_height)
            cam.get_feature_by_name("Width").set(cam_width)
            cam.get_feature_by_name("OffsetY").set(0)
            cam.get_feature_by_name("OffsetX").set(0)
            delta_x1 = 0
            delta_y1 = 0

    # Press 'q' to quit or 'c' to change class
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        fps_ls = np.convolve(fps_ls, np.ones(10) / 10, mode='valid')
        # Create the plot
        plt.plot(list(range(len(fps_ls[5:]))), fps_ls[5:], label="fps", color="blue", linestyle="-", linewidth=2)
        plt.xlabel("frame number")
        plt.ylabel("fps")
        plt.legend()
        plt.show()
        cv2.destroyAllWindows()
        exit(1)
    elif key == ord('c'):  # Change the class when 'c' is pressed
        new_class = input("message program: ")
        dic_json_answer = get_gpt_command(new_class)
        chosen_class = dic_json_answer["class"]
        print(f"Chosen class updated to: {chosen_class}")

    end_time = time.time()
    execution_time = 1 / (end_time - start_time)
    fps_ls.append(execution_time)


with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras ()
    with cams [0] as cam:
        #   cam.get_feature_by_name("ExposureTimeAbs").set(2000)
        # cam.get_feature_by_name("GainAuto").set("Continuous")
        cam.get_feature_by_name("Height").set(cam_height)
        cam.get_feature_by_name("Width").set(cam_width)
        cam.get_feature_by_name("OffsetY").set(0)
        cam.get_feature_by_name("OffsetX").set(0)
        cam.start_streaming(frame_handler)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit the program
                break
        print("stop")
        cam.stop_streaming()





