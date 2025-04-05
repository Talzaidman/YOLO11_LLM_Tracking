from ultralytics import YOLOWorld
import cv2
from text2command import get_gpt_command
from Ask_yolo import asking_yolo
import time
import matplotlib.pyplot as plt
import numpy as np
from vimba import *
from Cropped2center import cropped2center
from Cropped2sameplace import cropped2sameplace

# Load a model
model = YOLOWorld("yolov8s-worldv2.pt")
cam_width = 1388
cam_height = 1038
YOLO_STRIDE = 30
CHANGE_ROI_THRESH = 8
LIVE_STREAM_STRIDE = 3
target_found = False
edge_proximity_yolo_margin = 30
near_edge = True
padding = 50
fps_ls = []
x1, y1, x2, y2 = 0, 0, 0, 0
x1_ROI, y1_ROI, x2_ROI, y2_ROI = 0, 0, cam_width, cam_height
chosen_class = "man"
model.set_classes(["man"])
delta_x1 = 0
delta_y1 = 0
frame_count = 0
ROI_counter = 0
old_time = 0
pastframe = None

def frame_handler(camera, frame):
    global target_found, x1, y1, x2, y2, chosen_class, fps_ls, delta_x1, delta_y1, frame_count, old_time, near_edge, x1_ROI, y1_ROI, x2_ROI, y2_ROI, ROI_counter, fullimage

    new_time = time.time()
    execution_time = 1 / (new_time - old_time)
    if frame_count % YOLO_STRIDE != 1:
        fps_ls.append(execution_time)
    old_time = new_time

    camera.queue_frame(frame)

    numpy_array = frame.as_numpy_ndarray()
    frame = cv2.cvtColor(numpy_array, cv2.COLOR_BayerRG2BGR)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if x1_ROI == 0 and y1_ROI == 0 and x2_ROI == cam_width and y2_ROI == cam_height:
        fullimage = frame
    if fullimage is None:
        frame2show = cropped2center(frame)
    else:
        frame2show = cropped2sameplace(fullimage.copy(), frame, int(delta_x1),int(delta_y1))
    frame_small = cv2.resize(frame2show, (frame2show.shape[1] // 2, frame2show.shape[0] // 2))

    if execution_time < 25:
        cv2.imshow("Full Frame", frame_small)
    else:
        if frame_count %2 == 0:
            cv2.imshow("Full Frame", frame_small)
    frame_count+=1
    if frame_count > 20000:
        frame_count = 0

    # ret, frame = cap.read()
    if frame_count % YOLO_STRIDE == 0:
        ROI_counter += 1
        target_found = False
        results, x1, y1, x2, y2, target_found = asking_yolo(model,frame,chosen_class)
        print(x1_ROI, y1_ROI, x2_ROI, y2_ROI)
        print(x1+ delta_x1, y1+ delta_y1, x2+ delta_x1, y2+ delta_y1)

        # Check if the new BBox is near at least one of the edges
        near_left = abs(x1 + delta_x1 - x1_ROI) <= edge_proximity_yolo_margin
        near_right = abs((x2 + delta_x1) - (x1_ROI + x2_ROI)) <= edge_proximity_yolo_margin
        near_top = abs(y1 + delta_y1 - y1_ROI) <= edge_proximity_yolo_margin
        near_bottom = abs(y2 + delta_y1 - (y1_ROI + y2_ROI)) <= edge_proximity_yolo_margin
        near_edge = (near_left or near_right or near_top or near_bottom)

        if target_found:
            if near_edge or (x1_ROI == 0 and y1_ROI == 0 and x2_ROI == cam_width and y2_ROI == cam_height):
                x1 += delta_x1
                y1 += delta_y1
                x2 += delta_x1
                y2 += delta_y1
                cam.get_feature_by_name("Height").set(min(y2 - y1 + 2*padding,cam_height- max(0, y1 - padding)))
                cam.get_feature_by_name("Width").set(min(x2-x1 + 2*padding,cam_width- max(0, x1 - padding)))
                cam.get_feature_by_name("OffsetY").set(max(0, y1 - padding))
                cam.get_feature_by_name("OffsetX").set(max(0, x1 - padding))
                x1_ROI, y1_ROI, x2_ROI, y2_ROI = max(0, x1 - padding), max(0, y1 - padding), min(x2 - x1 + 2 * padding, cam_width - max(0, x1 - padding)), min(y2 - y1 + 2 * padding, cam_height - max(0, y1 - padding))
                delta_x1 = max(0, x1 - padding)
                delta_y1 = max(0, y1 - padding)
                print(
                    f"_______________ CHANGED ROI _________ROI_counter: {ROI_counter % CHANGE_ROI_THRESH == 0} ___ near_edge: {near_edge}")
                near_edge = False

        else:
            cam.get_feature_by_name("Height").set(cam_height)
            cam.get_feature_by_name("Width").set(cam_width)
            cam.get_feature_by_name("OffsetY").set(0)
            cam.get_feature_by_name("OffsetX").set(0)
            delta_x1 = 0
            delta_y1 = 0
            x1_ROI, y1_ROI, x2_ROI, y2_ROI = 0 , 0 , cam_width, cam_height

    # Press 'q' to quit or 'c' to change class
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        #fps_ls = np.convolve(fps_ls, np.ones(50) / 50, mode='valid')
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
        model.set_classes([chosen_class])
        print(f"Chosen class updated to: {chosen_class}")


with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras ()
    with cams [0] as cam:
        cam.get_feature_by_name("ExposureTimeAbs").set(2000)
        # cam.get_feature_by_name("GainAuto").set("Continuous")
        cam.get_feature_by_name("Height").set(cam_height)
        cam.get_feature_by_name("Width").set(cam_width)
        cam.get_feature_by_name("OffsetY").set(0)
        cam.get_feature_by_name("OffsetX").set(0)
        old_time = time.time()
        cam.start_streaming(frame_handler)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit the program
                break
        print("stop")
        cam.stop_streaming()





