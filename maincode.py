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
import os
import threading
import queue
from ultralytics.utils import LOGGER

# Turn off Ultralytics logging
LOGGER.setLevel(40)  # 40 = ERROR, suppresses INFO and WARNING

model = YOLOWorld("yolov8s-worldv2.pt")
cam_width = 1388
cam_height = 1038
YOLO_STRIDE = 30
CHANGE_ROI_THRESH = 8
LIVE_STREAM_FPS = 25
target_found = False
edge_proximity_yolo_margin = 10
near_edge = True
padding = 25
fps_ls = []
x1, y1, x2, y2 = 0, 0, 0, 0
x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height = 0, 0, cam_width, cam_height
chosen_class = "person"
model.set_classes(["person"])
delta_x1 = 0
delta_y1 = 0
frame_count = 0
ROI_counter = 0
old_frame_time = 0
old_frame_time_plot = 0
past_frame = None
full_image = None
loop_cond = True

# Add these at the beginning of your script
save_frames = False  # Flag to enable/disable frame saving
save_directory = "saved_frames"  # Directory to save frames
frame_save_interval = 5  # Save every Nth frame
frame_number = 0  # Counter for naming frames
frame_queue = queue.Queue(maxsize=100)  # Queue to hold frames to be saved
saving_thread_active = True  # Control flag for the saving thread

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)


# Define the thread function that will save frames
def frame_saving_thread():
    global saving_thread_active, frame_number

    while saving_thread_active:
        try:
            # Get frame from queue with timeout to allow checking the active flag
            frame_data = frame_queue.get(timeout=0.5)
            if frame_data is not None:
                # Unpack the data
                frame, frame_num, new_frame_time = frame_data
                # Save the frame
                save_path = os.path.join(save_directory, f"frame_{frame_num:06d}_{new_frame_time}.jpg")
                cv2.imwrite(save_path, frame)
                # Mark task as done
                frame_queue.task_done()
        except queue.Empty:
            # Queue was empty, just continue the loop
            continue

    # Process any remaining items in the queue before exiting
    while not frame_queue.empty():
        try:
            frame_data = frame_queue.get_nowait()
            if frame_data is not None:
                frame, frame_num = frame_data
                save_path = os.path.join(save_directory, f"frame_{frame_num:06d}.jpg")
                cv2.imwrite(save_path, frame)
                frame_queue.task_done()
        except queue.Empty:
            break


# Start the frame saving thread
save_thread = threading.Thread(target=frame_saving_thread, daemon=True)
save_thread.start()


def frame_handler(camera, frame):
    global old_frame_time_plot, \
        target_found, x1, y1, x2, y2, chosen_class, fps_ls, \
        delta_x1, delta_y1, frame_count, old_frame_time, near_edge, \
        x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height, ROI_counter, full_image, \
        save_frames, frame_number, saving_thread_active, loop_cond, key

    camera.queue_frame(frame)

    new_frame_time = time.time()

    if new_frame_time - old_frame_time > 0.0125:

        execution_time = 1 / (new_frame_time - old_frame_time)
        if frame_count % YOLO_STRIDE != 1:
            fps_ls.append(execution_time)
        old_frame_time = new_frame_time

        numpy_array = frame.as_numpy_ndarray()
        frame = cv2.cvtColor(numpy_array, cv2.COLOR_BayerRG2BGR)

        if x1_ROI == 0 and y1_ROI == 0 and x2_ROI_width == cam_width and y2_ROI_height == cam_height:
            full_image = frame

        if new_frame_time - old_frame_time_plot > 0.0333:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if full_image is None:
                frame2show = cropped2center(frame)
            else:
                frame2show = cropped2sameplace(full_image, frame, int(delta_x1), int(delta_y1))
            #cv2.rectangle(frame2show, (int(x1_ROI+5), int(y1_ROI+5)), (int(x1_ROI + x2_ROI_width-5), int(y1_ROI + y2_ROI_height-5)), (0, 0, 255), 1)
            frame_small = cv2.resize(frame2show, (frame2show.shape[1] // 2, frame2show.shape[0] // 2))
            cv2.imshow("Full Frame", frame_small)
            key = cv2.waitKey(1)
            if key == ord('q'):
                loop_cond = False
            old_frame_time_plot = new_frame_time

        if save_frames and frame_count % frame_save_interval == 0:
            # Make a copy of the frame to avoid modifying it while it's in the queue
            frame_copy = frame.copy()
            try:
                # Add to queue without blocking if queue is full
                frame_queue.put_nowait((frame_copy, frame_number, new_frame_time))
                frame_number += 1
            except queue.Full:
                # Queue is full, skip this frame
                print("Warning: Frame saving queue is full, skipping frame")

        frame_count+=1
        if frame_count > 20000:
            frame_count = 0

        if frame_count % YOLO_STRIDE == 0:
            ROI_counter += 1
            target_found = False
            yol_t = time.time()
            results, x1, y1, x2, y2, target_found = asking_yolo(model,frame,chosen_class)
            yol_t -= time.time()
            print(f"Yolo frame time: {-yol_t:.4f} sec")

            #print(x1+ delta_x1, y1+ delta_y1, x2+ delta_x1, y2+ delta_y1)
            #print(x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_width)

            # Check if the new BBox is near at least one of the edges
            near_left = abs(x1 + delta_x1 - x1_ROI) <= edge_proximity_yolo_margin
            near_right = abs((x2 + delta_x1) - (x1_ROI + x2_ROI_width)) <= edge_proximity_yolo_margin
            near_top = abs(y1 + delta_y1 - y1_ROI) <= edge_proximity_yolo_margin
            near_bottom = abs(y2 + delta_y1 - (y1_ROI + y2_ROI_height)) <= edge_proximity_yolo_margin
            near_edge = (near_left or near_right or near_top or near_bottom)

            if target_found:
                if near_edge or (x1_ROI == 0 and y1_ROI == 0 and x2_ROI_width == cam_width and y2_ROI_height == cam_height):
                    x1 += delta_x1
                    y1 += delta_y1
                    x2 += delta_x1
                    y2 += delta_y1
                    cam.get_feature_by_name("Height").set(min(y2 - y1 + 2*padding,cam_height- max(0, y1 - padding)))
                    cam.get_feature_by_name("Width").set(min(x2-x1 + 2*padding,cam_width- max(0, x1 - padding)))
                    cam.get_feature_by_name("OffsetY").set(max(0, y1 - padding))
                    cam.get_feature_by_name("OffsetX").set(max(0, x1 - padding))
                    x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height = max(0, x1 - padding), max(0, y1 - padding), min(x2 - x1 + 2 * padding, cam_width - max(0, x1 - padding)), min(y2 - y1 + 2 * padding, cam_height - max(0, y1 - padding))
                    delta_x1 = max(0, x1 - padding)
                    delta_y1 = max(0, y1 - padding)
                    print(
                        f"_______________ CHANGED ROI _________ROI_counter: {ROI_counter % CHANGE_ROI_THRESH == 0} ___ near_edge: {near_edge} ___ fullframe: {(not near_edge and not ROI_counter % CHANGE_ROI_THRESH == 0)}")
                    print(f"height is: {y2_ROI_height}")
                    near_edge = False

            else:
                cam.get_feature_by_name("Height").set(cam_height)
                cam.get_feature_by_name("Width").set(cam_width)
                cam.get_feature_by_name("OffsetY").set(0)
                cam.get_feature_by_name("OffsetX").set(0)
                delta_x1 = 0
                delta_y1 = 0
                x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height = 0 , 0 , cam_width, cam_height


    old_frame_time = new_frame_time

    # Press 'q' to quit or 'c' to change class
    """key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program

        saving_thread_active = False
        save_thread.join(timeout=5.0)
        #fps_ls = np.convolve(fps_ls, np.ones(20) / 20, mode='valid')
        # Create the plot
        plt.plot(list(range(len(fps_ls[5:]))), fps_ls[5:], label="fps", color="blue", linestyle="-", linewidth=2)

        maxfps = 1000000/cam.get_feature_by_name("ExposureTimeAbs").get()
        if maxfps < 80:
            plt.axhline(y=maxfps, color='r', linestyle='--', linewidth=2)

        plt.xlabel("frame number")
        plt.ylabel("fps")
        plt.legend()
        plt.show()
        cv2.destroyAllWindows()
        exit(1)
        """
    if key == ord('r'):
        # cam.get_feature_by_name("ExposureAuto").set("Once")
        cam.get_feature_by_name("ExposureTimeAbs").set(3000)
        cam.get_feature_by_name("rqGainRaw").set(25)
        # cam.get_feature_by_name("GainAuto").set("Once")
        cam.get_feature_by_name("BalanceWhiteAuto").set("Once")
    elif key == ord('s'):  # Toggle frame saving when 's' is pressed
        save_frames = not save_frames
        print(f"Frame saving {'enabled' if save_frames else 'disabled'}")
    elif key == ord('c'):  # Change the class when 'c' is pressed
        new_class = input("message program: ")
        dic_json_answer = get_gpt_command(new_class)
        chosen_class = dic_json_answer["class"]
        model.set_classes([chosen_class])
        print(f"Chosen class updated to: {chosen_class}")



with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras()
    with cams [0] as cam:
        cam.get_feature_by_name("ExposureAuto").set("Once")
        cam.get_feature_by_name("Height").set(cam_height)
        cam.get_feature_by_name("Width").set(cam_width)
        cam.get_feature_by_name("OffsetY").set(0)
        cam.get_feature_by_name("OffsetX").set(0)

        # Set fastest possible exposure
        cam.get_feature_by_name("ExposureTimeAbs").set(1000)  # 1ms exposure
        cam.get_feature_by_name("GainRaw").set(30)  # Compensate with gain

        # Check if your camera supports these features
        try:
            cam.get_feature_by_name("AcquisitionFrameRateEnable").set(True)
            cam.get_feature_by_name("AcquisitionFrameRate").set(100)  # Target 100 FPS
        except:
            pass

        # Try to optimize packet size if camera supports it
        try:
            # For USB3 cameras
            cam.get_feature_by_name("DeviceLinkThroughputLimit").set(1000000000)  # Max bandwidth
        except:
            pass

        old_frame_time = time.time()
        cam.start_streaming(frame_handler)
        try:
            while True:
                # Process UI events but don't wait
                key = cv2.waitKey(1) & 0xFF
                if not loop_cond:
                    break
        finally:
            # Show FPS graph on exit
            cam.stop_streaming()
            if len(fps_ls) > 5:
                fps_trimmed = fps_ls[5:]
                #fps_trimmed = np.convolve(fps_trimmed, np.ones(20) / 20, mode='valid')
                plt.figure(figsize=(10, 6))
                plt.plot(fps_trimmed, label="FPS", color="blue")
                plt.axhline(y=np.mean(fps_trimmed), color='g', linestyle='-', label=f"Avg: {np.mean(fps_trimmed):.1f}")

                max_theoretical_fps = 1000000 / cam.get_feature_by_name("ExposureTimeAbs").get()
                """plt.axhline(y=max_theoretical_fps, color='r', linestyle='--',
                            label=f"Theoretical max: {max_theoretical_fps:.1f}")
                """
                plt.title("Camera Frame Rate Performance")
                plt.xlabel("Frame Number")
                plt.ylabel("Frames Per Second")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()






