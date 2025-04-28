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

# Configuration parameters
cam_width = 1388
cam_height = 1038
YOLO_STRIDE = 20
CHANGE_ROI_THRESH = 8
DISPLAY_FPS = 30  # Target display refresh rate
TARGET_DETECTION_FPS = 15  # Target detection refresh rate
padding = 30
edge_proximity_yolo_margin = 10

# Create global variables for shared state
model = YOLOWorld("yolov8s-worldv2.pt")
chosen_class = "person"
model.set_classes([chosen_class])

# Frame and ROI tracking
target_found = False
near_edge = True
fps_ls = []
x1, y1, x2, y2 = 0, 0, 0, 0
x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height = 0, 0, cam_width, cam_height
delta_x1, delta_y1 = 0, 0
frame_count = 0
ROI_counter = 0
old_frame_time = 0
full_image = None

# Thread control flags
loop_cond = True
save_frames = False
saving_thread_active = True

# Create queues for thread communication
display_queue = queue.Queue(maxsize=2)  # Small queue for display frames
yolo_queue = queue.Queue(maxsize=1)  # Only process the latest frame
frame_queue = queue.Queue(maxsize=20)  # Queue for saving frames

# Create shared objects for thread communication
detection_results = {
    'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
    'target_found': False,
    'last_update_time': 0
}
detection_lock = threading.Lock()

# Create the save directory if it doesn't exist
save_directory = "saved_frames"
os.makedirs(save_directory, exist_ok=True)
frame_number = 0


# Define the thread function for YOLO detection
def yolo_detection_thread():
    global loop_cond

    print("Starting YOLO detection thread")
    frame_skips = 0

    while loop_cond:
        try:
            # Get latest frame with a timeout
            frame = yolo_queue.get(timeout=0.5)

            # Skip some frames if we're falling behind
            frame_skips += 1
            if frame_skips % 3 != 0:  # Process every 3rd frame when backed up
                yolo_queue.task_done()
                continue

            # Process with YOLO
            start_time = time.time()
            results, x1, y1, x2, y2, target_found = asking_yolo(model, frame, chosen_class)
            process_time = time.time() - start_time

            # Update shared results with a lock
            with detection_lock:
                detection_results['x1'] = x1
                detection_results['y1'] = y1
                detection_results['x2'] = x2
                detection_results['y2'] = y2
                detection_results['target_found'] = target_found
                detection_results['last_update_time'] = time.time()

            print(f"YOLO processing time: {process_time:.4f} seconds")
            yolo_queue.task_done()

            # Adaptive timing - sleep if we're processing too quickly
            if process_time < 1.0 / TARGET_DETECTION_FPS:
                time.sleep(1.0 / TARGET_DETECTION_FPS - process_time)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in YOLO thread: {e}")
            continue


# Define the thread function for display
def display_thread():
    global loop_cond, x1, y1, x2, y2, target_found, full_image

    print("Starting display thread")
    frame_count = 0

    while loop_cond:
        try:
            # Get frame with timeout
            frame_data = display_queue.get(timeout=0.1)
            if frame_data is not None:
                frame, delta_x, delta_y = frame_data
                frame_count += 1

                # Get the latest detection results
                with detection_lock:
                    x1 = detection_results['x1']
                    y1 = detection_results['y1']
                    x2 = detection_results['x2']
                    y2 = detection_results['y2']
                    target_found = detection_results['target_found']

                # Only process for display at target FPS
                try:
                    # Draw bounding box if target found
                    if target_found:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Draw ROI rectangle
                    cv2.rectangle(frame, (int(x1_ROI), int(y1_ROI)),
                                  (int(x1_ROI + x2_ROI_width), int(y1_ROI + y2_ROI_height)),
                                  (0, 0, 255), 1)

                    # Process for display
                    if full_image is None:
                        frame2show = frame.copy()
                    else:
                        try:
                            frame2show = cropped2sameplace(full_image, frame, int(delta_x), int(delta_y))
                        except Exception as e:
                            print(f"Error in cropped2sameplace: {e}")
                            frame2show = frame.copy()

                    # Resize and display
                    frame_small = cv2.resize(frame2show, (frame2show.shape[1] // 2, frame2show.shape[0] // 2))
                    cv2.imshow("Camera Feed", frame_small)

                    # Process key events
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        loop_cond = False

                except Exception as e:
                    print(f"Error displaying frame: {e}")

                # Mark as done
                display_queue.task_done()

            # Sleep to maintain target display FPS
            time.sleep(1.0 / DISPLAY_FPS)

        except queue.Empty:
            # No new frame available, just continue
            continue
        except Exception as e:
            print(f"Error in display thread: {e}")


# Define the thread function for saving frames
def frame_saving_thread():
    global saving_thread_active, frame_number

    print("Starting frame saving thread")

    while saving_thread_active:
        try:
            # Clear backlog if queue gets too large
            if frame_queue.qsize() > 15:
                # Clear old frames to prevent backlog
                for _ in range(5):
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                    except queue.Empty:
                        break

            # Get frame from queue with timeout
            frame_data = frame_queue.get(timeout=0.5)
            if frame_data is not None:
                # Unpack the data
                frame, frame_num, new_frame_time = frame_data
                # Save the frame
                save_path = os.path.join(save_directory, f"frame_{frame_num:06d}_{new_frame_time:.2f}.jpg")
                cv2.imwrite(save_path, frame)
                # Mark task as done
                frame_queue.task_done()
        except queue.Empty:
            # Queue was empty, just continue the loop
            continue
        except Exception as e:
            print(f"Error in saving thread: {e}")

    # Process any remaining items in the queue before exiting
    print("Cleaning up frame saving queue...")
    while not frame_queue.empty():
        try:
            frame_data = frame_queue.get_nowait()
            if frame_data is not None:
                frame, frame_num, timestamp = frame_data
                save_path = os.path.join(save_directory, f"frame_{frame_num:06d}_{timestamp:.2f}.jpg")
                cv2.imwrite(save_path, frame)
                frame_queue.task_done()
        except queue.Empty:
            break


# Camera frame handler - keeps processing minimal
def frame_handler(camera, frame):
    global old_frame_time, fps_ls, frame_count, full_image, x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height, \
        delta_x1, delta_y1, frame_number, save_frames, near_edge, ROI_counter, cam

    # Queue the frame immediately
    camera.queue_frame(frame)

    # Calculate FPS
    new_frame_time = time.time()
    if old_frame_time > 0:  # Skip first frame
        execution_time = 1 / (new_frame_time - old_frame_time)
        fps_ls.append(execution_time)
    old_frame_time = new_frame_time

    try:
        # Convert frame to usable format
        numpy_array = frame.as_numpy_ndarray()
        current_frame = cv2.cvtColor(numpy_array, cv2.COLOR_BayerRG2BGR)
        frame_count += 1

        # Store full image for reference
        if x1_ROI == 0 and y1_ROI == 0 and x2_ROI_width == cam_width and y2_ROI_height == cam_height:
            full_image = current_frame.copy()

        # Add to display queue (non-blocking)
        try:
            if display_queue.qsize() < display_queue.maxsize:
                display_queue.put_nowait((current_frame.copy(), delta_x1, delta_y1))
        except queue.Full:
            pass  # Skip frame if queue is full

        # Add to YOLO queue on interval (replace any existing frame)
        if frame_count % YOLO_STRIDE == 0:
            try:
                # Clear queue first
                while not yolo_queue.empty():
                    yolo_queue.get_nowait()
                    yolo_queue.task_done()
                # Add new frame
                yolo_queue.put_nowait(current_frame.copy())
            except queue.Full:
                pass

        # Save frames if enabled
        if save_frames and frame_count % 5 == 0:  # Save every 5th frame
            try:
                frame_queue.put_nowait((current_frame.copy(), frame_number, new_frame_time))
                frame_number += 1
            except queue.Full:
                pass  # Skip if queue is full

        # Process ROI updates based on detections (only occasionally)
        if frame_count % YOLO_STRIDE == 0:
            ROI_counter += 1

            # Get detection results
            with detection_lock:
                x1 = detection_results['x1']
                y1 = detection_results['y1']
                x2 = detection_results['x2']
                y2 = detection_results['y2']
                target_found = detection_results['target_found']
                last_update_time = detection_results['last_update_time']

            # Only update ROI if detection is recent (within 1 second)
            if target_found and (time.time() - last_update_time) < 1.0:
                # Check if bbox is near ROI edges
                near_left = abs(x1 + delta_x1 - x1_ROI) <= edge_proximity_yolo_margin
                near_right = abs((x2 + delta_x1) - (x1_ROI + x2_ROI_width)) <= edge_proximity_yolo_margin
                near_top = abs(y1 + delta_y1 - y1_ROI) <= edge_proximity_yolo_margin
                near_bottom = abs(y2 + delta_y1 - (y1_ROI + y2_ROI_height)) <= edge_proximity_yolo_margin
                near_edge = (near_left or near_right or near_top or near_bottom)

                if near_edge or (
                        x1_ROI == 0 and y1_ROI == 0 and x2_ROI_width == cam_width and y2_ROI_height == cam_height):
                    x1_adj = x1 + delta_x1
                    y1_adj = y1 + delta_y1
                    x2_adj = x2 + delta_x1
                    y2_adj = y2 + delta_y1

                    # Update camera ROI parameters
                    new_height = min(y2_adj - y1_adj + 2 * padding, cam_height - max(0, y1_adj - padding))
                    new_width = min(x2_adj - x1_adj + 2 * padding, cam_width - max(0, x1_adj - padding))
                    new_offset_y = max(0, y1_adj - padding)
                    new_offset_x = max(0, x1_adj - padding)

                    try:
                        cam.get_feature_by_name("Height").set(new_height)
                        cam.get_feature_by_name("Width").set(new_width)
                        cam.get_feature_by_name("OffsetY").set(new_offset_y)
                        cam.get_feature_by_name("OffsetX").set(new_offset_x)

                        # Update ROI tracking variables
                        x1_ROI = new_offset_x
                        y1_ROI = new_offset_y
                        x2_ROI_width = new_width
                        y2_ROI_height = new_height
                        delta_x1 = new_offset_x
                        delta_y1 = new_offset_y

                        print(f"Updated ROI: {x1_ROI},{y1_ROI},{x2_ROI_width},{y2_ROI_height}")
                    except Exception as e:
                        print(f"Error updating camera ROI: {e}")

                    near_edge = False
            elif not target_found and ROI_counter % CHANGE_ROI_THRESH == 0:
                # Reset to full frame when target not found
                try:
                    cam.get_feature_by_name("Height").set(cam_height)
                    cam.get_feature_by_name("Width").set(cam_width)
                    cam.get_feature_by_name("OffsetY").set(0)
                    cam.get_feature_by_name("OffsetX").set(0)
                    delta_x1 = 0
                    delta_y1 = 0
                    x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height = 0, 0, cam_width, cam_height
                    print("Reset to full frame")
                except Exception as e:
                    print(f"Error resetting to full frame: {e}")

    except Exception as e:
        print(f"Error in frame handler: {e}")


# Main execution
with Vimba.get_instance() as vimba:
    cams = vimba.get_all_cameras()
    with cams[0] as cam:
        # Configure camera
        try:
            cam.get_feature_by_name("ExposureAuto").set("Once")
            cam.get_feature_by_name("Height").set(cam_height)
            cam.get_feature_by_name("Width").set(cam_width)
            cam.get_feature_by_name("OffsetY").set(0)
            cam.get_feature_by_name("OffsetX").set(0)

            # Set fastest possible exposure
            cam.get_feature_by_name("ExposureTimeAbs").set(1000)  # 1ms exposure
            cam.get_feature_by_name("GainRaw").set(30)  # Compensate with gain

            # Try to optimize frame rate if supported
            try:
                cam.get_feature_by_name("AcquisitionFrameRateEnable").set(True)
                cam.get_feature_by_name("AcquisitionFrameRate").set(100)  # Target 100 FPS
            except:
                print("Frame rate control not supported by camera")

            # Try to optimize packet size if camera supports it
            try:
                cam.get_feature_by_name("DeviceLinkThroughputLimit").set(1000000000)  # Max bandwidth
            except:
                pass
        except Exception as e:
            print(f"Error configuring camera: {e}")

        # Start threads
        yolo_thread = threading.Thread(target=yolo_detection_thread, daemon=True)
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        save_thread = threading.Thread(target=frame_saving_thread, daemon=True)

        yolo_thread.start()
        display_thread_obj.start()
        save_thread.start()

        # Start camera streaming
        old_frame_time = time.time()
        cam.start_streaming(frame_handler)

        print("Press 'q' to quit")

        # Simple main loop - just keep program alive and check for quit
        try:
            while loop_cond:
                time.sleep(0.1)  # Sleep to avoid CPU spin
                # Exit handling done in display thread with 'q' key
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup everything
            print("Cleaning up...")
            loop_cond = False
            saving_thread_active = False

            # Stop camera streaming
            try:
                cam.stop_streaming()
            except:
                pass

            # Wait for threads to finish
            try:
                yolo_thread.join(timeout=2.0)
                display_thread_obj.join(timeout=2.0)
                save_thread.join(timeout=2.0)
            except:
                pass

            # Destroy all windows
            cv2.destroyAllWindows()

            # Show FPS graph
            if len(fps_ls) > 5:
                fps_trimmed = fps_ls[5:]
                plt.figure(figsize=(10, 6))
                plt.plot(fps_trimmed, label="FPS", color="blue")
                """plt.axhline(y=np.mean(fps_trimmed), color='g', linestyle='-',
                            label=f"Avg: {np.mean(fps_trimmed):.1f} FPS")"""

                # Calculate theoretical max
                max_theoretical_fps = 1000000 / cam.get_feature_by_name("ExposureTimeAbs").get()
                """plt.axhline(y=max_theoretical_fps, color='r', linestyle='--',
                            label=f"Theoretical max: {max_theoretical_fps:.1f} FPS")
                """
                plt.title("Camera Frame Rate Performance")
                plt.xlabel("Frame Number")
                plt.ylabel("Frames Per Second")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

            print("Exiting program")