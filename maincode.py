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
from filterpy.kalman import KalmanFilter  # New import for Kalman filtering

# Turn off Ultralytics logging
LOGGER.setLevel(40)  # 40 = ERROR, suppresses INFO and WARNING

# Kalman Filter implementation for object tracking
class ObjectTracker:
    def __init__(self):
        # State: [x, y, width, height, vx, vy, vw, vh]
        # We track position, size and their velocities
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (motion model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ])
        
        # Measurement matrix (maps state to measurement)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # width
            [0, 0, 0, 1, 0, 0, 0, 0],  # height
        ])
        
        # Initialize state covariance matrix with high uncertainty
        self.kf.P = np.eye(8) * 1000
        
        # Process noise - how much randomness in motion model
        # Increase uncertainty for velocity components
        self.kf.Q = np.eye(8) * 0.1
        self.kf.Q[4:, 4:] *= 10  # More uncertainty in velocity
        
        # Measurement noise - how noisy are our detections
        # Increase measurement noise to trust predictions more when detections are uncertain
        self.kf.R = np.eye(4) * 10
        
        # Initial state
        self.kf.x = np.zeros((8, 1))
        
        # Track initialized state
        self.initialized = False
        self.frames_since_detection = 0
        self.max_prediction_frames = 30  # Maximum frames to predict without detection
        
        # Smoothing parameters
        self.alpha = 0.7  # Smoothing factor for position updates
        self.last_bbox = None  # Store last valid bbox for smoothing
        
    def init(self, bbox):
        """Initialize tracker with first detection [x1, y1, x2, y2]"""
        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Initialize state [x, y, w, h, vx, vy, vw, vh]
        self.kf.x = np.array([[x], [y], [w], [h], [0], [0], [0], [0]])
        self.initialized = True
        self.frames_since_detection = 0
        self.last_bbox = bbox
        
    def predict(self):
        """Predict next state"""
        if not self.initialized:
            return None
            
        self.kf.predict()
        self.frames_since_detection += 1
        
        # Extract predicted bbox
        x, y = self.kf.x[0, 0], self.kf.x[1, 0]
        w, h = self.kf.x[2, 0], self.kf.x[3, 0]
        vx, vy = self.kf.x[4, 0], self.kf.x[5, 0]
        
        # Convert to bbox format [x1, y1, x2, y2]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        return [x1, y1, x2, y2, vx, vy]
        
    def update(self, bbox):
        """Update with new measurement [x1, y1, x2, y2]"""
        if bbox is None:
            # No detection, just maintain prediction
            return
            
        # Apply smoothing if we have a previous bbox
        if self.last_bbox is not None:
            # Smooth position and size
            bbox = [
                self.alpha * bbox[0] + (1 - self.alpha) * self.last_bbox[0],
                self.alpha * bbox[1] + (1 - self.alpha) * self.last_bbox[1],
                self.alpha * bbox[2] + (1 - self.alpha) * self.last_bbox[2],
                self.alpha * bbox[3] + (1 - self.alpha) * self.last_bbox[3]
            ]
        
        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Initialize if not already done
        if not self.initialized:
            self.init(bbox)
            return
            
        # Create measurement
        z = np.array([[x], [y], [w], [h]])
        
        # Update Kalman filter
        self.kf.update(z)
        self.frames_since_detection = 0
        self.last_bbox = bbox
        
    def is_valid(self):
        """Check if tracker is still valid based on time since last detection"""
        return self.initialized and self.frames_since_detection < self.max_prediction_frames
        
    def copy(self):
        """Create a deep copy of this tracker"""
        new_tracker = ObjectTracker()
        new_tracker.kf.x = self.kf.x.copy()
        new_tracker.kf.P = self.kf.P.copy()
        new_tracker.initialized = self.initialized
        new_tracker.frames_since_detection = self.frames_since_detection
        return new_tracker

# Configuration parameters
cam_width = 1388
cam_height = 1038
YOLO_STRIDE = 20
CHANGE_ROI_THRESH = 8
DISPLAY_FPS = 30  # Target display refresh rate
TARGET_DETECTION_FPS = 15  # Target detection refresh rate
padding = 50
edge_proximity_yolo_margin = 10

# Kalman filter parameters
prediction_horizon = 5  # Look ahead this many frames for ROI adjustment
enable_predictive_roi = True  # Flag to enable/disable predictive ROI
velocity_threshold = 2.0  # Minimum velocity to consider for prediction
direction_weight = 0.3  # How much to weight future direction vs current position

# Fixed ROI parameters for 'j' key
FIXED_ROI_X = 840
FIXED_ROI_Y = 712
FIXED_ROI_WIDTH = 140
FIXED_ROI_HEIGHT = 140

# Create global variables for shared state
model = YOLOWorld("yolov8s-worldv2.pt")
chosen_class = "person"
model.set_classes([chosen_class])
tracker = ObjectTracker()  # Initialize Kalman tracker

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
save_frames = True
saving_thread_active = True
roi_fixed = False  # New flag to control ROI behavior

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

# Maximum number of saved frames to keep
MAX_SAVED_FRAMES = 1000

def cleanup_old_frames():
    """Clean up old frames if we exceed the maximum limit."""
    try:
        frames = sorted(os.listdir(save_directory))
        if len(frames) > MAX_SAVED_FRAMES:
            for frame in frames[:-MAX_SAVED_FRAMES]:
                os.remove(os.path.join(save_directory, frame))
    except Exception as e:
        print(f"Error cleaning up old frames: {e}")

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
    global x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height, delta_x1, delta_y1, cam
    global roi_fixed  # Add this to global declarations

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

                    # Draw Kalman prediction if available
                    if target_found and tracker.is_valid():
                        prediction = tracker.predict()
                        if prediction:
                            pred_x1, pred_y1, pred_x2, pred_y2, vx, vy = prediction
                            # Draw predicted bounding box with a different color
                            cv2.rectangle(frame, (int(pred_x1), int(pred_y1)), (int(pred_x2), int(pred_y2)), 
                                         (0, 255, 255), 1)  # Yellow for prediction
                            
                            # Draw velocity vector
                            center_x = int((pred_x1 + pred_x2) / 2)
                            center_y = int((pred_y1 + pred_y2) / 2)
                            end_x = int(center_x + vx * 10)  # Scale for visibility
                            end_y = int(center_y + vy * 10)
                            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 165, 255), 2)

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
                    elif key == ord('j'):
                        # Set ROI to fixed place and prevent auto-reset
                        try:
                            cam.get_feature_by_name("Height").set(FIXED_ROI_HEIGHT)
                            cam.get_feature_by_name("Width").set(FIXED_ROI_WIDTH)
                            cam.get_feature_by_name("OffsetY").set(FIXED_ROI_Y)
                            cam.get_feature_by_name("OffsetX").set(FIXED_ROI_X)
                            x1_ROI = FIXED_ROI_X
                            y1_ROI = FIXED_ROI_Y
                            x2_ROI_width = FIXED_ROI_WIDTH
                            y2_ROI_height = FIXED_ROI_HEIGHT
                            delta_x1 = FIXED_ROI_X
                            delta_y1 = FIXED_ROI_Y
                            roi_fixed = True  # Set the flag to prevent auto-reset
                            print(f"Set ROI to fixed place: {FIXED_ROI_X},{FIXED_ROI_Y},{FIXED_ROI_WIDTH},{FIXED_ROI_HEIGHT}")
                            print("ROI auto-reset disabled. Press 'p' to re-enable.")
                        except Exception as e:
                            print(f"Error setting fixed ROI: {e}")
                    elif key == ord('p'):
                        # Re-enable ROI auto-reset
                        roi_fixed = False
                        print("ROI auto-reset re-enabled")

                except Exception as e:
                    print(f"Error displaying frame: {e}")

                # Mark as done
                display_queue.task_done()

            # Sleep to maintain target display FPS
            time.sleep(1.0 / DISPLAY_FPS)

        except queue.Empty:
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
                frame, frame_num, new_frame_time, fps_str = frame_data
                # Save the frame with FPS in filename
                save_path = os.path.join(save_directory, f"frame_{frame_num:06d}_{fps_str}fps_{new_frame_time:.2f}.jpg")
                cv2.imwrite(save_path, frame)
                # Clean up old frames periodically
                if frame_num % 100 == 0:
                    cleanup_old_frames()
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
                frame, frame_num, timestamp, fps_str = frame_data
                save_path = os.path.join(save_directory, f"frame_{frame_num:06d}_{fps_str}fps_{timestamp:.2f}.jpg")
                cv2.imwrite(save_path, frame)
                frame_queue.task_done()
        except queue.Empty:
            break


# Camera frame handler - keeps processing minimal
def frame_handler(camera, frame):
    global old_frame_time, fps_ls, frame_count, full_image, x1_ROI, y1_ROI, x2_ROI_width, y2_ROI_height, \
        delta_x1, delta_y1, frame_number, save_frames, near_edge, ROI_counter, cam, tracker, roi_fixed

    # Queue the frame immediately
    camera.queue_frame(frame)

    # Calculate FPS
    new_frame_time = time.time()
    current_fps = 0
    if old_frame_time > 0:  # Skip first frame
        current_fps = 1 / (new_frame_time - old_frame_time)
        fps_ls.append(current_fps)
        # Keep only last 1000 FPS measurements
        if len(fps_ls) > 1000:
            fps_ls = fps_ls[-1000:]
    old_frame_time = new_frame_time

    try:
        # Convert frame to usable format with validation
        numpy_array = frame.as_numpy_ndarray()
        if numpy_array is None or numpy_array.size == 0:
            print("Warning: Invalid frame received")
            return
            
        try:
            current_frame = cv2.cvtColor(numpy_array, cv2.COLOR_BayerRG2BGR)
        except cv2.error as e:
            print(f"Error converting frame: {e}")
            return
            
        if current_frame is None or current_frame.size == 0:
            print("Warning: Frame conversion failed")
            return
            
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
        if save_frames:  # Save every frame
            try:
                # Format FPS to 2 decimal places
                fps_str = f"{current_fps:.2f}"
                frame_queue.put_nowait((current_frame.copy(), frame_number, new_frame_time, fps_str))
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
                # Update the Kalman filter with the current detection
                tracker.update([x1, y1, x2, y2])
                
                # Get prediction for future position
                if enable_predictive_roi and tracker.is_valid():
                    # Make a deep copy of tracker for prediction
                    future_tracker = tracker.copy()
                    
                    # Predict future position (multiple steps ahead)
                    for _ in range(prediction_horizon):
                        future_tracker.predict()
                    
                    future_pred = future_tracker.predict()
                    if future_pred:
                        future_x1, future_y1, future_x2, future_y2, vx, vy = future_pred
                        
                        # Only consider prediction if velocity is significant
                        if abs(vx) > velocity_threshold or abs(vy) > velocity_threshold:
                            # Calculate if the object will leave the ROI based on prediction
                            will_leave_left = (future_x1 + delta_x1) < (x1_ROI + edge_proximity_yolo_margin)
                            will_leave_right = (future_x2 + delta_x1) > (x1_ROI + x2_ROI_width - edge_proximity_yolo_margin)
                            will_leave_top = (future_y1 + delta_y1) < (y1_ROI + edge_proximity_yolo_margin)
                            will_leave_bottom = (future_y2 + delta_y1) > (y1_ROI + y2_ROI_height - edge_proximity_yolo_margin)
                            
                            # Object predicted to leave ROI soon
                            predicted_to_leave = (will_leave_left or will_leave_right or 
                                                will_leave_top or will_leave_bottom)
                            
                            # Force ROI update even if not near edge now, but predicted to leave soon
                            if predicted_to_leave:
                                near_edge = True
                                print(f"Kalman predicted object leaving ROI. Updating ROI proactively.")
                                
                                # Adjust bounds to look ahead in the direction of motion
                                if abs(vx) > velocity_threshold:  # If there's significant horizontal motion
                                    # Add extra padding in the direction of motion
                                    x_adjustment = vx * direction_weight * prediction_horizon
                                    if vx > 0:  # Moving right
                                        x2 = max(x2, x2 + x_adjustment)
                                    else:  # Moving left
                                        x1 = min(x1, x1 + x_adjustment)
                                
                                if abs(vy) > velocity_threshold:  # If there's significant vertical motion
                                    # Add extra padding in the direction of motion
                                    y_adjustment = vy * direction_weight * prediction_horizon
                                    if vy > 0:  # Moving down
                                        y2 = max(y2, y2 + y_adjustment)
                                    else:  # Moving up
                                        y1 = min(y1, y1 + y_adjustment)
                
                # Check if bbox is near ROI edges
                near_left = abs(x1 + delta_x1 - x1_ROI) <= edge_proximity_yolo_margin
                near_right = abs((x2 + delta_x1) - (x1_ROI + x2_ROI_width)) <= edge_proximity_yolo_margin
                near_top = abs(y1 + delta_y1 - y1_ROI) <= edge_proximity_yolo_margin
                near_bottom = abs(y2 + delta_y1 - (y1_ROI + y2_ROI_height)) <= edge_proximity_yolo_margin
                
                # Update near_edge if it's not already set by the prediction
                if not near_edge:
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
            elif not target_found and ROI_counter % CHANGE_ROI_THRESH == 0 and not roi_fixed:
                # Let Kalman tracker predict even without a detection
                tracker.predict()
                
                # Reset to full frame when target not found (only if roi_fixed is False)
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
            cam.get_feature_by_name("BalanceWhiteAuto").set("Once")
            cam.get_feature_by_name("Height").set(cam_height)
            cam.get_feature_by_name("Width").set(cam_width)
            cam.get_feature_by_name("OffsetY").set(0)
            cam.get_feature_by_name("OffsetX").set(0)

            # Set fastest possible exposure
            #cam.get_feature_by_name("ExposureTimeAbs").set(1000)  # 1ms exposure
            #cam.get_feature_by_name("GainRaw").set(30)  # Compensate with gain

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