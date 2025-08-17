import cv2
import csv

video_path = r"C:\Users\zaita\PycharmProjects\yolosss\results\Untitled video - Made with Clipchamp (9).mp4"
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Cannot read video")
    cap.release()
    exit()

# Select ROI for the stain
bbox = cv2.selectROI("Select Stain on Blade", frame, False)
x, y, w, h = map(int, bbox)
template = frame[y:y+h, x:x+w].copy()
cv2.destroyAllWindows()

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (frame.shape[1], frame.shape[0])
out = cv2.VideoWriter(r'C:\Users\zaita\PycharmProjects\yolosss\results\tracked_output.mp4', fourcc, fps, frame_size)

positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Template matching
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2
    positions.append((center_x, center_y))

    # Draw tracking visualization
    cv2.rectangle(frame, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    # Show and save frame
    cv2.imshow("Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Save positions to CSV
with open('stain_positions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    writer.writerows(positions)

print("Tracking complete. Output saved to 'tracked_output.mp4' and 'stain_positions.csv'")
