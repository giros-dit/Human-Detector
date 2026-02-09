import sys
import fcntl
import os

# Create a lock file to prevent multiple instances
# Use current directory instead of /tmp to avoid permission issues
lock_file_path = os.path.join(os.path.dirname(__file__), '.person_webcam.lock')
lock_file = open(lock_file_path, 'w')
try:
    fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except IOError:
    print("Another instance is already running.")
    sys.exit(0)

# Verify X11 Display is available for X Forwarding
if 'DISPLAY' not in os.environ:
    print("[ERROR] DISPLAY environment variable not set.")
    print("Make sure you connected with: ssh -X user@server")
    sys.exit(1)
else:
    print(f"[INFO] Using DISPLAY: {os.environ['DISPLAY']}")

import cv2
from ultralytics import YOLO

# Set OpenCV to use X11 backend explicitly
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'

def detect_person_yolov8():
    # Load pretrained YOLOv8s model
    model = YOLO("yolov8n.pt")  # or 'yolov8s.pt' for more accuracy

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam couldn't be opened.")
        return

    print("[INFO] Running YOLOv8 Person Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference
        results = model(frame, stream=True)

        person_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Only keep class 'person' (COCO class id 0)
                if cls == 0 and conf > 0.5:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {person_count}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f'Total Persons: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection complete.")

if __name__ == "__main__":
    detect_person_yolov8()