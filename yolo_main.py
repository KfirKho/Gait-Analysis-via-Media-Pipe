import cv2
import torch
from ultralytics import YOLO

# Load YOLO-Pose model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n-pose.pt").to(device)

# Open video
cap = cv2.VideoCapture("walking_human.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO-Pose inference
    results = model([frame])

    for result in results:
        if hasattr(result, "keypoints") and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Extract keypoints
            
            # Draw keypoints and print coordinates on the screen
            for i, keypoint in enumerate(keypoints[0]):  # Access first person in the keypoints array
                x, y = keypoint  # Each row in keypoints is an [x, y] pair
                
                if x == 0 and y == 0:  # Skip invalid keypoints (0,0)
                    continue
                
                x, y = int(x), int(y)  # Convert to integer coordinates
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw keypoint
                cv2.putText(frame, f"{i}: ({x},{y})", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "No keypoints detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frame with keypoints
    cv2.imshow("YOLO Pose Keypoints", frame)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
