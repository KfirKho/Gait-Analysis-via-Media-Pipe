import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawings = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculate the angle between three points a, b, and c."""
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(r"walking_skeleton_slowed.mp4")

# Initialize Mediapipe Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error.")
            break

        # Convert frame to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make pose detection
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            height, width, _ = frame.shape

            # Convert normalized coordinates (0-1) to pixel values
            def get_pixel_coords(landmark):
                return int(landmark.x * width), int(landmark.y * height)

            shoulder = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            elbow = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            wrist = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
            left_ankle = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
            right_ankle = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

            # Calculate Step Width
            step_width = abs(left_ankle[0] - right_ankle[0])  # Difference in X-coordinates

            # Display Step Width
            cv2.putText(image, f"Step Width: {step_width}px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)

            # Calculate Angle (Elbow Flexion)
            angle = calculate_angle(shoulder, elbow, wrist)

            # Display Angle
            cv2.putText(image, f"Elbow Angle: {int(angle)} degrees", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)

        except Exception as e:
            print("Error processing landmarks:", e)

        # Draw Pose Landmarks
        mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawings.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                   mp_drawings.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Show Video
        cv2.imshow('Gait Analysis', image)

        # Press 'q' to exit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
