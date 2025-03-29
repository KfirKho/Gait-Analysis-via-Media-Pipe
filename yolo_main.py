from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
results = model.track(r"walking_human.mp4",show=True)
