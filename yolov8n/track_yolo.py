from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Start tracking objects in a video
# You can also use live video streams or webcam input
model.track(source=r"D:\Program Files\road_snow\yolov8n\static\uploads\deer_hpy.mp4",save=True)