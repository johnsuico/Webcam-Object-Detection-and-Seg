from ultralytics import YOLO

# Load desired model
# Using segmentation model for better visualization of detected objects
model = YOLO("yolo26n-seg.pt")

# Use model to predict and segment objects on an image
results = model.predict(source="sample_pictures\street.jpg", show=True, save=True)