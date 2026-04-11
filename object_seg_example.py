from ultralytics import YOLO

# Load desired model
# Using segmentation model for better visualization of detected objects
# Uncomment line below to use the smaller model for faster inference, but it may be less accurate
# Model is also not prompt-based, so it will detect all classes instead of just the person class
# model = YOLO("yolo26n-seg.pt")

# Using the larger model for better accuracy, but it may be slower
# Using yoloe to use prompt detection
model = YOLO("yoloe-26l-seg.pt")
model.set_classes(["person"])

# Use model to predict and segment objects on an image
results = model.predict(source="sample_pictures\street.jpg", show=True, save=True)