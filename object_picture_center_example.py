import cv2
from ultralytics import YOLO

# Load desired model
# Using segmentation model for better visualization of detected objects
# Uncomment line below to use the smaller model for faster inference, but it may be less accurate
# Model is also not prompt-based, so it will detect all classes instead of just the person class
# model = YOLO("yolo26n-seg.pt")

# Using the larger model for better accuracy, but it may be slower
# Using yoloe to use prompt detection
model = YOLO("yoloe-26l-seg.pt")
model.set_classes(["cup"])

# Use model to predict and segment objects on an image
# Set save to false so we can control where we save the resulting picture and what we name it
results = model.predict(source="sample_pictures\cup.jpg", save=False)

# results[0].plot() draws the bounding boxes, masks, and labels on the image
# It returns a NumPy array (BGR format, standard for OpenCV)
annotated_image = results[0].plot()

# Check if there are any bounding boxes detected
if results[0].boxes is not None:

    # results[0].boxes.xywh gives us: [center_x, center_y, width, height]
    # .cpu() moves it from GPU to CPU (if you are using a GPU)
    # .numpy() converts it to a NumPy array
    # .astype(int) converts the decimal coordinates to whole pixels
    boxes_xywh = results[0].boxes.xywh.cpu().numpy().astype(int)  # Get bounding boxes in xywh format

    # Loop through every detected box
    for box in boxes_xywh:
        center_x = box[0]
        center_y = box[1]

        # Draw a red circle at the center of the bounding box
        cv2.circle(annotated_image, (center_x, center_y), radius=15, color=(0, 0, 255), thickness=-1)


### Custom saving route
# ================================
# Save it exactly where you want with exactly what name you want
custom_save_path = "bound_box_center.jpg"
cv2.imwrite(custom_save_path, annotated_image)
# ================================