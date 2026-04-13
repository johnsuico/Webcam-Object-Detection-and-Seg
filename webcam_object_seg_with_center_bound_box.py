import cv2, platform
from ultralytics import YOLO

# Load desired model
# Using prompt-based model for better accuracy in detecting specific classes (e.g., person)
# Uncomment lines below to use yoloe-26l-seg
# ================================
model = YOLO("yoloe-26l-seg.pt")
# Add prompt to only detect the certain items
model.set_classes(["person", "mouse", "keyboard", "guitar", "phone"])
# ================================

# Initalize the webcam
# Check between Windows vs Linux
if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
elif platform.system() == "Linux":
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Success: Webcam opened successfully.")
    print("Press 'q' to exit the webcam feed.")

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Perform object detection and segmentation
    results = model(frame, verbose=False)

    # Extract object output
    detected_objects = set()

    # Check if object is in frame
    if results and len(results[0].boxes) > 0:
        # Get the ID numbers
        class_ids = results[0].boxes.cls.cpu().numpy()

        # Convert IDs to string names using mode.names and add it to detected_objects set
        for class_id in class_ids:
            detected_objects.add(model.names[int(class_id)])
    
    # Print the detected objects in the console
    if len(detected_objects) > 0:
        print(detected_objects)
    if len(detected_objects) == 0:
        print("No objects detected.")

    # Display the results on the frame
    annotated_frame = results[0].plot()

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
            cv2.circle(annotated_frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

    # Display the annotated frame
    cv2.imshow('Webcam Object Segmentation', annotated_frame)

    # Break the loop is 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()