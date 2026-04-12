import cv2
from ultralytics import YOLO

# Load desired model
# Using prompt-based model for better accuracy in detecting specific classes (e.g., person)
# Uncomment lines below to use yoloe-26l-seg
# ================================
model = YOLO("yoloe-26l-seg.pt")
# Add prompt to only detect the certain items
model.set_classes(["person", "mouse", "keyboard", "guitar", "phone"])
# ================================

# Use yolo26n-seg for faster inference, but it may be less accurate and will detect all classes
# Uncomment line below to use yolo26n-seg
# ================================
# model = YOLO("yolo26n-seg.pt")
# ================================

# Initalize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
    results = model(frame)

    # Display the results on the frame
    annotated_frame = results[0].plot()

    # --- CONFIGURATION ---
    KNOWN_REAL_WORLD_HEIGHT_M = 1.7  # Still calculated in meters for the formula
    FOCAL_LENGTH_PX = 200            
    METERS_TO_FEET = 3.28084         # Conversion factor
    # ---------------------

    # Check if there are any bounding boxes detected
    if results[0].boxes is not None:

        # ==================================================
        # DRAW CENTER POINT OF BOUNDING BOXES
        # ==================================================
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
        # ==================================================
        # END DRAW CENTER POINT OF BOUNDING BOXES
        # ==================================================

        # ==================================================
        # CALCULATE AND DISPLAY DISTANCE
        # ==================================================
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)  # Get bounding boxes in xyxy format

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box

            # Find center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate the pixel height of the bounding box
            pixel_height = y2 - y1

            # Prevent division by zero just in case
            if pixel_height > 0:
                # Calculate distance
                distance_m = (KNOWN_REAL_WORLD_HEIGHT_M * FOCAL_LENGTH_PX) / pixel_height

                # Convert distance to feet
                distance_ft = distance_m * METERS_TO_FEET

                # Display the distance on the annotated frame
                label = f"{distance_ft:.2f} ft"

                # Measure the size of the text to create a background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Calculate where the bottom-left corner should go
                text_x = center_x - (text_width // 2)
                text_y = center_y + (text_height // 2)

                # Put the distance label above the bounding box
                cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # ==================================================
        # END CALCULATE AND DISPLAY DISTANCE
        # ==================================================

    # Display the annotated frame
    cv2.imshow('Webcam Object Segmentation', annotated_frame)

    # Break the loop is 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()