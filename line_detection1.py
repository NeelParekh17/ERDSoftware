import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained YOLO model for object detection
model = YOLO("runs/detect/train/weights/best.pt")  # Adjust the path as necessary

# Step 1: Load the image
image_path = 'final_dataset/obj_train_data/img51.png'  # Adjust the path as necessary
img = cv2.imread(image_path)

# Check if image loaded correctly
if img is None:
    raise ValueError(f"Image not found at the path: {image_path}")

# Step 2: Preprocess the image (Convert to grayscale and apply Gaussian blur)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Apply Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Step 4: Run YOLO for object detection (entities, attributes, relationships)
results = model(image_path)

# Extract bounding boxes
bboxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
classes = results[0].boxes.cls.cpu().numpy()  # Get class labels
class_names = results[0].names  # Class names (Entity, Attribute, Relationship, etc.)

# Step 5: Apply Hough Line Transform for line detection
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Copy the original image for drawing lines
line_img = img.copy()

# Step 6: Function to check if a line is inside a bounding box
def is_line_inside_bbox(line, bbox):
    x1, y1, x2, y2 = line
    x_min, y_min, x_max, y_max = bbox
    return (x1 > x_min and x1 < x_max and y1 > y_min and y1 < y_max) and \
           (x2 > x_min and x2 < x_max and y2 > y_min and y2 < y_max)

# Step 7: Classify lines as part of the object or as connecting lines
if lines is not None:  # Check if any lines were detected
    for line in lines:
        x1, y1, x2, y2 = line[0]
        is_inside_box = False
        
        # Check if the line is inside any bounding box
        for bbox in bboxes:
            if is_line_inside_bbox([x1, y1, x2, y2], bbox):
                is_inside_box = True
                break

        if not is_inside_box:
            # Line is outside bounding boxes -> considered a connecting line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for connecting lines
        else:
            # Line is inside a bounding box -> considered part of the entity/attribute
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for lines inside boxes

# Optional: Save the result
cv2.imwrite('output_with_lines.png', line_img)