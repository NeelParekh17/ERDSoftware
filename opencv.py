import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "C:/Users/NEEL PAREKH/OneDrive/Desktop/ERD_Schema/Dataset/img1.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edged = cv2.Canny(blurred, 30, 150)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to detect shape
def detect_shape(contour):
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # Calculate bounding box and aspect ratio
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    
    if hull_area == 0:
        return "Unknown"
    
    solidity = float(area) / hull_area
    
    if len(approx) == 4:
        # Use rotated bounding box to determine angle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate width and height of rotated box
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        
        # Calculate the angle of the box
        angle = rect[2]
        if width < height:
            angle = angle + 90
        
        if 45 - 10 < abs(angle) < 45 + 10:
            return "Diamond"
        else:
            return "Rectangle"
    elif len(approx) > 6:
        if solidity > 0.8:
            return "Ellipse"
    return "Unknown"

# Loop over the contours to identify shapes
for contour in contours:
    shape = detect_shape(contour)
    
    # Filter based on area to ignore text and very small shapes
    area = cv2.contourArea(contour)
    if area < 100 or area > 10000:  # Adjust these values as needed
        continue
    
    if shape in ["Rectangle", "Ellipse", "Diamond"]:
        # Draw the contours and label the shapes
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the output image with detected shapes
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()