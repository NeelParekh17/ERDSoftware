import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('neel/img1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Optional: Use Canny edge detection (if you have a noisy image)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Use Hough Line Transform to detect lines
lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

# If lines are detected
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Save the result
cv.imwrite('neel/detected_lines.png', img)
print("Line detection complete. Result saved as 'detected_lines.png'.")
