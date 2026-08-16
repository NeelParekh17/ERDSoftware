import cv2
import numpy as np

# Load the image
img = cv2.imread('neel/img1.png', cv2.IMREAD_GRAYSCALE)

# Use edge detection
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# Use Hough Line Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

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
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Save the result to a file instead of displaying it
cv2.imwrite('neel/detected_lines.png', img)
print("Lines detected and saved as 'detected_lines.png'.")
