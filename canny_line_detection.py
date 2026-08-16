from __future__ import print_function
import cv2 as cv
import argparse

# Parameters
max_lowThreshold = 100
ratio = 3
kernel_size = 3

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))  # Apply blur to the grayscale image
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)  # Canny edge detection
    mask = detected_edges != 0  # Create mask where edges were detected
    dst = src * (mask[:,:,None].astype(src.dtype))  # Mask original image to show edges
    # Save the result as an image file instead of displaying it
    cv.imwrite('neel/detected_edges_output.png', dst)
    print("Edge detection complete. Result saved as 'detected_edges_output.png'.")

# Parsing arguments (you can skip this if running manually)
parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to the input image.', default='neel/img1.png')
args = parser.parse_args()

# Load the input image
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Convert image to grayscale
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Run edge detection
CannyThreshold(0)