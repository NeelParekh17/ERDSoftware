import cv2
import pytesseract
import re
import os
from collections import defaultdict

# Path to the Tesseract executable (update this path as per your installation)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Path to the folder containing the cropped bounding box images
bounding_box_folder = "bounding_box"

# Initialize a dictionary to store detected text by type
detected_text_by_type = defaultdict(list)

# Open a file to write the results
with open("bounding_box/ocr_results.txt", "w") as result_file:
    # Loop through all the images in the bounding_box folder
    for file_name in os.listdir(bounding_box_folder):
        if file_name.endswith('.png'):  # Process only PNG files
            # Extract the box type from the file name (assuming format like 'Entity_1.png')
            match = re.match(r"(\w+)_\d+\.png", file_name)
            if match:
                box_type = match.group(1)

                # Skip 'line' type boxes
                if box_type.lower() == 'line':
                    continue

                # Path to the cropped image
                image_path = os.path.join(bounding_box_folder, file_name)

                # Load the image
                image = cv2.imread(image_path)

                # Perform OCR on the cropped image
                text = pytesseract.image_to_string(image)
                clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text).strip()

                # Write filename and detected text to the file
                if clean_text:
                    result_file.write(f"{file_name} -- {clean_text}\n")
                else:
                    # If no text detected, print "Nothing detected" and the file name
                    result_file.write(f"{file_name} -- Nothing detected\n")
                    print(f"Nothing detected for file: {file_name}")
            # else:
            #     result_file.write(f"{file_name} -- Filename format not recognized\n")

print("OCR results saved to ocr_results.txt")
