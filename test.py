import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import cv2
from prettytable import PrettyTable

# Paths
input_folder = 'neel'
output_folder = 'C:/Users/NEEL PAREKH/OneDrive/Desktop/testpreprocessing/processed_data/'

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the list for rows
rows = []

# 1. Adjust Image Size for YOLOv8
def adjust_image_size(image_path, output_path, size=(960, 960)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS for high-quality downsampling
        img.save(output_path)

# 2. Data Augmentation (Improved Version)
def augment_image(image_path, output_folder):
    img = Image.open(image_path)
    augmented_paths = []
    
    # Define possible augmentations
    def random_flip(img):
        if np.random.rand() > 0.5:
            img = ImageOps.mirror(img)  # Horizontal flip
        if np.random.rand() > 0.5:
            img = ImageOps.flip(img)  # Vertical flip
        return img
    
    def random_rotation(img):
        angle = random.choice([90, 270])
        img = img.rotate(angle)
        return img
    
    
    # Apply augmentations and save images
    for i in range(2):  # Save 2 augmented versions
        augmented_img = img.copy()
        
        # Apply augmentations
        augmented_img = random_flip(augmented_img)
        augmented_img = random_rotation(augmented_img)
        
        # Save the augmented image
        augmented_filename = f'augmented_{i}_{os.path.basename(image_path)}'
        augmented_img.save(os.path.join(output_folder, augmented_filename))
        augmented_paths.append(augmented_filename)
    
    return augmented_paths

# 3. Normalize Data
def normalize_image(image_path, output_path):
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = (img_array * 255).astype(np.uint8)  # Convert back to 0-255 range
    img = Image.fromarray(img_array)
    img.save(output_path)

# 4. Grayscale Conversion
def convert_to_grayscale(image_path, output_path):
    with Image.open(image_path) as img:
        gray_img = img.convert('L')  # Convert to grayscale
        gray_img.save(output_path)

# Function to calculate image metrics
def calculate_metrics(image_path):
    img = cv2.imread(image_path)
    dimensions = img.shape[:2]  # (Height, Width)
    file_size_kb = os.path.getsize(image_path) / 1024  # File size in KB
    mean_intensity = np.mean(img)
    stddev_intensity = np.std(img)
    return dimensions, file_size_kb, mean_intensity, stddev_intensity

# Process images and update the table
for file_name in os.listdir(input_folder):
    if file_name.endswith('.png') or file_name.endswith('.jpeg'):
        file_path = os.path.join(input_folder, file_name)
        
        # Calculate metrics for original image
        original_dimensions, original_size_kb, original_mean, original_stddev = calculate_metrics(file_path)
        
        # Adjust image size
        resized_path = os.path.join(output_folder, f'resized_{file_name}')
        adjust_image_size(file_path, resized_path)
        
        # Normalize data
        normalized_path = os.path.join(output_folder, f'normalized_{file_name}')
        normalize_image(resized_path, normalized_path)
        
        # Convert to grayscale
        grayscale_path = os.path.join(output_folder, f'grayscale_{file_name}')
        convert_to_grayscale(normalized_path, grayscale_path)
        
        # Perform data augmentation
        augment_folder = os.path.join(output_folder, 'augmented_images')
        if not os.path.exists(augment_folder):
            os.makedirs(augment_folder)
        augment_paths = augment_image(grayscale_path, augment_folder)
        
        # Calculate metrics for processed image
        processed_dimensions, processed_size_kb, processed_mean, processed_stddev = calculate_metrics(grayscale_path)
        
        # Add data to rows
        rows.append([
            file_name,
            f'{original_dimensions[1]}x{original_dimensions[0]}',
            f'{processed_dimensions[1]}x{processed_dimensions[0]}',
            f'{original_size_kb:.2f}',
            f'{processed_size_kb:.2f}',
            f'{original_mean:.2f}',
            f'{original_stddev:.2f}',
            'Processed and Augmented'
        ])

# Create and display the table
columns = ['File Name', 'Original Dimensions', 'Resized Dimensions', 'Original Size (KB)', 'Processed Size (KB)',
           'Mean Pixel Intensity', 'Standard Deviation', 'Visual Quality']

table = PrettyTable(columns)
for row in rows:
    table.add_row(row)

print(table)

# Save the dataframe to a CSV file
df = pd.DataFrame(rows, columns=columns)
df.to_csv('preprocessing_evaluation_416.csv', index=False)
