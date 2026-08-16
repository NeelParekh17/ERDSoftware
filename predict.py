from ultralytics import YOLO
import cv2
import os

# Define the class names in the order you specified
class_names = [
    "relationship", "weak entity", "primary-key", "Entity", "total", 
    "line", "derived attribute", "attribute", "weak relationship", 
    "multivalued attribute", "partial key"
]

# Load the model
model = YOLO("runs/detect/train/weights/best.pt")

# Path to the image and output folder
image_path = "C:/Users/NEEL PAREKH/OneDrive/Desktop/ERD_Schema/Dataset/img1.png"
output_folder = "bounding_box"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Run prediction on the image
results = model(image_path, imgsz=640)

# Load the image for saving bounding boxes later
image = cv2.imread(image_path)

# Open a file to write the predictions
with open(os.path.join(output_folder, "predicted_output.txt"), "w") as f:
    for result in results:
        for i, box in enumerate(result.boxes):
            # Get class ID and map it to the class name
            class_id = int(box.cls)
            class_name = class_names[class_id]

            # Exclude "line" class
            if class_name == "line":
                continue

            # Get the bounding box coordinates (xywh)
            box_xywh = box.xywh.squeeze().tolist()  # Convert tensor to a flat list
            if len(box_xywh) == 4:
                # Extract the values
                x_center, y_center, width, height = box_xywh

                # Calculate top-left corner (x1, y1) and bottom-right (x2, y2)
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Write the class name and bounding box information to the file
                f.write(f"{class_name}: x_center={x_center}, y_center={y_center}, width={width}, height={height}\n")

                # Crop the bounding box area from the image
                cropped_img = image[y1:y2, x1:x2]

                # Save the cropped image with a name based on the class and index
                cropped_image_path = os.path.join(output_folder, f"{class_name}_{i+1}.png")
                cv2.imwrite(cropped_image_path, cropped_img)
            else:
                f.write(f"{class_name}: Invalid bounding box dimensions {box_xywh}\n")

print("Predictions saved to predicted_output.txt and images saved in bounding_box folder.")
