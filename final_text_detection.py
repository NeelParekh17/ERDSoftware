import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the pre-trained model and processor locally
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def query_local(filename):
    """Process an image file and perform OCR locally."""
    try:
        # Open the image
        image = Image.open(filename).convert("RGB")
        
        # Preprocess the image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # Generate text from the image
        generated_ids = model.generate(pixel_values)
        
        # Decode the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Remove any double quotes from the text
        generated_text = generated_text.replace('"', '')
        
        return generated_text
    except Exception as e:
        return {"Error": f"Error processing {filename}: {str(e)}"}

def get_type_from_filename(filename):
    """Determine the type based on the filename."""
    if "multivalued attribute" in filename.lower():
        return 'Multivalued Attribute'
    elif "derived attribute" in filename.lower():
        return 'Derived Attribute'
    elif "attribute" in filename.lower():
        return 'Attribute'
    elif "weak entity" in filename.lower():
        return 'Weak Entity'
    elif "entity" in filename.lower():
        return 'Entity'
    elif "weak relationship" in filename.lower():
        return 'Weak Relationship'
    elif "relationship" in filename.lower():
        return 'Relationship'
    elif "primary-key" in filename.lower():
        return 'Primary Key'
    elif "partial key" in filename.lower():
        return 'Partial Key'
    else:
        return 'Unknown'

def process_images_in_folder(folder_path):
    """Process all images in the specified folder."""
    results = []

    # Open a log file for debugging
    with open("debug_log.txt", "w") as log_file:
        # Loop through all image files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)

                try:
                    # Query the local model
                    output = query_local(image_path)
                    log_file.write(f"File: {filename}, Recognized Text: {output}\n")

                    extracted_info = {
                        'Entity': '',
                        'Relationship': '',
                        'Weak Entity': '',
                        'Primary Key': '',
                        'Derived Attribute': '',
                        'Attribute': '',
                        'Weak Relationship': '',
                        'Multivalued Attribute': '',
                        'Partial Key': ''
                    }

                    # Determine the category based on the filename
                    category = get_type_from_filename(filename)

                    # Process valid text only, skip 'Unknown' category
                    if output and category != 'Unknown':
                        # Remove double quotes from the text
                        extracted_info[category] = output.strip().replace('"', '')  # Double quotes removed
                        results.append(extracted_info)
                    else:
                        log_file.write(f"Skipping {filename} due to 'Unknown' category or empty output.\n")
                except Exception as e:
                    results.append({"Error": f"Error processing {filename}: {str(e)}"})
                    log_file.write(f"Error processing {filename}: {str(e)}\n")

    return results

# Path to the folder containing images
folder_path = "C:/Users/NEEL PAREKH/OneDrive/Desktop/ERD_Schema/bounding_box"

# Process images in the folder
extracted_results = process_images_in_folder(folder_path)

# Print the extracted results for verification
for result in extracted_results:
    print(result)

# Save the results to a file
with open("extracted_words.txt", "w") as f:
    for result in extracted_results:
        f.write(str(result) + "\n")