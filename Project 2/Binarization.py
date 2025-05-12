import cv2
import numpy as np
from PIL import Image
import os
import time

# Configuration
INPUT_DIR = "documents/original_images/"
OUTPUT_DIR = "documents/binarized_images/"
IMAGE_RANGE = range(270, 310)  # Only process 305-309
WINDOW_SIZE = 15  # Sauvola window size
K = 0.06         # Sauvola k parameter

def sauvola_binarization(image_path):
    """Apply Sauvola binarization using OpenCV"""
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Sauvola thresholding
    binary = cv2.ximgproc.niBlackThreshold(
        img,
        maxValue=255,
        type=cv2.THRESH_BINARY,
        blockSize=WINDOW_SIZE,
        k=K,
        binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA
    )
    return binary

def process_images():
    """Process all images in the specified range"""
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting binarization...")
    start_time = time.time()
    
    for i in IMAGE_RANGE:
        input_path = os.path.join(INPUT_DIR, f"{i}.jpg")
        output_path = os.path.join(OUTPUT_DIR, f"{i}b.jpg")
        
        # Skip if input doesn't exist
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping")
            continue
            
        try:
            # Process and save image
            binary_img = sauvola_binarization(input_path)
            Image.fromarray(binary_img).save(output_path)
            print(f"Processed {i}.jpg")
        except Exception as e:
            print(f"Error processing {i}.jpg: {str(e)}")
    
    print(f"Binarization completed in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    # Check if processing is needed
    existing_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    
    if existing_files < len(IMAGE_RANGE):
        process_images()
    else:
        print("All images already processed")