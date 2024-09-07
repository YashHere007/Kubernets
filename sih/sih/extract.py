import cv2
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import streamlit as st
from PIL import Image
import numpy as np
import os

# Function to check if the image is of low quality
def is_low_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100

# Function to enhance image clarity
def enhance_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding for better contrast
    enhanced_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    
    # Apply sharpening if image is detected as low quality
    if is_low_quality(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
        
    # Resize image to ensure better OCR results
    height, width = enhanced_image.shape[:2]
    scale_factor = min(1200/width, 1200/height)  # Resize to a max dimension of 1200 pixels
    enhanced_image = cv2.resize(enhanced_image, (int(width*scale_factor), int(height*scale_factor)))
    
    # Denoising the image
    enhanced_image = cv2.fastNlMeansDenoising(enhanced_image, None, 30, 7, 21)
    
    return enhanced_image

# Function to extract text from an image using DocTR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    enhanced_image = enhance_image(image)
    
    temp_image_path = "enhanced_temp_image.jpg"
    cv2.imwrite(temp_image_path, enhanced_image)
    
    doc = DocumentFile.from_images(temp_image_path)
    
    # Using a supported OCR model
    model = ocr_predictor("db_resnet50", pretrained=True)

    result = model(doc)
    
    text = ''
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                text += ' '.join([word.value for word in line.words]) + '\n'
    
    os.remove(temp_image_path)
    
    return text

# Streamlit UI
st.title("Text Extracon from Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image_np)
    
    st.write("Extracting text...")
    text = extract_text_from_image(temp_image_path)
    
    st.text_area("Extracted Text", text)
    
    st.write("Text extraction completed!")
