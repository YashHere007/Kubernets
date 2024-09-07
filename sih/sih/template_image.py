import cv2
import numpy as np
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load the OCR model
model = ocr_predictor(pretrained=True)

# Streamlit GUI
st.title("Template Matching and Text Extraction")

# File uploader
uploaded_file = st.file_uploader("Choose an input image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    input_img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(input_img, cv2.IMREAD_GRAYSCALE)

    # Load the template image
    template_img = cv2.imread('C:/Users/myash/Downloads/gate.png', cv2.IMREAD_GRAYSCALE)

    # Ensure input image is larger than the template image
    if input_img.shape[0] < template_img.shape[0] or input_img.shape[1] < template_img.shape[1]:
        scale_ratio = max(template_img.shape[0] / input_img.shape[0], 
                          template_img.shape[1] / input_img.shape[1])
        input_img = cv2.resize(input_img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

    # Apply thresholding to improve image quality
    _, template_thresh = cv2.threshold(template_img, 150, 255, cv2.THRESH_BINARY_INV)
    _, input_thresh = cv2.threshold(input_img, 150, 255, cv2.THRESH_BINARY_INV)

    # Template matching
    result = cv2.matchTemplate(input_thresh, template_thresh, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # Define the threshold for a valid match
    threshold = 0.8

    if max_val >= threshold:
        st.success("The input image matches the template!")

        # Regions to extract from the input image (define your regions here)
        regions_to_extract = [
            (240, 320, 400, 480),  # Adjust coordinates based on the region
            (550, 610, 350, 750),
            (70, 105, 423, 132),
            (67, 156, 222, 177),
            (176, 259, 253, 280),
            # Add more regions as needed
        ]

        extracted_texts = []

        for region in regions_to_extract:
            x_start, x_end, y_start, y_end = region
            region_img = input_img[y_start:y_end, x_start:x_end]  # Corrected indexing

            # Extract text using DocTR
            doc = DocumentFile.from_images([region_img])
            result = model(doc)
            extracted_text = result.pages[0].content
            extracted_texts.append(extracted_text)

        # Display extracted text
        st.subheader("Extracted Text:")
        for idx, text in enumerate(extracted_texts):
            st.write(f"Region {idx + 1}: {text}")

    else:
        st.error("Invalid Image: The input image does not match the template.")
