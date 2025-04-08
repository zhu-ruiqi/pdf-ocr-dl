import streamlit as st
from pdfminer.high_level import extract_text
from PIL import Image
import io

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    text = extract_text(pdf_file)
    return text

# Function to extract images from PDF (for this example, we will just display an image from file upload)
def extract_pdf_images(pdf_file):
    image = Image.open(pdf_file)
    return image

# Streamlit App
def main():
    st.title("PDF Text and Image Extraction")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        # Extract text
        text = extract_pdf_text(uploaded_file)
        st.subheader("Extracted Text:")
        st.text(text)

        # Extract and display images (just for the sake of example, we display images here)
        st.subheader("Extracted Images:")
        image = extract_pdf_images(uploaded_file)
        st.image(image, caption="Extracted Image", use_column_width=True)

if __name__ == "__main__":
    main()
