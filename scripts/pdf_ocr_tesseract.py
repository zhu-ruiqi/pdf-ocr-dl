import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r"D:\projects\tools\tesseract-ocr\tesseract.exe"

def ocr_pdf_tesseract(pdf_path, lang="eng", save_txt=True):
    """
    Perform OCR on a PDF file using Tesseract.

    Args:
        pdf_path (str): Path to the input PDF file.
        lang (str): Language(s) for OCR, e.g., "eng", "chi_sim", or "eng+chi_sim".
        save_txt (bool): Whether to save the output text to a .txt file.

    Returns:
        str: Extracted text from the entire PDF.
    """
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return ""

    print(f"[INFO] Processing PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=300)
    print(f"[INFO] Total pages: {len(images)}")

    all_text = ""
    for i, img in enumerate(images):
        print(f"[INFO] Performing OCR on page {i+1}...")
        text = pytesseract.image_to_string(img, lang=lang)
        all_text += f"\n--- Page {i+1} ---\n{text}"

    if save_txt:
        txt_path = pdf_path.replace(".pdf", "_ocr_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"[INFO] OCR result saved to: {txt_path}")

    return all_text

if __name__ == "__main__":
    pdf_path = "pdf-ocr-dl/data/raw_pdfs/scanned_sample_file_en.pdf"  
    text = ocr_pdf_tesseract(pdf_path, lang="eng+chi_sim")
    print(text[:1000])