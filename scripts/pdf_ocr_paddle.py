import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import cv2
from pdf2image import convert_from_path
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

from scripts.ocr_postprocess import clean_ocr_text

# Path to a Chinese TTF font file (you can change it if needed)
FONT_PATH = r"C:\Windows\Fonts\simfang.ttf"

# Initialize PaddleOCR (Chinese + English)
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch')  # 'ch' includes Chinese and English


def visualize_ocr_result(pil_img, result, save_path):
    """
    Draw OCR results on the image and save it.
    """
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    vis_img = draw_ocr(img_cv, boxes, txts, scores, font_path=FONT_PATH)
    cv2.imwrite(save_path, vis_img)

def ocr_pdf_paddle(pdf_path, save_txt=True, visualize=True):
    """
    Perform OCR on a PDF file using PaddleOCR.
    """
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return ""

    print(f"[INFO] Processing PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=300)
    print(f"[INFO] Total pages: {len(images)}")

    all_text = ""
    output_dir = os.path.join(os.path.dirname(pdf_path), "ocr_outputs")
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        print(f"[INFO] OCR page {i + 1}...")

        img_path = os.path.join(output_dir, f"temp_page_{i}.jpg")
        vis_path = os.path.join(output_dir, f"ocr_page_{i + 1}.jpg")

        img.save(img_path, "JPEG")
        result = ocr_engine.ocr(img_path, cls=True)
        page_text = "\n".join([line[1][0] for line in result[0]])
        all_text += f"\n--- Page {i + 1} ---\n{page_text}"

        if visualize:
            visualize_ocr_result(img, result[0], vis_path)
            print(f"[INFO] Saved visualized image to: {vis_path}")

        os.remove(img_path)

    if save_txt:
        txt_path = pdf_path.replace(".pdf", "_paddleocr_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"[SUCCESS] OCR result saved to: {txt_path}")

    return all_text


if __name__ == "__main__":
    pdf_path = "pdf-ocr-dl/data/raw_pdfs/scanned_sample_file_cn.pdf"
    text = ocr_pdf_paddle(pdf_path)
    print(text[:1000])

    cleaned_text = clean_ocr_text(text)
    print("\n[INFO] Cleaned OCR result (preview):\n")
    print(cleaned_text[:1000])
