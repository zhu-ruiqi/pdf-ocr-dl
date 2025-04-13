import fitz  # PyMuPDF
from io import BytesIO

def classify_pdf_type(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")

    total_pages = 0
    text_pages = 0
    scan_pages = 0
    mixed_pages = 0

    for page in doc:
        text = page.get_text().strip()
        image_count = len(page.get_images(full=True))
        total_pages += 1

        if len(text) < 100 and image_count > 0:
            scan_pages += 1
        elif len(text) > 300 and image_count == 0:
            text_pages += 1
        else:
            mixed_pages += 1

    if scan_pages / total_pages > 0.7:
        return "scan_pdf"
    elif text_pages / total_pages > 0.7:
        return "text_pdf"
    else:
        return "mixed_pdf"
