import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += f"\n--- Page {page_num} ---\n{text}"
    doc.close()
    return full_text

if __name__ == "__main__":
    path = "pdf-ocr-dl/data/raw_pdfs/sample2.pdf"  
    result = extract_text_from_pdf(path)
    print(result[:1000])  

