import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_dir="pdf-ocr-dl/data/outputs/images"):
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_extracted = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
                total_extracted += 1

    print(f"[SUCCESS] Extracted {total_extracted} images to: {output_dir}")

if __name__ == "__main__":
    pdf_path = "pdf-ocr-dl/data/raw_pdfs/sample2.pdf"
    extract_images_from_pdf(pdf_path)
