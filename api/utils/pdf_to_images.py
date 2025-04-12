from api.utils import path_manager as paths
import os
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path: str, output_folder: str = "./outputs/images"):
    """
    Convert PDF to page-level images.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Directory to save the output images.

    Returns:
        List[str]: List of image file paths.
    """
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # High resolution
        img_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths
