import os
from paddleocr import PPStructure, save_structure_res
import cv2

def analyze_pdf_layout(image_path: str, output_folder: str = "./output"):
    """
    Run layout analysis on a single image (converted from PDF page).

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Directory to save structured outputs.

    Returns:
        list: A list of layout elements with type, bbox, and content.
    """
    # Create output directory if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Load layout analysis engine
    table_engine = PPStructure(layout=True, show_log=True, ocr=True)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image from: {image_path}")

    # Analyze layout
    result = table_engine(image)

    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_structure_res(result, output_folder, base_name)

    return result

if __name__ == "__main__":
    # test_image = "pdf-ocr-dl/data/outputs/images/page_1.png"
    # output_dir = "pdf-ocr-dl/data/outputs/layout_analysis"
    # res = analyze_pdf_layout(test_image, output_dir)
    # print(f"Analysis result for {test_image}:\n", res)
    image_folder = "pdf-ocr-dl/data/outputs/pages"
    output_dir = "pdf-ocr-dl/data/outputs/layout_analysis"
    # Iterate from page_1.png to page_15.png
    for page_num in range(4, 5):
        image_filename = f"page_{page_num}.png"
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"[WARN] Image not found: {image_path}")
            continue

        try:
            res = analyze_pdf_layout(image_path, output_dir)
            print(f"[OK] Finished page {page_num}")
        except Exception as e:
            print(f"[ERROR] Failed page {page_num}: {e}")