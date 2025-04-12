from api.utils import path_manager as paths
import os
from paddleocr import PPStructure, save_structure_res
import cv2

def analyze_pdf_layout(image_path: str, output_folder: str = os.path.join(paths.OUTPUT_DIR, "layout")):
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

    # # Analyze layout
    # result = table_engine(image)

    # # Save results
    # base_name = os.path.splitext(os.path.basename(image_path))[0]
    # save_structure_res(result, output_folder, base_name)
    # return result

    # Analyze layout
    raw_result = table_engine(image)

    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_structure_res(raw_result, output_folder, base_name)

    # ✅ to JSON 
    final_result = []
    for block in raw_result:
        final_result.append({
            "type": block.get("type", "unknown"),
            "bbox": [float(x) for x in block.get("bbox", [])],
            "text": block.get("res", "") if isinstance(block.get("res"), str) else str(block.get("res"))
        })
    return final_result

    
