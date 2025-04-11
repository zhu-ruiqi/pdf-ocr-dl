import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddle.inference import Config, create_predictor

# ========== CONFIG ==========
PDF_PATH = "pdf-ocr-dl/data/raw_pdfs/table_sample_en.pdf"
OUTPUT_DIR = "pdf-ocr-dl/data/outputs/layout_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_GPU = False  # ← Change to True if you want to use GPU

# Label map (customize this if needed)
label_map = {
    0: "Text", 1: "Title", 2: "Table", 3: "Figure", 4: "List",
    5: "Header", 6: "Footer", 7: "Quote", 8: "Formula", 9: "Caption"
}

# Inference model folders
model_dirs = [
    r"D:\project-pdf\pdf-ocr-dl\models\picodet_lcnet_x1_0_fgd_layout_cdla_infer",
    r"D:\project-pdf\pdf-ocr-dl\models\picodet_lcnet_x1_0_fgd_layout_infer",
    r"D:\project-pdf\pdf-ocr-dl\models\picodet_lcnet_x1_0_fgd_layout_table_infer",
]

# ========== STEP 1: Convert PDF to images ==========
def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=150)
    image_paths = []
    for idx, image in enumerate(images):
        img_path = os.path.join(OUTPUT_DIR, f"page_{idx+1}.jpg")
        image.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths

# ========== STEP 2: Load inference model ==========
def load_model(model_dir, use_gpu=False):
    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")
    config = Config(model_file, params_file)

    if use_gpu:
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
        config.enable_mkldnn()

    config.enable_memory_optim()
    config.disable_glog_info()
    predictor = create_predictor(config)
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    return predictor, input_names, output_names

# ========== STEP 3: Preprocess image ==========
def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (640, 640))  # Resize for model
    image = image[:, :, ::-1].astype('float32') / 255.0
    image = np.transpose(image, [2, 0, 1])
    return image

# ========== STEP 4: Run inference ==========
def run_inference(predictor, input_names, output_names, image):
    image = np.expand_dims(image, axis=0)  
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(image.shape) 
    input_tensor.copy_from_cpu(image)

    predictor.run()
    output_tensor = predictor.get_output_handle(output_names[0])
    results = output_tensor.copy_to_cpu()
    return results

def parse_detections_safely(result, conf_thresh=0.4):
    """
    Parse model output into usable detection results.
    Avoid index errors and handle empty results.
    Returns: list of [class_id, score, x1, y1, x2, y2]
    """
    detections = []
    try:
        raw = result[0]
        if len(raw) == 0:
            print("[ℹ] No objects detected.")
            return []
        for det in raw:
            if isinstance(det, (list, np.ndarray)) and len(det) >= 6:
                class_id, score, x1, y1, x2, y2 = det[1:7]
                if score > conf_thresh:
                    detections.append([class_id, score, x1, y1, x2, y2])
            else:
                print(f"[⚠] Skipping invalid detection format: {det}")
    except Exception as e:
        print(f"[❌] Error parsing detection results: {e}")
    return detections

# ========== STEP 5: Visualize ==========
def visualize(img_path, detections_all, label_map):
    image = cv2.imread(img_path)
    for det in detections_all:
        class_id, score, x1, y1, x2, y2 = det
        label = f"{label_map.get(int(class_id), 'Unknown')} {score:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    save_path = img_path.replace(".jpg", "_detected.jpg")
    cv2.imwrite(save_path, image)
    print(f"[✔] Saved: {save_path}")

# ========== STEP 6: Analyze layout ==========
def analyze_layout_on_image(img_path):
    input_image = load_image(img_path)
    detections_all = []

    for model_dir in model_dirs:
        predictor, input_names, output_names = load_model(model_dir, use_gpu=USE_GPU)
        result = run_inference(predictor, input_names, output_names, input_image)
        detections = parse_detections_safely(result)
        detections_all.extend(detections) # remove batch_id

    visualize(img_path, detections_all, label_map)

# ========== MAIN ==========
if __name__ == "__main__":
    image_paths = convert_pdf_to_images(PDF_PATH)
    for img_path in image_paths:
        analyze_layout_on_image(img_path)
