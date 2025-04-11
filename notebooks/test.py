import os
import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

# Load inference model from a given directory
def load_model(model_dir, use_gpu=False):
    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")

    config = Config(model_file, params_file)
    
    if use_gpu:
        config.enable_use_gpu(100, 0)  # 100MB memory pool, GPU id 0
    else:
        config.disable_gpu()
        config.enable_mkldnn()  # Optional: accelerate CPU

    config.enable_memory_optim()
    config.disable_glog_info()
    predictor = create_predictor(config)

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    return predictor, input_names, output_names

# Run inference on a single image
def run_inference(predictor, input_names, output_names, image):
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape([1, 3, image.shape[0], image.shape[1]])
    input_tensor.copy_from_cpu(image)

    predictor.run()
    output_tensor = predictor.get_output_handle(output_names[0])
    results = output_tensor.copy_to_cpu()
    return results

# Load and preprocess image
def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (640, 640))  # Resize to match model input
    image = image[:, :, ::-1].astype('float32') / 255.0  # BGR to RGB + normalize
    image = np.transpose(image, [2, 0, 1])  # HWC -> CHW
    return image

# Visualize detection results
def visualize(image_path, detections_all, label_map):
    image = cv2.imread(image_path)
    for det in detections_all:
        class_id, score, x1, y1, x2, y2 = det
        label = f"{label_map.get(int(class_id), 'Unknown')} {score:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("Layout Detection", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    # === Modify image path here ===
    image_path = "test.jpg"  # Change to your actual image file

    # === Model directories ===
    model_dirs = [
        r"D:\project-pdf\pdf-ocr-dl\models\picodet_lcnet_x1_0_fgd_layout_cdla_infer",
        r"D:\project-pdf\pdf-ocr-dl\models\picodet_lcnet_x1_0_fgd_layout_infer",
        r"D:\project-pdf\pdf-ocr-dl\models\picodet_lcnet_x1_0_fgd_layout_table_infer",
    ]

    # === Enable GPU or not ===
    use_gpu = True  # Set to False if you want to run on CPU

    # === Unified label map (you can customize based on actual model labels) ===
    label_map = {
        0: "Text",
        1: "Title",
        2: "Table",
        3: "Figure",
        4: "List",
        5: "Header",
        6: "Footer",
        7: "Quote",
        8: "Formula",
        9: "Caption"
    }

    # Preprocess input image
    input_image = load_image(image_path)
    detections_all = []

    # Run all models and collect results
    for model_dir in model_dirs:
        predictor, input_names, output_names = load_model(model_dir, use_gpu=use_gpu)
        result = run_inference(predictor, input_names, output_names, input_image)
        
        for det in result[0]:
            if det[2] > 0.4:  # confidence threshold
                detections_all.append(det[1:])  # remove batch id

    # Show detection boxes
    visualize(image_path, detections_all, label_map)
