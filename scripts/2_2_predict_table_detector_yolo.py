import os
import shutil
import subprocess

def run_yolo_predict(
    weights_path="runs/train/table-detector/weights/best.pt",
    source_path="../data/yolo_table_dataset/images/train",
    output_dir="../data/outputs/yolo_preds",
    conf_threshold=0.3
):
    """
    Run YOLOv5 detection on images and save predictions to the output directory.

    Args:
        weights_path (str): Path to the trained YOLOv5 weights (.pt file).
        source_path (str): Directory containing input images.
        output_dir (str): Directory to save prediction results (annotated images).
        conf_threshold (float): Confidence threshold for detections.
    """
    # Prepare absolute paths
    current_dir = os.path.abspath(os.getcwd())
    yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../yolov5"))
    weights_abs = os.path.abspath(os.path.join(yolo_path, weights_path))
    source_abs = os.path.abspath(os.path.join(yolo_path, source_path))
    output_abs = os.path.abspath(os.path.join(yolo_path, output_dir))

    # Check if model and input folder exist
    if not os.path.exists(weights_abs):
        raise FileNotFoundError(f"[ERROR] Model weights not found: {weights_abs}")
    if not os.path.exists(source_abs):
        raise FileNotFoundError(f"[ERROR] Image source directory not found: {source_abs}")

    # Change working directory to YOLOv5
    os.chdir(yolo_path)

    # Clear previous detection results
    detect_dir = os.path.join("runs", "detect")
    if os.path.exists(detect_dir):
        shutil.rmtree(detect_dir)

    # Construct YOLOv5 detection command
    command = [
        "python", "detect.py",
        "--weights", weights_abs,
        "--img", "640",
        "--conf", str(conf_threshold),
        "--source", source_abs
    ]

    print(f"[INFO] Running YOLOv5 detection...\n{' '.join(command)}")
    subprocess.run(command, check=True)

    # Locate and copy output images to the target directory
    latest_output = os.path.join("runs", "detect", "exp")
    if not os.path.exists(latest_output):
        raise FileNotFoundError("[ERROR] YOLOv5 did not generate output in 'runs/detect/exp'.")

    os.makedirs(output_abs, exist_ok=True)
    saved_files = 0
    for file in os.listdir(latest_output):
        if file.endswith((".jpg", ".png")):
            shutil.copy(os.path.join(latest_output, file), os.path.join(output_abs, file))
            saved_files += 1

    print(f"[✅ DONE] Saved {saved_files} prediction images to: {output_abs}")

    # Return to the original working directory
    os.chdir(current_dir)

if __name__ == "__main__":
    run_yolo_predict()
