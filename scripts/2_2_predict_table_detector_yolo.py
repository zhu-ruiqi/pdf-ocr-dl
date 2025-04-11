import os
import shutil
import subprocess

def run_yolo_predict(
    weights_path="yolov5/runs/train/table-detector-test/weights/best.pt",
    source_path="data/yolo_table_dataset/images/train/",
    output_dir="data/outputs/yolo_preds",
    conf_threshold=0.3
):
    """
    Run YOLOv5 table detection and save predictions to output folder.

    Args:
        weights_path (str): Path to trained YOLOv5 model (.pt file).
        source_path (str): Folder containing input images.
        output_dir (str): Where to save YOLOv5 predictions.
        conf_threshold (float): Confidence threshold.
    """
    current_dir = os.path.abspath(os.getcwd())
    yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../yolov5"))
    source_abs = os.path.abspath(os.path.join(current_dir, source_path))
    output_abs = os.path.abspath(os.path.join(current_dir, output_dir))

    # Step 1: Change directory to YOLOv5
    os.chdir(yolo_path)

    # Step 2: Clear old results
    detect_dir = os.path.join("runs", "detect")
    if os.path.exists(detect_dir):
        shutil.rmtree(detect_dir)

    # Step 3: Run detection
    command = [
        "python", "detect.py",
        "--weights", weights_path,
        "--img", "640",
        "--conf", str(conf_threshold),
        "--source", source_abs
    ]
    print(f"[INFO] Running YOLOv5 detection...\n{' '.join(command)}")
    subprocess.run(command, check=True)

    # Step 4: Copy predictions
    latest_output = os.path.join("runs", "detect", "exp")
    if not os.path.exists(latest_output):
        raise FileNotFoundError("[ERROR] YOLOv5 did not produce expected output in 'runs/detect/exp'.")

    os.makedirs(output_abs, exist_ok=True)
    saved_files = 0
    for file in os.listdir(latest_output):
        if file.endswith((".jpg", ".png")):
            shutil.copy(os.path.join(latest_output, file), os.path.join(output_abs, file))
            saved_files += 1

    print(f"[INFO] Copied {saved_files} prediction images to: {output_abs}")

    # Step 5: Return to original directory
    os.chdir(current_dir)

if __name__ == "__main__":
    run_yolo_predict()
