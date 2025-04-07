import os
import shutil

def run_yolo_predict(
    weights_path="yolov5/runs/train/table-detector-test/weights/best.pt",
    source_path="pdf-ocr-dl/data/yolo_table_dataset/images/train/",
    output_dir="pdf-ocr-dl/data/outputs/yolo_preds",
    conf_threshold=0.3
):

    # os.chdir("yolov5")
    yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))
    os.chdir(yolo_path)

    if os.path.exists("runs/detect"):
        shutil.rmtree("runs/detect")

    command = (
        f"python detect.py --weights {weights_path} "
        f"--img 640 --conf {conf_threshold} "
        f"--source ../{source_path}"
    )

    print(f"[INFO] Running YOLOv5 detection...\n{command}")
    os.system(command)

    latest_output = "runs/detect/exp"
    os.makedirs(f"../{output_dir}", exist_ok=True)
    for file in os.listdir(latest_output):
        if file.endswith((".jpg", ".png")):
            shutil.copy(os.path.join(latest_output, file), f"../{output_dir}/{file}")

    print(f"[INFO] Predictions saved to: {output_dir}")

    os.chdir("..") 

if __name__ == "__main__":
    run_yolo_predict()
