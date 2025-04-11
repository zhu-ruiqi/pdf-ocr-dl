import os
import subprocess

def train_yolo_table_detector(
    weights="yolov5s.pt",
    epochs=100,
    batch_size=16,
    img_size=640,
    project="runs/train",
    name="table-detector"
):
    """
    Train YOLOv5 model for table detection.

    Args:
        weights (str): Pretrained weights to start from (e.g., yolov5s.pt).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size per iteration.
        img_size (int): Input image size.
        project (str): Project directory to save runs.
        name (str): Name of this training run.
    """

    # 获取 yolov5 路径，并切换过去
    current_dir = os.path.abspath(os.getcwd())
    yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../yolov5"))
    os.chdir(yolo_path)

    # 拼接正确的 table.yaml 路径（相对于 yolov5）
    data_yaml = os.path.abspath(os.path.join(yolo_path, "../data/yolo_table_dataset/table.yaml"))

    # ⚠️ 检查数据文件是否存在
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"[ERROR] Dataset YAML 不存在：{data_yaml}")

    # 组装训练命令
    command = [
        "python", "train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", data_yaml,
        "--weights", weights,
        "--project", project,
        "--name", name,
        "--exist-ok"
    ]

    print(f"[INFO] Training command:\n{' '.join(command)}")
    subprocess.run(command, check=True)

    print("\n✅ [INFO] 训练完成！请查看 runs/train/{name}/weights/best.pt")
    os.chdir(current_dir)

if __name__ == "__main__":
    train_yolo_table_detector()
