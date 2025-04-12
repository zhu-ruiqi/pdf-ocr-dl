from api.utils import path_manager as paths
import os
import shutil
import subprocess

def run_yolo_predict(
    image_path,  # 接收上传图片的路径
    conf_threshold=0.3
):
    """
    Run YOLOv5 detection on a single image and save predictions to the output directory.
    """

    # 项目根路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # yolov5 所在路径（平行于 api 目录）
    yolo_path = os.path.join(project_root, "yolov5")

    # 模型权重路径
    weights_abs = os.path.join(yolo_path, "runs", "train", "table-detector", "weights", "best.pt")

    # 输入图片路径（上传文件传入）
    image_abs = os.path.abspath(image_path)

    # 输出路径
    output_abs = os.path.join(project_root, "outputs", "yolo_preds")

    # 检查模型和图片路径
    if not os.path.exists(weights_abs):
        raise FileNotFoundError(f"[ERROR] 模型文件不存在: {weights_abs}")
    if not os.path.exists(image_abs):
        raise FileNotFoundError(f"[ERROR] 图片文件不存在: {image_abs}")

    # 切换到 yolov5 目录运行命令
    current_dir = os.getcwd()
    os.chdir(yolo_path)

    # 清除之前的检测结果
    detect_dir = os.path.join("runs", "detect")
    if os.path.exists(detect_dir):
        shutil.rmtree(detect_dir)

    # 构建并运行 YOLO 检测命令
    command = [
        "python", "detect.py",
        "--weights", weights_abs,
        "--img", "640",
        "--conf", str(conf_threshold),
        "--source", image_abs
    ]

    print(f"[INFO] Running YOLOv5 detect:\n{' '.join(command)}")
    subprocess.run(command, check=True)

    # 拷贝预测结果到目标目录
    result_dir = os.path.join("runs", "detect", "exp")
    if not os.path.exists(result_dir):
        raise FileNotFoundError("[ERROR] YOLO 没有输出预测结果。")

    os.makedirs(output_abs, exist_ok=True)
    for file in os.listdir(result_dir):
        if file.endswith((".jpg", ".png")):
            shutil.copy(os.path.join(result_dir, file), os.path.join(output_abs, file))

    print(f"[✅ DONE] 检测结果已保存到: {output_abs}")
    os.chdir(current_dir)
