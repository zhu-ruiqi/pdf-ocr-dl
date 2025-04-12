import os

# 自动定位项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 核心路径（只到 uploads、outputs、temp）
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# 自动创建主要目录
for path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(path, exist_ok=True)

