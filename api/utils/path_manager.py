import os

# 自动定位项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
TEMP_DIR = os.path.join(DATA_DIR, "temp")


# for path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
#     os.makedirs(path, exist_ok=True)


print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("UPLOAD_DIR:", UPLOAD_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("TEMP_DIR:", TEMP_DIR)