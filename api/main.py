# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse

# import shutil
# import os

# from utils.pdf_text import extract_text_from_pdf

# app = FastAPI()

# # UPLOAD_DIR = "api/uploads"
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# UPLOAD_DIR = os.path.join(project_root, "api", "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/upload-pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     # Save uploaded PDF
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Call your text extraction function
#     text = extract_text_from_pdf(file_path)

#     return JSONResponse({
#         "filename": file.filename,
#         "text": text[:1000]  # 只返回前1000字符预览
#     })


# python env pdf-ocr-dl
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, shutil

from api.utils.pdf_text import extract_text_from_pdf
from api.utils.pdf_ocr_paddle import ocr_pdf_paddle
from api.utils.pdf_ocr_tesseract import ocr_pdf_tesseract
from api.utils.pdf_layout import analyze_pdf_layout
from api.utils.pdf_to_images import convert_pdf_to_images
from api.utils.pdf_table import extract_tables_from_pdf
from api.utils.yolo_detector import run_yolo_predict
from api.utils.image_extractor import extract_images_from_pdf
# from api.utils.ner_bert_infer import run_bert_ner
from api.utils.ner_bert_wrapper import run_bert_ner_in_subprocess
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from api.utils import path_manager as paths

# 分类器一般不上传 PDF，所以暂时不放进接口

app = FastAPI()
UPLOAD_DIR = "api/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 工具函数：保存上传文件
def save_upload_file(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

@app.post("/text/extract")
async def api_extract_text(file: UploadFile = File(...)):
    path = save_upload_file(file)
    text = extract_text_from_pdf(path)
    return JSONResponse({"filename": file.filename, "text_preview": text[:1000]})


@app.post("/ocr/paddle")
async def api_ocr_paddle(file: UploadFile = File(...)):
    path = save_upload_file(file)
    text = ocr_pdf_paddle(path)
    return JSONResponse({"filename": file.filename, "ocr_preview": text[:1000]})


@app.post("/ocr/tesseract")
async def api_ocr_tesseract(file: UploadFile = File(...)):
    path = save_upload_file(file)
    text = ocr_pdf_tesseract(path)
    return JSONResponse({"filename": file.filename, "ocr_preview": text[:1000]})


@app.post("/layout/analyze")
async def api_layout_analysis(file: UploadFile = File(...)):
    path = save_upload_file(file)
    image_paths = convert_pdf_to_images(path)  
    results = []
    for img in image_paths:
        try:
            layout = analyze_pdf_layout(img)
            results.append({"image": img, "layout": layout})
        except Exception as e:
            results.append({"image": img, "error": str(e)})
    return results

@app.post("/table/extract")
async def api_extract_tables(file: UploadFile = File(...)):
    path = save_upload_file(file)
    extract_tables_from_pdf(path)
    return JSONResponse({"status": "table extraction done", "file": file.filename})


@app.post("/images/extract")
async def api_extract_images(file: UploadFile = File(...)):
    path = save_upload_file(file)
    extract_images_from_pdf(path)
    return JSONResponse({"status": "image extraction done", "file": file.filename})


@app.post("/table/detect-yolo")
async def detect_yolo(file: UploadFile = File(...)):
    upload_dir = "uploads/yolo_input"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_yolo_predict(image_path=file_path)

    return JSONResponse({"status": "YOLO completed", "file": file.filename})


# @app.post("/ner/bert")
# async def api_ner_bert(text: str):
#     entities = run_bert_ner(text)
#     return JSONResponse({"text": text, "entities": entities})

@app.post("/ner/bert")
async def bert_ner(file: UploadFile = File(...)):
    text = await file.read()
    text = text.decode("utf-8")

    result = run_bert_ner_in_subprocess(text)
    return JSONResponse(content={"entities": result})


@app.post("/ner/bert-text")
async def bert_ner_text(text: str = Form(...)):
    result = run_bert_ner_in_subprocess(text)
    return JSONResponse(content={"entities": result})
