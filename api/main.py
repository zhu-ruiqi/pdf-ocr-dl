from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from utils.pdf_text import extract_text_from_pdf

app = FastAPI()

# UPLOAD_DIR = "api/uploads"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(project_root, "api", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call your text extraction function
    text = extract_text_from_pdf(file_path)

    return JSONResponse({
        "filename": file.filename,
        "text": text[:1000]  # 只返回前1000字符预览
    })