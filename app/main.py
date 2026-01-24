# app/main.py
from fastapi import FastAPI, UploadFile, File
import shutil, os
from app.ocr import extract_content

app = FastAPI(title="OCR API")


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_content(temp_path, file.content_type)
    finally:
        os.remove(temp_path)

    return {
        "fileName": file.filename,
        "extractedText": extracted_text
    }
