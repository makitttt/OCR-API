from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware

from app.ocr import extract_invoice_as_json

# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="OCR Invoice API",
    description="Extract structured invoice data from images or PDFs",
    version="1.0.0"
)

# -------------------------
# CORS SETUP
# -------------------------
origins = [
    "http://localhost:5888",   # your Flutter web dev server
    "http://127.0.0.1:5888",
    "*"  # allow all origins (for testing only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# OCR UPLOAD ENDPOINT
# -------------------------
@app.post("/ocr/invoice")
async def extract_invoice(file: UploadFile = File(...)):
    """
    Upload an invoice (image or PDF) and receive structured invoice JSON
    """
    file_id = str(uuid.uuid4())
    temp_path = f"/tmp/{file_id}_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        invoice_data = extract_invoice_as_json(temp_path, file.content_type)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "fileName": file.filename,
        "invoice": invoice_data
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
