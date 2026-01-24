# app/ocr.py
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_path


def extract_text_from_image(path: str) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image).strip()


def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.strip()


def extract_text_from_scanned_pdf(path: str) -> str:
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text.strip()


def extract_content(path: str, content_type: str) -> str:
    if content_type.startswith("image"):
        return extract_text_from_image(path)
    if content_type == "application/pdf":
        text = extract_text_from_pdf(path)
        if not text:
            text = extract_text_from_scanned_pdf(path)
        return text
    raise ValueError("Unsupported file type")
