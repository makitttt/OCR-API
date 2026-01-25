import re
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
import mimetypes


# -------------------------
# OCR EXTRACTION
# -------------------------

def extract_text_from_image(path: str) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image, lang="eng").strip()


def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_scanned_pdf(path: str) -> str:
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="eng") + "\n"
    return text.strip()


def extract_content(path: str, content_type: str) -> str:
    # fallback to mime detection if content_type is unknown
    if not content_type or content_type == "application/octet-stream":
        content_type, _ = mimetypes.guess_type(path)

    if content_type and content_type.startswith("image"):
        return extract_text_from_image(path)

    if content_type == "application/pdf":
        text = extract_text_from_pdf(path)
        if not text:
            text = extract_text_from_scanned_pdf(path)
        return text

    raise ValueError(f"Unsupported file type: {content_type}")


# -------------------------
# INVOICE PARSER
# -------------------------

def parse_invoice(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    invoice = {
        "vendor": extract_vendor(lines),
        "invoice_number": extract_invoice_number(text),
        "invoice_date": extract_invoice_date(text),
        "currency": detect_currency(text),
        "subtotal": extract_amount(text, ["subtotal", "sub total"]),
        "tax": extract_amount(text, ["tax", "gst", "vat"]),
        "total": extract_amount(text, ["total", "amount due", "grand total"]),
        "line_items": extract_line_items(lines)
    }

    return invoice


# -------------------------
# FIELD EXTRACTORS
# -------------------------

def extract_vendor(lines: List[str]) -> Optional[str]:
    for line in lines[:5]:
        if "invoice" not in line.lower() and len(line) > 3:
            return line
    return None


def extract_invoice_number(text: str) -> Optional[str]:
    match = re.search(
        r"(invoice\s*(no|number)?[:\-]?\s*)([A-Z0-9\-]+)",
        text,
        re.IGNORECASE
    )
    return match.group(3) if match else None


def extract_invoice_date(text: str) -> Optional[str]:
    match = re.search(
        r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})",
        text
    )
    return match.group(1) if match else None


def detect_currency(text: str) -> Optional[str]:
    if "₹" in text or "INR" in text:
        return "INR"
    if "$" in text or "USD" in text:
        return "USD"
    if "€" in text:
        return "EUR"
    return None


def extract_amount(text: str, keywords: List[str]) -> Optional[float]:
    for key in keywords:
        pattern = rf"{key}[^0-9]*([\$₹€]?\s?[\d,]+\.\d{{2}})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            return float(re.sub(r"[^\d.]", "", value))
    return None


# -------------------------
# LINE ITEMS (REAL INVOICE FRIENDLY)
# -------------------------

def extract_line_items(lines: List[str]) -> List[Dict[str, Any]]:
    items = []

    for line in lines:
        # Example matches:
        # "Service Fees 200.00"
        # "Labor: 5 hours at $75 375.00"

        amount_match = re.search(r"([\$₹€]?\s?[\d,]+\.\d{2})$", line)
        if not amount_match:
            continue

        amount = float(re.sub(r"[^\d.]", "", amount_match.group(1)))

        qty_match = re.search(r"(\d+)\s*(hours|hrs|qty|x)?", line, re.IGNORECASE)
        qty = int(qty_match.group(1)) if qty_match else 1

        description = re.sub(r"([\$₹€]?\s?[\d,]+\.\d{2})", "", line).strip(" :-")

        unit_price = round(amount / qty, 2) if qty else amount

        items.append({
            "description": description,
            "quantity": qty,
            "unit_price": unit_price,
            "amount": amount
        })

    return items


# -------------------------
# PUBLIC API METHOD
# -------------------------

def extract_invoice_as_json(path: str, content_type: str) -> Dict[str, Any]:
    text = extract_content(path, content_type)
    return parse_invoice(text)
