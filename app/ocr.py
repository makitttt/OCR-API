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
# COMMON INVOICE TEMPLATE (labels/keywords used across many invoice formats)
# -------------------------

# Keywords to look for when extracting amounts (order can matter for disambiguation)
SUBTOTAL_KEYWORDS = [
    "subtotal", "sub total", "net amount", "amount before tax", "taxable amount",
    "total before tax", "goods total", "services total", "amount"
]
TAX_KEYWORDS = [
    "tax", "gst", "vat", "cgst", "sgst", "igst", "tax amount", "duty",
    "central tax", "state tax", "total tax"
]
TOTAL_KEYWORDS = [
    "grand total", "total amount", "amount due", "balance due", "amount payable",
    "total", "net total", "total payable", "final amount", "invoice total",
    "sum total", "total due"
]

# Invoice number label variants
INVOICE_NUMBER_PATTERNS = [
    r"invoice\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9\-/]+)",
    r"inv\.?\s*(?:no\.?|#)?\s*[:\-]?\s*([A-Z0-9\-/]+)",
    r"bill\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9\-/]+)",
    r"reference\s*[:\-]?\s*([A-Z0-9\-/]+)",
    r"ref\.?\s*[:\-]?\s*([A-Z0-9\-/]+)",
    r"invoice\s*id\s*[:\-]?\s*([A-Z0-9\-/]+)",
]

# Date format patterns (day/month/year and variants)
DATE_PATTERNS = [
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    r"(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})",
    r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})",
    r"date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
    r"invoice\s*date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
]

# Vendor / "From" section markers
VENDOR_MARKERS = ["from", "bill from", "seller", "supplier", "issued by", "vendor", "company"]

# Line item table header keywords (for detecting table structure)
LINE_ITEM_HEADERS = [
    "description", "item", "particulars", "details", "product", "service",
    "qty", "quantity", "rate", "unit price", "price", "amount", "total"
]


def _normalize_text(text: str) -> str:
    """Collapse whitespace and fix common OCR issues for matching."""
    if not text:
        return ""
    # Replace multiple spaces/tabs/newlines with single space
    text = re.sub(r"[\s\t]+", " ", text)
    return text.strip()


def _parse_amount_from_string(s: str) -> Optional[float]:
    """Extract numeric value from string like '$1,234.56' or '1234.56'."""
    if not s:
        return None
    cleaned = re.sub(r"[^\d.]", "", s)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# -------------------------
# INVOICE PARSER
# -------------------------

def parse_invoice(text: str) -> Dict[str, Any]:
    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
    normalized_text = _normalize_text(text)
    lines = [_normalize_text(l) for l in raw_lines]

    invoice = {
        "vendor": extract_vendor(lines, normalized_text),
        "invoice_number": extract_invoice_number(normalized_text),
        "invoice_date": extract_invoice_date(normalized_text),
        "currency": detect_currency(text),
        "subtotal": extract_amount(normalized_text, SUBTOTAL_KEYWORDS),
        "tax": extract_amount(normalized_text, TAX_KEYWORDS),
        "total": extract_amount(normalized_text, TOTAL_KEYWORDS),
        "line_items": extract_line_items(lines, normalized_text)
    }

    return invoice


# -------------------------
# FIELD EXTRACTORS
# -------------------------

def extract_vendor(lines: List[str], full_text: str) -> Optional[str]:
    """Try vendor markers first, then first substantial non-invoice line in top section."""
    lower_text = full_text.lower()
    # 1) Line after "From" / "Bill From" / "Seller" etc.
    for marker in VENDOR_MARKERS:
        idx = lower_text.find(marker)
        if idx == -1:
            continue
        # Take rest of line or next line as vendor name
        snippet = full_text[idx:idx + 120]
        after_marker = re.sub(r"^.*?" + re.escape(marker) + r"\s*[:\-]?\s*", "", snippet, 1, re.IGNORECASE)
        after_marker = _normalize_text(after_marker.split("\n")[0])
        if len(after_marker) > 2 and not re.match(r"^[\d\s\-\/\.]+$", after_marker):
            return after_marker[:200]

    # 2) First lines that look like company name (not invoice, not date, not number-only)
    for line in lines[:12]:
        line_lower = line.lower()
        if "invoice" in line_lower or "bill" in line_lower and "from" not in line_lower:
            continue
        if re.match(r"^[\d\s\-\/\.\,\$₹€]+$", line) or re.match(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$", line):
            continue
        if len(line) < 4:
            continue
        # Skip obvious headers
        if line_lower in ("date", "qty", "amount", "description", "total", "subtotal"):
            continue
        return line[:200]

    return None


def extract_invoice_number(text: str) -> Optional[str]:
    for pattern in INVOICE_NUMBER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = match.group(1).strip()
            if len(num) >= 1 and len(num) <= 50:
                return num
    return None


def extract_invoice_date(text: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def detect_currency(text: str) -> Optional[str]:
    if "₹" in text or "INR" in text or "Rs." in text or "Rs " in text:
        return "INR"
    if "$" in text or "USD" in text:
        return "USD"
    if "€" in text or "EUR" in text:
        return "EUR"
    if "£" in text or "GBP" in text:
        return "GBP"
    return "INR"  # default for many Indian invoices


def extract_amount(text: str, keywords: List[str]) -> Optional[float]:
    """Try each keyword; support both decimal (123.45) and integer amounts.
    Uses word boundary so e.g. 'total' does not match 'subtotal'.
    """
    for key in keywords:
        key_pattern = r"\b" + re.escape(key) + r"\b"
        pattern_dec = key_pattern + r"[^\d]*?[\$₹€\s]*([\d,]+\.\d{2})\s*"
        match = re.search(pattern_dec, text, re.IGNORECASE)
        if match:
            val = _parse_amount_from_string(match.group(1))
            if val is not None:
                return val
        pattern_int = key_pattern + r"[^\d]*?[\$₹€\s]*([\d,]+)\s*"
        match = re.search(pattern_int, text, re.IGNORECASE)
        if match:
            val = _parse_amount_from_string(match.group(1))
            if val is not None:
                return val
    return None


# -------------------------
# LINE ITEMS (table + line-based; common template)
# -------------------------

def _is_likely_header_line(line: str) -> bool:
    lower = line.lower()
    words = re.split(r"[\s\|\t]+", lower)
    return any(h in w for w in words for h in LINE_ITEM_HEADERS)


def _is_section_header(desc: str) -> bool:
    """Skip lines that are just subtotal/tax/total section labels."""
    lower = desc.lower().strip()
    if not lower or len(lower) < 2:
        return True
    headers = (
        "subtotal", "sub total", "tax", "gst", "vat", "total", "grand total",
        "amount due", "balance", "payment", "invoice total", "net amount"
    )
    return lower in headers or any(lower == h for h in headers)


def _parse_table_like_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a line that looks like: Description   Qty   Rate   Amount (numbers separated by spaces/tabs)."""
    # Match 2–4 numbers (qty, rate, amount or just amount)
    numbers = re.findall(r"[\$₹€]?\s*[\d,]+(?:\.\d{2})?", line)
    if not numbers:
        return None
    amounts = [_parse_amount_from_string(n) for n in numbers]
    amounts = [a for a in amounts if a is not None]
    if not amounts:
        return None
    # Last number is usually amount
    amount = amounts[-1]
    # Remove all amount-like parts from line to get description
    desc_line = re.sub(r"[\$₹€]?\s*[\d,]+(?:\.\d{2})?", " ", line)
    desc_line = re.sub(r"\s+", " ", desc_line).strip(" \t|:-")
    if len(amounts) >= 3:
        qty = max(1, int(round(amounts[0])))
        unit_price = amounts[1]
    elif len(amounts) == 2:
        qty = 1
        unit_price = amounts[0]
    else:
        qty = 1
        unit_price = amount
    description = desc_line if desc_line else "Item"
    return {
        "description": description[:500],
        "quantity": max(1, qty),
        "unit_price": round(unit_price, 2),
        "amount": round(amount, 2)
    }


def extract_line_items(lines: List[str], full_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()

    for line in lines:
        if _is_likely_header_line(line):
            continue
        # 1) Line ending with amount (e.g. "Service Fee    200.00" or "200.00")
        amount_at_end = re.search(r"([\$₹€]?\s*[\d,]+\.\d{2})\s*$", line)
        if amount_at_end:
            amount = _parse_amount_from_string(amount_at_end.group(1))
            if amount is not None:
                desc = re.sub(r"[\$₹€]?\s*[\d,]+\.\d{2}\s*$", "", line).strip(" \t:-|")
                if _is_section_header(desc):
                    continue
                qty_match = re.search(r"(\d+)\s*(?:hours|hrs|qty|x|\*)?", line, re.IGNORECASE)
                qty = int(qty_match.group(1)) if qty_match else 1
                unit_price = round(amount / qty, 2) if qty else amount
                key = (desc[:80], qty, unit_price, amount)
                if key not in seen:
                    seen.add(key)
                    items.append({
                        "description": desc[:500] if desc else "Item",
                        "quantity": qty,
                        "unit_price": unit_price,
                        "amount": round(amount, 2)
                    })
            continue

        # 2) Integer amount at end
        amount_int = re.search(r"([\$₹€]?\s*[\d,]+)\s*$", line)
        if amount_int:
            amount = _parse_amount_from_string(amount_int.group(1))
            if amount is not None and amount > 0:
                desc = re.sub(r"[\$₹€]?\s*[\d,]+\s*$", "", line).strip(" \t:-|")
                if len(desc) < 2 or _is_section_header(desc):
                    continue
                qty = 1
                unit_price = amount
                key = (desc[:80], 1, amount, amount)
                if key not in seen:
                    seen.add(key)
                    items.append({
                        "description": desc[:500],
                        "quantity": qty,
                        "unit_price": round(unit_price, 2),
                        "amount": round(amount, 2)
                    })
            continue

        # 3) Table-like: multiple numbers in one line (Description  Qty  Rate  Amount)
        parsed = _parse_table_like_line(line)
        if parsed and parsed["description"] and not _is_section_header(parsed["description"]):
            key = (parsed["description"][:80], parsed["quantity"], parsed["unit_price"], parsed["amount"])
            if key not in seen:
                seen.add(key)
                items.append(parsed)
    return items


# -------------------------
# PUBLIC API METHOD
# -------------------------

def extract_invoice_as_json(path: str, content_type: str) -> Dict[str, Any]:
    text = extract_content(path, content_type)
    return parse_invoice(text)
