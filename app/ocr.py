import re
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
import mimetypes


# -------------------------
# OCR EXTRACTION (English-only for invoice text)
# -------------------------
TESSERACT_CONFIG = "-l eng --psm 6"


def extract_text_from_image(path: str) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image, lang="eng", config=TESSERACT_CONFIG).strip()


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
        text += pytesseract.image_to_string(page, lang="eng", config=TESSERACT_CONFIG) + "\n"
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

    vendor = extract_vendor(lines, normalized_text)
    if vendor:
        vendor = _clean_description(vendor) or None  # Latin-only, matches English invoice

    invoice = {
        "vendor": vendor,
        "invoice_number": extract_invoice_number(normalized_text),
        "invoice_date": extract_invoice_date(normalized_text),
        "currency": detect_currency(text),
        "subtotal": extract_amount(normalized_text, SUBTOTAL_KEYWORDS),
        "tax": extract_amount(normalized_text, TAX_KEYWORDS),
        "total": extract_amount(normalized_text, TOTAL_KEYWORDS),
        "line_items": extract_line_items(raw_lines, lines, normalized_text)
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
    """Skip only if line looks like a table header row (mostly header keywords)."""
    lower = line.lower().strip()
    words = re.split(r"[\s\|\t]+", lower)
    if not words:
        return True
    # Count how many words are (or contain) header keywords
    header_like = sum(1 for w in words if any(h in w for h in LINE_ITEM_HEADERS))
    # Skip only if majority of words are header-like (e.g. "Description  Qty  Rate  Amount")
    return len(words) <= 6 and header_like >= 2


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


def _keep_latin_for_description(text: str) -> str:
    """
    Keep only Latin letters (including accented), digits, and common punctuation.
    Strips other scripts so English invoices don't get mixed with other-language OCR.
    """
    if not text:
        return ""
    result = []
    for c in text:
        if ord(c) <= 0x7F:
            result.append(c)  # ASCII
        elif 0x00C0 <= ord(c) <= 0x024F:
            result.append(c)  # Latin extended
        elif c in " \t\n":
            result.append(c)
        # else: skip other scripts (Cyrillic, Arabic, Devanagari, etc.)
    return "".join(result)


def _clean_description(raw: str) -> str:
    """Trim and normalize description; keep Latin/English only so descriptions match invoice."""
    if not raw:
        return ""
    s = _keep_latin_for_description(raw)
    s = re.sub(r"\s+", " ", s).strip(" \t|:\-–—")
    return s[:500] if s else ""


def _parse_columns_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a table row by splitting on 2+ spaces. Trailing numeric columns = qty, rate, amount.
    Keeps description accurate (first column(s)) instead of stripping numbers from whole line.
    """
    # Split by 2+ spaces or tabs to get columns
    parts = re.split(r"\s{2,}|\t+", line.strip())
    if len(parts) < 2:
        return None
    # Trailing parts that look like numbers (amount, rate, qty)
    numeric_parts: List[Optional[float]] = []
    for i in range(len(parts) - 1, -1, -1):
        val = _parse_amount_from_string(parts[i])
        if val is not None and val >= 0:
            numeric_parts.insert(0, val)
        else:
            break
    if not numeric_parts:
        return None
    # Description = all non-numeric columns joined
    desc_end = len(parts) - len(numeric_parts)
    if desc_end <= 0:
        return None
    description = _clean_description(" ".join(parts[:desc_end]))
    if not description or _is_section_header(description):
        return None
    amount = numeric_parts[-1]
    if len(numeric_parts) >= 3:
        qty = max(1, int(round(numeric_parts[0])))
        unit_price = numeric_parts[1]
    elif len(numeric_parts) == 2:
        qty = 1
        unit_price = numeric_parts[0]
    else:
        qty = 1
        unit_price = amount
    return {
        "description": description,
        "quantity": max(1, qty),
        "unit_price": round(unit_price, 2),
        "amount": round(amount, 2)
    }


def _parse_table_like_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a line that looks like: Description   Qty   Rate   Amount (numbers separated by spaces/tabs)."""
    # Try column-based first (preserves description from first column(s))
    col_parsed = _parse_columns_line(line)
    if col_parsed is not None:
        return col_parsed
    # Fallback: match 2–4 numbers and strip them from line for description
    numbers = re.findall(r"[\$₹€]?\s*[\d,]+(?:\.\d{2})?", line)
    if not numbers:
        return None
    amounts = [_parse_amount_from_string(n) for n in numbers]
    amounts = [a for a in amounts if a is not None]
    if not amounts:
        return None
    amount = amounts[-1]
    desc_line = re.sub(r"[\$₹€]?\s*[\d,]+(?:\.\d{2})?", " ", line)
    description = _clean_description(desc_line)
    if not description:
        description = "Item"
    if len(amounts) >= 3:
        qty = max(1, int(round(amounts[0])))
        unit_price = amounts[1]
    elif len(amounts) == 2:
        qty = 1
        unit_price = amounts[0]
    else:
        qty = 1
        unit_price = amount
    return {
        "description": description,
        "quantity": max(1, qty),
        "unit_price": round(unit_price, 2),
        "amount": round(amount, 2)
    }


def extract_line_items(
    raw_lines: List[str], lines: List[str], full_text: str
) -> List[Dict[str, Any]]:
    """
    raw_lines: original lines (spaces preserved) for column-based parsing.
    lines: normalized (single space) for header/amount checks.
    """
    items: List[Dict[str, Any]] = []
    seen: set = set()

    for i, line in enumerate(lines):
        raw = raw_lines[i] if i < len(raw_lines) else line
        if _is_likely_header_line(line):
            continue
        # 1) Column-based: split by 2+ spaces so description = first column(s)
        parsed_col = _parse_columns_line(raw)
        if parsed_col and parsed_col["description"]:
            key = (
                parsed_col["description"][:80],
                parsed_col["quantity"],
                parsed_col["unit_price"],
                parsed_col["amount"],
            )
            if key not in seen:
                seen.add(key)
                items.append(parsed_col)
            continue

        # 2) Line ending with decimal amount (e.g. "Service Fee    200.00")
        amount_at_end = re.search(r"([\$₹€]?\s*[\d,]+\.\d{2})\s*$", raw)
        if amount_at_end:
            amount = _parse_amount_from_string(amount_at_end.group(1))
            if amount is not None:
                desc = re.sub(r"[\$₹€]?\s*[\d,]+\.\d{2}\s*$", "", raw).strip(" \t:-|")
                desc = _clean_description(desc)
                if desc and not _is_section_header(desc):
                    qty_match = re.search(
                        r"(\d+)\s*(?:hours|hrs|qty|x|\*)?", raw, re.IGNORECASE
                    )
                    qty = int(qty_match.group(1)) if qty_match else 1
                    unit_price = round(amount / qty, 2) if qty else amount
                    key = (desc[:80], qty, unit_price, amount)
                    if key not in seen:
                        seen.add(key)
                        items.append({
                            "description": desc,
                            "quantity": qty,
                            "unit_price": unit_price,
                            "amount": round(amount, 2),
                        })
            continue

        # 3) Integer amount at end
        amount_int = re.search(r"([\$₹€]?\s*[\d,]+)\s*$", raw)
        if amount_int:
            amount = _parse_amount_from_string(amount_int.group(1))
            if amount is not None and amount > 0:
                desc = re.sub(r"[\$₹€]?\s*[\d,]+\s*$", "", raw).strip(" \t:-|")
                desc = _clean_description(desc)
                if len(desc) >= 2 and not _is_section_header(desc):
                    qty = 1
                    unit_price = amount
                    key = (desc[:80], 1, amount, amount)
                    if key not in seen:
                        seen.add(key)
                        items.append({
                            "description": desc,
                            "quantity": qty,
                            "unit_price": round(unit_price, 2),
                            "amount": round(amount, 2),
                        })
            continue

        # 4) Table-like fallback: multiple numbers in one line
        parsed = _parse_table_like_line(raw)
        if parsed and parsed["description"] and not _is_section_header(parsed["description"]):
            key = (
                parsed["description"][:80],
                parsed["quantity"],
                parsed["unit_price"],
                parsed["amount"],
            )
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
