# # backend/parsers.py
# import os
# import fitz  # pymupdf
# import pdfplumber
# from PIL import Image
# import pytesseract
# from docx import Document
# from io import BytesIO
# from loguru import logger

# TESSERACT_CMD = os.getenv("TESSERACT_CMD")
# if TESSERACT_CMD:
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# def parse_pdf_bytes(data: bytes) -> str:
#     """Extract text from PDF bytes using pdfplumber first, fallback to pytesseract on pages with no text."""
#     text_chunks = []
#     try:
#         with pdfplumber.open(BytesIO(data)) as pdf:
#             for page in pdf.pages:
#                 txt = page.extract_text() or ""
#                 if txt.strip():
#                     text_chunks.append(txt)
#                 else:
#                     # fallback: render page to image and OCR
#                     pil = page.to_image(resolution=150).original
#                     ocr = pytesseract.image_to_string(pil)
#                     text_chunks.append(ocr)
#     except Exception:
#         logger.exception("pdfplumber parsing failed, fallback to pymupdf+ocr")
#         doc = fitz.open(stream=data, filetype="pdf")
#         for page in doc:
#             txt = page.get_text()
#             if txt.strip():
#                 text_chunks.append(txt)
#             else:
#                 pix = page.get_pixmap(dpi=150)
#                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                 text_chunks.append(pytesseract.image_to_string(img))
#     return "\n".join(text_chunks)


# def parse_docx_bytes(data: bytes) -> str:
#     f = BytesIO(data)
#     doc = Document(f)
#     paragraphs = [p.text for p in doc.paragraphs]
#     return "\n".join(paragraphs)


# def parse_image_bytes(data: bytes) -> str:
#     img = Image.open(BytesIO(data))
#     return pytesseract.image_to_string(img)


# def parse_any_file(filename: str, data: bytes) -> str:
#     ext = filename.lower().split('.')[-1]
#     if ext == 'pdf':
#         return parse_pdf_bytes(data)
#     if ext in ('docx', 'doc'):
#         return parse_docx_bytes(data)
#     if ext in ('png', 'jpg', 'jpeg', 'tiff'):
#         return parse_image_bytes(data)
#     # fallback: try pdf
#     return parse_pdf_bytes(data)



import os
from loguru import logger

def parse_document(file_path: str) -> str:
    """
    Parse document from various formats (PDF, DOCX, images).
    
    Args:
        file_path: Path to the document file
    
    Returns:
        Extracted text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            return _parse_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return _parse_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            return _parse_image(file_path)
        else:
            # Try as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.exception(f"Failed to parse file: {file_path}")
        raise


def _parse_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        logger.info(f"Extracted text from PDF: {len(text)} chars")
        return text
    except ImportError:
        logger.error("pdfplumber not installed")
        raise
    except Exception as e:
        logger.exception("PDF parsing failed")
        raise


def _parse_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        logger.info(f"Extracted text from DOCX: {len(text)} chars")
        return text
    except ImportError:
        logger.error("python-docx not installed")
        raise
    except Exception as e:
        logger.exception("DOCX parsing failed")
        raise


def _parse_image(file_path: str) -> str:
    """Extract text from images using OCR"""
    try:
        import pytesseract
        from PIL import Image
        
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        logger.info(f"Extracted text from image: {len(text)} chars")
        return text
    except ImportError:
        logger.error("pytesseract or Pillow not installed")
        raise
    except Exception as e:
        logger.exception("Image OCR failed")
        raise