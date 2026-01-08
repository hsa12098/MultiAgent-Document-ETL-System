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