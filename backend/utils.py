
# # backend/utils.py
# from loguru import logger
# import re


# def clean_text(text: str) -> str:
#     text = text.replace('\r\n', '\n')
#     # collapse multiple newlines
#     text = re.sub(r"\n{2,}", "\n\n", text)
#     # strip trailing whitespace
#     return text.strip()


# def safe_filename(filename: str) -> str:
#     # simple sanitizer
#     return re.sub(r"[^a-zA-Z0-9._-]", "_", filename)


# def make_prompt_for_classification(raw_text: str) -> str:
#     return f"Given the text below, classify the document into one of: Resume, Invoice, Contract, Bank Statement, Generic Document. Provide a single label.\n---\n{raw_text[:8000]}"


# def make_prompt_for_extraction(schema_metadata: dict, raw_text: str) -> str:
#     """
#     Compose RAG prompt that contains schema examples and asks model to extract JSON fields.
#     `schema_metadata` should contain 'schema' and 'examples' and 'template' keys.
#     """
#     schema = schema_metadata.get('schema', {})
#     examples = schema_metadata.get('examples', [])
#     template = schema_metadata.get('template', '')
#     s = """
# You are an extraction assistant. Extract fields following the schema and template.
# Return only valid JSON. If field is missing use null.

# Schema:
# """
#     s += str(schema) + "\n\n"
#     if examples:
#         s += "Examples:\n"
#         for ex in examples[:3]:
#             s += str(ex) + "\n\n"
#     if template:
#         s += "Template:\n" + template + "\n\n"
#     s += "Document text:\n" + raw_text[:14000]
#     s += "\n\nProduce a single JSON object following the schema exactly."
#     return s








from loguru import logger
import re
import json

def make_prompt_for_classification(raw_text: str) -> str:
    """
    Generate prompt for document classification.
    
    Args:
        raw_text: Extracted document text
    
    Returns:
        Prompt string for Groq
    """
    prompt = f"""Classify the following document into ONE of these categories:
    - resume
    - invoice
    - contract
    - receipt
    - form
    - report
    - other
    
    Document text (first 500 chars):
    {raw_text[:500]}
    
    Return ONLY the category name, nothing else."""
    
    return prompt


def make_prompt_for_extraction(schema: dict, raw_text: str) -> str:
    """
    Generate prompt for structured data extraction.
    
    Args:
        schema: Target schema with field definitions
        raw_text: Document text to extract from
    
    Returns:
        Prompt string for Groq
    """
    schema_str = ""
    fields = schema.get("metadata", {}).get("schema", {}).get("fields")
    if fields:
        schema_str = "\n".join([f"- {k}: {v}" for k, v in fields.items()])
    else:
        schema_str = "Extract all relevant information from the document."
    
    prompt = f"""Extract structured data from the document below following this schema:

Fields to extract:
{schema_str}

Document:
{raw_text}

Return the extracted data as a valid JSON object. Only include fields that are found in the document.
If a field is not present, omit it from the JSON."""
    
    return prompt


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\-\(\):]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text