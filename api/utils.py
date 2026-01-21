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

    - invoice
    - contract
    - receipt
    - form
    - report
    - transcript
    - cover letter
    - resume
    - other.
    
    Document text (first 500 chars):
    {raw_text[:500]}
    
    Be careful to choose accurately the category that best fits the text in the document. Check for keywords and context in the text.
    Return ONLY the category name, nothing else."""
    
    return prompt


def make_prompt_for_extraction(schema: dict, raw_text: str, doc_type: str,feedback: str | None = None) -> str:
    """
    Prompt for extraction agent.
    Supports retry with validation feedback.
    """
    schema_str = ""
    fields = schema.get("metadata", {}).get("schema", {}).get("fields", {})

    if fields:
        schema_str = json.dumps(fields, indent=2)
    else:
        schema_str = "Extract all relevant structured information."

    feedback_str = ""
    if feedback:
        feedback_str = (
            "\n\nPrevious extraction had issues:\n"
            + "\n".join([f"- {f}" for f in feedback])
            + "\nPlease correct these issues."
        )

    prompt = f"""
You are an information extraction agent.

Your task:
- Extract the structured data accurately from the document
- Output valid JSON only
- If the text in the Document is irrelevant with respect to the document type, then look for the correct document type from the text in the document for information.
- Add + with phone number if not present.
- If a field contains only one date, consider that as End Date.(Education: Start Date: Unspecified , End Date: 2025). 
- Convert the fields to standardtized formats (e.g., dates to YYYY-MM-DD, amounts to numeric values) if possible.

Document Type: {doc_type}

Schema (guidelines, not strict requirements):
{schema_str}

Document:
{raw_text}

Feedback:
{feedback_str}


CRITICAL rules:
- Your response MUST be a valid JSON object
- Do NOT wrap the JSON in markdown
- Do NOT include any text before or after the JSON
- Ensure all brackets are closed and commas are valid
- Do not invent fields that are not present in the document text.
- Do not include fields with None or empty fields in the output.
"""

    return prompt

def make_prompt_for_validation(schema: dict, extracted: dict, raw_text: str, doc_type: str) -> str:
    """
    Prompt for validation agent.
    The validator decides whether re-extraction is needed.
    """
    schema_str = ""
    fields = schema.get("metadata", {}).get("schema", {}).get("fields", {})

    if fields:
        schema_str = json.dumps(fields, indent=2)
    else:
        schema_str = "Extract all relevant structured information."
    prompt = f"""
You are a Validation Agent validating structured data extracted from a document.

Document type: {doc_type}
Schema (reference only; not all fields are required):
{json.dumps(schema_str, indent=2)}
Extracted Data:
{json.dumps(extracted, indent=2)}

Your task:

Validate issues such as:
- Invalid formats (e.g., malformed emails, impossible dates, non-numeric amounts)
- Do not assume all schema fields must be present in the extracted data.
- If no material issues exist, state that the extraction is valid.
- The dates may only contain Year, which is fine for that specific case.

Output format (JSON only):

If issues are found:
{{
  "is_valid": false,
  "feedback": [
    "Describe issue 1 clearly",
    "Describe issue 2 clearly"
  ]
}}

If extraction is acceptable:
{{
  "is_valid": true,
  "feedback": []
}}
Do not hallucinate errors.
Return ONLY the JSON object.

IMPORTANT:
- Do NOT infer missing fields solely because they appear in the schema.
- Strictly output json only.
"""
    return prompt



def safe_json_load(text: str):
    if not text:
        raise ValueError("Empty LLM output")

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))

    raise ValueError("No valid JSON object found in LLM output")

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

def extract_json(text: str) -> dict:
    # Remove markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    # Find ALL JSON-looking blocks
    candidates = re.findall(r"\{[\s\S]*?\}", text)

    if not candidates:
        raise ValueError("No JSON object found in model output")

    # Try parsing largest blocks first (most likely complete)
    candidates.sort(key=len, reverse=True)

    for block in candidates:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    raise ValueError("All extracted JSON blocks failed to parse")