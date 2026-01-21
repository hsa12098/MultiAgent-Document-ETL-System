"""
Multi-Agent Document Intelligence ETL System
Entry point for Vercel deployment
"""
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from src.agents import process_document

TEMP_DIR = Path("./temp_uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Multi-Agent Document Intelligence ETL System",
    description="Intelligent document processing pipeline using LLMs and vector embeddings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    import re
    # Remove path separators and special characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Document ETL Pipeline"
    }


@app.post("/process")
async def process_document_endpoint(file: UploadFile = File(...)):
    """
    Main endpoint: Upload and process document through ETL pipeline.
    
    Args:
        file: Document file (PDF, DOCX, image)
    
    Returns:
        JSON with extracted data, metadata, and processing logs
    """
    temp_file_path = None
    logs = []
    
    try:
        # Step 1: Save uploaded file temporarily
        filename = safe_filename(file.filename)
        temp_file_path = TEMP_DIR / filename
        
        logger.info(f"Receiving file: {filename}")
        logs.append(f"📁 Received file: {filename}")
        
        with open(temp_file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        logger.info(f"File saved to temp location: {temp_file_path}")
        
        # Step 2: Process through ETL pipeline
        logger.info("Starting ETL pipeline for: %s", filename)
        logs.append("🔄 Starting ETL pipeline...")

        resp = process_document(str(temp_file_path)) or {}
        # Ensure dict and attach logs/filename
        if not isinstance(resp, dict):
            logger.error(f"process_document returned unexpected type: {type(resp)}")
            resp = {}

        resp.setdefault("logs", [])
        resp["logs"].extend(logs)
        resp["filename"] = filename

        if resp.get("success") is True:
            logger.info("Document processed successfully")
            resp["logs"].append("✅ Document processed successfully")
        else:
            logger.warning(f"Document processing failed: {resp.get('error')}")
            resp["logs"].append(f"❌ Error: {resp.get('error')}")

        return JSONResponse(resp)

    
    except Exception as e:
        logger.exception("Unexpected error in process_document_endpoint")
        logs.append(f"❌ Unexpected error: {str(e)}")
        return JSONResponse(
            {
                "success": False,
                "error": str(e),
                "logs": logs,
                "filename": filename if 'filename' in locals() else None
            },
            status_code=500
        )
    
    finally:
        # Clean up temporary files
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") != "production"
    )
