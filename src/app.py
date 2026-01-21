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
        
        return JSONResponse({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "data": None,
            "logs": logs
        }, status_code=500)
    
    finally:
        # Cleanup temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")



@app.post("/upload-schema")
async def upload_schema(
    doc_type: str = Form(...),
    schema_json: str = Form(...)
):    
    """
    Upload and store a document schema in Pinecone for RAG retrieval.
    
    Args:
        doc_type: Document type (e.g., 'invoice', 'resume')
        schema_json: JSON string containing schema definition
    
    Returns:
        Success response with schema ID
    """
    try:
        import json
        from backend.pinecone_utils import pinecone_client

        logger.info(f"Uploading schema for doc_type: {doc_type}")

        schema = json.loads(schema_json)
        schema_id = f"{doc_type}_schema_{abs(hash(schema_json)) % 100000}"

        metadata = {
            "type": doc_type,
            "schema": schema,
            "description": f"Schema definition for {doc_type} documents"
        }

        text_repr = f"{doc_type} schema: {json.dumps(schema)}"

        pinecone_client.upsert_schema(
            id=schema_id,
            text=text_repr,
            metadata=metadata,
            namespace="document-schemas"
        )

        logger.info(f"Schema uploaded successfully: {schema_id}")

        return JSONResponse({
            "success": True,
            "schema_id": schema_id,
            "doc_type": doc_type,
            "message": f"Schema for '{doc_type}' uploaded successfully"
        })

    except json.JSONDecodeError:
        logger.error("Invalid JSON in schema_json")
        return JSONResponse({"success": False, "error": "Invalid JSON schema"}, status_code=400)

    except Exception as e:
        logger.exception("Failed to upload schema")
        return JSONResponse({"success": False, "error": f"Schema upload failed: {str(e)}"}, status_code=500)


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics and configuration"""
    return JSONResponse({
        "service": "Document ETL Pipeline",
        "version": "1.0.0",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "384")),
        "pinecone_index": os.getenv("PINECONE_INDEX", "document-schemas"),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "temp_upload_dir": str(TEMP_DIR)
    })


if __name__ == "__main__":
    logger.info("Starting Document ETL Pipeline API")
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development"
    )
