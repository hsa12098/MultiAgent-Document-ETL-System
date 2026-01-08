import os
import json
from loguru import logger
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "documents")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_file_and_get_url(local_file_path: str, remote_file_name: str) -> str:
    """
    Upload file to Supabase storage and return public URL.
    
    Args:
        local_file_path: Path to local file
        remote_file_name: Name for file in storage
    
    Returns:
        Public URL of uploaded file
    """
    try:
        with open(local_file_path, 'rb') as f:
            file_data = f.read()
        
        # Upload to Supabase
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(
            remote_file_name,
            file_data
        )
        
        # Get public URL
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(remote_file_name)
        logger.info(f"File uploaded to Supabase: {remote_file_name}")
        return public_url
    
    except Exception as e:
        logger.exception("Failed to upload file to Supabase")
        raise


def insert_document_record(doc_type: str, raw_text: str, structured: dict, file_url: str, pinecone_id: str, metadata: dict) -> str:
    """
    Insert document record into Supabase database.
    
    Args:
        doc_type: Document type (from classification)
        raw_text: Raw extracted text
        structured: Structured extracted data
        file_url: URL of uploaded file
        pinecone_id: Pinecone ID of the document
        metadata: Metadata of the document
    Returns:
        Record ID
    """
    try:
        record = {
            "doc_type": doc_type,
            "raw_text": raw_text,
            "structured_data": structured,
            "file_url": file_url,
            "pinecone_id": pinecone_id,
            "metadata": metadata
        }
        
        response = supabase.table("documents").insert(record).execute()
        record_id = response.data[0]["id"] if response.data else None
        
        logger.info(f"Document record inserted with ID: {record_id}")
        return record_id
    
    except Exception as e:
        logger.exception("Failed to insert document record")
        raise