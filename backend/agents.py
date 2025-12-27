import os
import json
import re
from loguru import logger
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any

from backend.groq_client import groq_client
from backend.pinecone_utils import pinecone_client
from backend.utils import make_prompt_for_classification, make_prompt_for_extraction, clean_text
from backend.parsers import parse_document
from backend.supabase_client import insert_document_record, upload_file_and_get_url


# ============ State Definition ============
class DocumentState(TypedDict, total=False):
    """Shared state across all agents in the graph"""
    file_path: str
    raw_text: str
    doc_type: str
    schema: dict
    extracted: dict
    validated: dict
    file_url: str
    record_id: str
    error: str
    response: dict  
    pinecone_id: str

# ============ Agent Nodes ============

def parsing_agent(state: DocumentState) -> DocumentState:
    """
    1️⃣ Parsing Agent: Convert raw files into clean text.
    Input: PDF/DOCX/image file path
    Output: Plain text
    """
    try:
        logger.info(f"Parsing document from {state['file_path']}")
        raw_text = parse_document(state["file_path"])
        state["raw_text"] = clean_text(raw_text)
        logger.info(f"Parsed text length: {len(state['raw_text'])}")
    except Exception as e:
        logger.exception("Parsing agent failed")
        state["error"] = f"Parsing failed: {str(e)}"
    
    return state


def classification_agent(state: DocumentState) -> DocumentState:
    """
    2️⃣ Classification Agent: Determine document type.
    Input: Parsed text
    Output: Document type label (resume, invoice, contract, etc.)
    """
    if state.get("error"):
        return state
    
    try:
        logger.info("Classifying document")
        prompt = make_prompt_for_classification(state["raw_text"])
        label = groq_client.generate(prompt, max_tokens=32, temperature=0.0).strip()
        label = label.split('\n')[0].strip().lower()
        state["doc_type"] = label
        logger.info(f"Document classified as: {label}")
    except Exception as e:
        logger.exception("Classification agent failed")
        state["error"] = f"Classification failed: {str(e)}"
    
    return state


def rag_schema_agent(state: DocumentState) -> DocumentState:
    """
    3️⃣ RAG Schema Retrieval Agent: Retrieve extraction schema from Pinecone using embeddings.
    Input: Document type + raw text
    Output: JSON schema with fields and descriptions
    """
    if state.get("error"):
        return state
    
    try:
        doc_type = state.get("doc_type", "unknown")
        raw_text = state.get("raw_text", "")
        
        logger.info(f"RAG Agent: Retrieving schema for doc_type={doc_type}")
        
        # Strategy 1: Query by document type metadata filter
        matches = pinecone_client.query_by_doc_type(doc_type, top_k=3)
        
        if matches:
            best_match = matches[0]
            state["pinecone_id"] = best_match.get("id")              
            state["schema"] = best_match.get("metadata", {})
            score = best_match.get("score", 0)
            logger.info(f"RAG Agent: Retrieved schema for {doc_type} (score: {score:.4f})")
        else:
            # Strategy 2: Fallback to semantic search using raw text
            logger.info("RAG Agent: No exact match for doc_type, attempting semantic search")
            
            search_text = raw_text[:1000] if raw_text else doc_type
            matches = pinecone_client.query_schema(search_text, top_k=3)
            
            if matches:
                best_match = matches[0]
                state["pinecone_id"] = best_match.get("id")
                state["schema"] = best_match.get("metadata", {})
                score = best_match.get("score", 0)
                logger.info(f"RAG Agent: Found schema via semantic search (score: {score:.4f})")
            else:
                # Strategy 3: Use fallback schema
                logger.warning(f"RAG Agent: No schema found for {doc_type}, using fallback")
                state["pinecone_id"] = None
                state["schema"] = {
                    "fields": {},
                    "description": f"Auto-generated schema for {doc_type}",
                    "type": doc_type
                }
    
    except Exception as e:
        logger.exception("RAG schema agent failed")
        state["error"] = f"Schema retrieval failed: {str(e)}"
        state["schema"] = {"fields": {}, "description": "Error retrieving schema"}
    
    return state


def extraction_agent(state: DocumentState) -> DocumentState:
    """
    4️⃣ Extraction Agent: Convert unstructured text to structured JSON.
    Input: Document text + schema
    Output: Raw extracted JSON
    """
    if state.get("error"):
        return state
    
    try:
        logger.info("Extracting structured data")
        inner_json_str = state["schema"].get("schema")
        state["schema"] = json.loads(inner_json_str)
        prompt = make_prompt_for_extraction(state["schema"], state["raw_text"])
        output = groq_client.generate(prompt, max_tokens=2048, temperature=0.0)
        
        parsed = None
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", output)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.error("Failed to parse extracted JSON block")
        
        if parsed:
            state["extracted"] = parsed
            logger.info(f"Extraction successful: {len(parsed)} fields extracted")
        else:
            logger.error("Failed to parse extraction output")
            state["error"] = "Extraction parsing failed"
            state["extracted"] = {}
    
    except Exception as e:
        logger.exception("Extraction agent failed")
        state["error"] = f"Extraction failed: {str(e)}"
        state["extracted"] = {}
    
    return state


def validation_agent(state: DocumentState) -> DocumentState:
    """
    5️⃣ Validation Agent: Ensure extracted data is usable.
    Input: Extracted JSON
    Output: Cleaned and normalized JSON
    """
    if state.get("error"):
        return state
    
    try:
        logger.info("Validating extracted data")
        validated = state["extracted"].copy() if state["extracted"] else {}
        
        if "email" in validated and validated["email"]:
            validated["email"] = str(validated["email"]).lower().strip()
        
        if "phone" in validated and validated["phone"]:
            validated["phone"] = re.sub(r"\D", "", str(validated["phone"]))
        
        if "amount" in validated and validated["amount"]:
            try:
                validated["amount"] = float(validated["amount"])
            except (ValueError, TypeError):
                logger.warning("Failed to convert amount to float")
        
        from datetime import datetime
        validated["_validated_at"] = datetime.now().isoformat()
        
        state["validated"] = validated
        logger.info(f"Validation successful: {len(validated)} fields validated")
    
    except Exception as e:
        logger.exception("Validation agent failed")
        state["error"] = f"Validation failed: {str(e)}"
        state["validated"] = state.get("extracted", {})
    
    return state


def persistence_agent(state: DocumentState) -> DocumentState:
    """
    6️⃣ Persistence Agent: Store results in Supabase.
    Input: Validated JSON + metadata
    Output: Stored record ID
    """
    if state.get("error"):
        return state
    
    try:
        logger.info("Persisting document to Supabase")
        remote_name = os.path.basename(state["file_path"])
        state["file_url"] = upload_file_and_get_url(state["file_path"], remote_name)
        record_id = insert_document_record(
            doc_type=state["doc_type"],
            raw_text=state["raw_text"],
            structured=state["validated"],
            file_url=state["file_url"],
            pinecone_id=state["pinecone_id"],
            metadata=state["schema"]
        )
        state["record_id"] = record_id
        logger.info(f"Document persisted with ID: {record_id}")
    
    except Exception as e:
        logger.exception("Persistence agent failed")
        state["error"] = f"Persistence failed: {str(e)}"
    
    return state


def response_agent(state: DocumentState) -> DocumentState:
    """
    7️⃣ Response Agent: Format final output for API response.
    Input: Validated + stored result
    Output: Clean API response
    """
    logger.info("Formatting response")
    
    if state.get("error"):
        state["response"] = {
            "success": False,
            "error": state["error"],
            "data": None
        }
    else:
        state["response"] = {
            "success": True,
            "error": None,
            "data": {
                "record_id": state.get("record_id"),
                "doc_type": state.get("doc_type"),
                "extracted": state.get("validated", {}),
                "file_url": state.get("file_url", "")
            }
        }
    
    logger.info(f"Response formatted: success={state['response'].get('success')}")
    return state


# ============ Build Graph ============

def build_document_etl_graph():
    """Construct the LangGraph state machine for document ETL"""
    
    graph = StateGraph(DocumentState)
    
    # Add all nodes
    graph.add_node("parsing", parsing_agent)
    graph.add_node("classification", classification_agent)
    graph.add_node("rag_schema", rag_schema_agent)
    graph.add_node("extraction", extraction_agent)
    graph.add_node("validation", validation_agent)
    graph.add_node("persistence", persistence_agent)
    graph.add_node("response", response_agent)
    
    # Define edges (sequential workflow)
    graph.add_edge(START, "parsing")
    graph.add_edge("parsing", "classification")
    graph.add_edge("classification", "rag_schema")
    graph.add_edge("rag_schema", "extraction")
    graph.add_edge("extraction", "validation")
    graph.add_edge("validation", "persistence")
    graph.add_edge("persistence", "response")
    graph.add_edge("response", END)
    
    return graph.compile()


# ============ Initialize Graph ============
document_etl_graph = build_document_etl_graph()


def process_document(file_path: str) -> dict:
    """
    Main entry point: process a document through the ETL pipeline.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        Response dict with success status and extracted data
    """
    try:
        initial_state: DocumentState = {
            "file_path": file_path,
            "raw_text": "",
            "doc_type": "",
            "schema": {},
            "extracted": {},
            "validated": {},
            "file_url": "",
            "record_id": None,
            "error": None,
            "response": {}
        }
        
        logger.info(f"Starting document processing for: {file_path}")
        final_state = document_etl_graph.invoke(initial_state)
        logger.info("Document processing complete")
        
        # Extract and return the response dict
        response = final_state.get("response")
        if not response or response is None:
            logger.error("response_agent did not set response field in state")
            return {
                "success": False,
                "error": "Response formatting failed",
                "data": None
            }
        
        return response
    
    except Exception as e:
        logger.exception(f"process_document failed with exception: {str(e)}")
        return {
            "success": False,
            "error": f"Pipeline execution failed: {str(e)}",
            "data": None
        }