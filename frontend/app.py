import streamlit as st
import requests
import json
import os
from datetime import datetime

API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Page configuration
st.set_page_config(
    page_title='Doc ETL - Multi-Agent',
    page_icon='📄',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('📄 Multi-Agent Document Intelligence ETL')
st.markdown('---')

# Sidebar configuration
with st.sidebar:
    st.header('⚙️ Configuration')
    
    # Check API health
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Unavailable")
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to {API_URL}")
    
    st.divider()
    
    # Get system stats
    try:
        stats_resp = requests.get(f"{API_URL}/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            with st.expander("📊 System Info"):
                st.write(f"**Embedding Model:** {stats.get('embedding_model')}")
                st.write(f"**Embedding Dimension:** {stats.get('embedding_dimension')}")
                st.write(f"**LLM Model:** {stats.get('groq_model')}")
                st.write(f"**Pinecone Index:** {stats.get('pinecone_index')}")
    except Exception as e:
        st.warning(f"Could not fetch system stats: {str(e)}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📤 Process Document", "📚 Upload Schema", "📋 About"])

# ============ TAB 1: Process Document ============
with tab1:
    st.header("Process Document")
    st.markdown("Upload a document (PDF, DOCX, or image) to extract structured data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            'Choose a document',
            type=['pdf', 'docx', 'doc', 'jpg', 'jpeg', 'png', 'tiff'],
            help='Supported formats: PDF, DOCX, JPG, PNG, TIFF'
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        process_button = st.button('🚀 Start Processing', use_container_width=True)
    
    if process_button:
        if uploaded_file is None:
            st.error("❌ Please upload a file first")
        else:
            with st.spinner(f'📥 Processing {uploaded_file.name}...'):
                try:
                    # Prepare request
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Call /process endpoint
                    response = requests.post(
                        f"{API_URL}/process",
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('success'):
                            st.success('✅ Processing Complete!')
                            
                            # Display results in organized sections
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Document Type", result['data'].get('doc_type', 'N/A'))
                            
                            with col2:
                                st.metric("Record ID", result['data'].get('record_id', 'N/A')[:12] + "...")
                            
                            with col3:
                                st.metric("Status", "✅ Stored" if result['data'].get('record_id') else "⚠️ Not stored")
                            
                            st.divider()
                            
                            # Extracted Data
                            st.subheader("📊 Extracted Structured Data")
                            extracted = result['data'].get('extracted', {})
                            if extracted:
                                st.json(extracted)
                                
                                # Download extracted data
                                json_str = json.dumps(extracted, indent=2)
                                st.download_button(
                                    label="⬇️ Download JSON",
                                    data=json_str,
                                    file_name=f"{result['data'].get('doc_type', 'document')}_extracted.json",
                                    mime="application/json"
                                )
                            else:
                                st.info("No data extracted")
                            
                            st.divider()
                            
                            # File URL
                            if result['data'].get('file_url'):
                                st.subheader("📁 File Storage")
                                st.markdown(f"[📥 View Uploaded File]({result['data'].get('file_url')})")
                            
                            # Processing logs
                            with st.expander("📋 Processing Logs"):
                                for log in result.get('logs', []):
                                    st.text(log)
                        
                        else:
                            st.error(f"❌ Processing Failed: {result.get('error')}")
                            with st.expander("📋 Error Details"):
                                for log in result.get('logs', []):
                                    st.text(log)
                    
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.text(response.text)
                
                except requests.exceptions.Timeout:
                    st.error("❌ Request timeout. The document may be too large or processing took too long.")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")


# ============ TAB 2: Upload Schema ============
with tab2:
    st.header("Upload Document Schema")
    st.markdown("Define and upload extraction schemas for specific document types")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        doc_type = st.text_input(
            "Document Type",
            placeholder="e.g., invoice, resume, contract",
            help="The type/category of documents this schema applies to"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
    
    # Schema JSON editor
    st.subheader("Schema Definition")
    schema_template = {
        "fields": {
            "field_name_1": "Field description",
            "field_name_2": "Field description"
        },
        "description": "Schema description",
        "examples": []
    }
    
    schema_json = st.text_area(
        "Schema JSON",
        value=json.dumps(schema_template, indent=2),
        height=300,
        help="Define the fields and structure for this document type"
    )
    
    # Upload button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        upload_schema_btn = st.button("📤 Upload Schema", use_container_width=True)
    
    with col2:
        # Copy template button
        st.download_button(
            label="📋 Copy Template",
            data=json.dumps(schema_template, indent=2),
            file_name="schema_template.json",
            mime="application/json",
            use_container_width=True
        )
    
    if upload_schema_btn:
        if not doc_type.strip():
            st.error("❌ Please enter a document type")
        else:
            try:
                # Validate JSON
                schema = json.loads(schema_json)
                
                with st.spinner("📤 Uploading schema..."):
                    response = requests.post(
                        f"{API_URL}/upload-schema",
                        data={
                            "doc_type": doc_type,
                            "schema_json": schema_json
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            st.success(f"✅ Schema uploaded successfully!")
                            st.info(f"Schema ID: `{result.get('schema_id')}`")
                            st.markdown(f"**Message:** {result.get('message')}")
                        else:
                            st.error(f"❌ Upload failed: {result.get('error')}")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.text(response.text)
            
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {str(e)}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {API_URL}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")


# ============ TAB 3: About ============
with tab3:
    st.header("About Document ETL Pipeline")
    
    st.markdown("""
    ### 🎯 Overview
    Multi-Agent Document Intelligence ETL System is an intelligent document processing pipeline that:
    
    1. **📄 Parses** documents in multiple formats (PDF, DOCX, images with OCR)
    2. **🏷️ Classifies** documents by type using LLM
    3. **🔍 Retrieves** relevant extraction schemas from vector database using embeddings
    4. **📊 Extracts** structured data from unstructured documents
    5. **✅ Validates** extracted information
    6. **💾 Persists** results to database
    
    ### 🏗️ Architecture
    
    **Backend (FastAPI + LangGraph):**
    - Multi-agent system using LangGraph state machine
    - 7 specialized agents (Parsing, Classification, RAG, Extraction, Validation, Persistence, Response)
    - LLM powered by Groq (llama-3.1-8b-instant)
    - Vector embeddings via Hugging Face Sentence Transformers
    - Vector database: Pinecone (serverless)
    - Data storage: Supabase (PostgreSQL + Object Storage)
    
    **Frontend (Streamlit):**
    - Intuitive document upload interface
    - Schema management interface
    - Real-time processing logs
    - Extracted data visualization and download
    
    ### 🚀 Features
    
    - **Multi-Format Support:** PDF, DOCX, JPG, PNG, TIFF
    - **OCR Capability:** Extract text from scanned documents
    - **Schema-Driven Extraction:** Customize extraction per document type
    - **Vector Embeddings:** Semantic search for relevant schemas
    - **Type Validation:** Automatic field type coercion
    - **File Storage:** Uploaded files stored in Supabase
    - **Database Persistence:** Structured data stored in PostgreSQL
    
    ### 🛠️ Technologies
    
    | Component | Technology |
    |-----------|-----------|
    | LLM | Groq (llama-3.1-8b-instant) |
    | Embeddings | Hugging Face Sentence Transformers |
    | Vector DB | Pinecone (serverless) |
    | Orchestration | LangGraph |
    | Backend | FastAPI |
    | Frontend | Streamlit |
    | Storage | Supabase (PostgreSQL + S3-like) |
    
    ### 📚 Example Workflows
    
    **Invoice Processing:**
    - Upload invoice PDF → Classify as "invoice" → Retrieve invoice schema from Pinecone
    - Extract: invoice_number, amount, vendor, date → Validate and store
    
    **Resume Screening:**
    - Upload resume PDF → Classify as "resume" → Retrieve resume schema
    - Extract: name, email, phone, skills, experience → Store in database
    
    ### 🔐 Security
    
    - API keys stored in environment variables
    - File uploads temporarily stored and cleaned up
    - CORS enabled for frontend access
    - Request validation on all endpoints
    
    ### 📖 Getting Started
    
    1. **Upload Document:** Use the "Process Document" tab
    2. **Define Schemas:** Use the "Upload Schema" tab to create schemas for your document types
    3. **Monitor Processing:** Check logs in real-time
    4. **Download Results:** Export extracted data as JSON
    
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Endpoint", API_URL)
    with col2:
        st.metric("Frontend", "Streamlit")
    with col3:
        st.metric("Version", "1.0.0")