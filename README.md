# Multi-Agent Document Intelligence ETL System

This repo is a scaffold for running a multi-agent ETL pipeline that ingests documents, classifies them, retrieves schema templates from Pinecone, performs RAG-driven extraction using Groq / Llama, validates and stores structured JSON in Supabase Postgres, and uploads original files to Supabase Storage.

## Option 1: Docker (Production)

1. Copy `.env.example` below to `.env` and fill in your API keys.

2. Or run backend + Streamlit UI together via Docker Compose:

```bash
# from repository root
docker compose up --build
```

Services:
- Backend API: http://localhost:8000
- Streamlit UI: http://localhost:8501 (talks to backend via `API_URL=http://backend:8000` inside the Compose network)

## Option 2: Local Development (Recommended for Inferencing)

1. Copy `.env.example` to `.env` and fill in all required API keys (see Environment Variables section below).

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the backend API server (Terminal 1):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at `http://localhost:8000`.

5. Run the Streamlit frontend (Terminal 2):

```bash
streamlit run frontend\app.py
```

The frontend will be available at `http://localhost:8501`.

## API Endpoints

The backend exposes the following FastAPI endpoints:

- **GET** `/health` - Health check endpoint
- **POST** `/process` - Main inference endpoint to process documents through the ETL pipeline
- **POST** `/upload-schema` - Upload document schemas to Pinecone for RAG retrieval
- **GET** `/stats` - Get pipeline configuration and statistics

## ETL Pipeline Workflow

The document processing pipeline follows this sequence:

1. **Parsing Agent** → Extracts text from PDF/DOCX/images
2. **Classification Agent** → Identifies document type (resume, invoice, contract, etc.)
3. **RAG Schema Agent** → Retrieves extraction schema from Pinecone vector store
4. **Extraction Agent** → LLM extracts structured data based on schema
5. **Validation Agent** → Normalizes and validates extracted data
6. **Persistence Agent** → Uploads file to Supabase Storage and saves record to database
7. **Response Agent** → Formats final JSON response

## Database

Create a `documents` table in your Supabase Postgres with the following schema:

```sql
CREATE TABLE public.documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_type text,
  raw_text text,
  structured_data jsonb,
  file_url text,
  created_at timestamptz DEFAULT now()
);
```

## Environment Variables

Required environment variables (set in `.env`):

```bash
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET=documents

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1
PINECONE_INDEX=document-schemas

# Groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# API Server
APP_HOST=0.0.0.0
APP_PORT=8000
ENV=development
```
