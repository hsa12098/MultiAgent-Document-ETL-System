# Multi-Agent Document Intelligence ETL System

This repo is a scaffold for running a multi-agent ETL pipeline that ingests documents, classifies them, retrieves schema templates from Pinecone, performs RAG-driven extraction using Groq / Llama, validates and stores structured JSON in Supabase Postgres, and uploads original files to Supabase Storage.

## Quickstart

1. Copy `.env.example` to `.env` and fill keys.
2. Build and run backend container

```bash
# from repository root
cd backend
docker build -t doc-etl-backend .
docker run -e SUPABASE_KEY="$SUPABASE_KEY" -e SUPABASE_URL="$SUPABASE_URL" -p 8000:8000 doc-etl-backend
````

3. Run Streamlit frontend (locally)

```bash
pip install -r backend/requirements.txt
cd frontend
streamlit run app.py
```

## Database

Create a `documents` table in your Supabase Postgres with the following schema:

```sql
CREATE TABLE public.documents (
  id uuid PRIMARY KEY,
  type text,
  raw_text text,
  structured_json jsonb,
  file_url text,
  created_at timestamptz
);
```

## Notes & TODOs

* The `groq_client.py` is a simple HTTP wrapper; replace with official SDK if available.
* Pinecone embedding retrieval should use a proper embedding model. The scaffold expects schema entries under namespace `document-schemas`.
* LangGraph node class usage may require small adaptation depending on LangGraph package version. The scaffold shows agent logic and how to link them in `main.py` sequentially for simplicity.
* Add authentication (API keys / JWT) for production.
* Add retries, metrics, distributed task queue if scaling.
