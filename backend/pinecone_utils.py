# backend/pinecone_utils.py
import os
import json
from loguru import logger
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "document-schemas")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize sentence transformer
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


class PineconeClient:
    def __init__(self, index_name=PINECONE_INDEX_NAME):
        self.index_name = index_name
        self.embedding_model = embedding_model
        
        # Create index if it doesn't exist
        if not pc.has_index(index_name):
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                ),
                deletion_protection="disabled",
            )
            logger.info(f"Index {index_name} created successfully")
        
        # Get index reference
        self.index = pc.Index(index_name)

    def embed_text(self, text: str) -> list:
        """Convert text to embedding vector using sentence-transformer."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.exception("Failed to embed text")
            raise

    def upsert_schema(self, id: str, text: str, metadata: dict, namespace: str = "document-schemas"):
        """
        Embed text and upsert vector with metadata to Pinecone.
        
        Args:
            id: Unique identifier for the vector
            text: Text content to embed
            metadata: Dictionary containing schema, examples, template, etc.
            namespace: Pinecone namespace (default: "document-schemas")
        """
        try:
            # Generate embedding
            vector = self.embed_text(text)
            
            # Sanitize metadata: convert dict/list into JSON strings, lists -> list[str]
            safe_meta = {}
            for k, v in (metadata or {}).items():
                if isinstance(v, dict):
                    safe_meta[k] = json.dumps(v)
                elif isinstance(v, list):
                    # convert list items to strings
                    safe_meta[k] = [str(x) for x in v]
                elif isinstance(v, (str, int, float, bool)) or v is None:
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)

            record = {"id": id, "values": vector, "metadata": safe_meta}
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[record],
                namespace=namespace
            )
            logger.info(f"Upserted schema {id} to namespace {namespace}")
        except Exception as e:
            logger.exception("Failed to upsert schema")
            raise

    def query_schema(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: str = "document-schemas",
        filter_metadata: dict = None
    ) -> list:
        """
        Query Pinecone using text similarity.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            namespace: Pinecone namespace to query
            filter_metadata: Optional metadata filter dict
            
        Returns:
            List of matching results with metadata
        """
        try:
            # Embed the query text
            query_vector = self.embed_text(query_text)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter_metadata
            )
            
            # Extract matches
            matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
            logger.info(f"Query returned {len(matches)} matches from namespace {namespace}")
            return matches
        except Exception as e:
            logger.exception("Failed to query schema")
            raise

    def query_by_doc_type(
        self,
        doc_type: str,
        top_k: int = 5,
        namespace: str = "document-schemas"
    ) -> list:
        """
        Query schemas by document type using metadata filter.
        
        Args:
            doc_type: Document type to filter by
            top_k: Number of results to return
            namespace: Pinecone namespace
            
        Returns:
            List of matching schemas
        """
        try:
            # Use a simple query vector (dummy) and filter by metadata
            # For better results, embed a query like "schema for {doc_type}"
            query_text = f"schema for {doc_type}"
            query_vector = self.embed_text(query_text)
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter={"type": {"$eq": doc_type}}
            )
            
            matches = results.get('matches', [])
            logger.info(f"Found {len(matches)} schemas for doc_type={doc_type}")
            return matches
        except Exception as e:
            logger.exception("Failed to query by doc type")
            raise


# Global instance
pinecone_client = PineconeClient()