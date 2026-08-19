"""
RAG Document QA System - Main API Entry Point
================================================
This module implements the FastAPI application for the RAG Document QA System.
It provides endpoints for uploading documents, asking questions, and managing
the RAG pipeline state.

Endpoints:
- GET  /         - Health check
- POST /upload   - Upload a document (PDF/TXT)
- POST /ask      - Ask a question about the uploaded document
- POST /clear    - Clear the current document from the vector store
- GET  /status   - Get the current system status
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from dotenv import load_dotenv

# Import the RAG pipeline singleton getter
# get_pipeline() returns a single shared instance of RAGPipeline
from rag_pipeline import get_pipeline, RAGPipeline

# ================================================================
# ENVIRONMENT SETUP
# ================================================================

# Load environment variables from .env file
# This reads GEMINI_API_KEY and any other env vars
load_dotenv()

# Configure logging to show INFO level messages
# This gives us visibility into what the server is doing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# FASTAPI APPLICATION SETUP
# ================================================================

# Initialize the FastAPI app with metadata
# This metadata appears in the /docs (Swagger UI) and /redoc pages
app = FastAPI(
    title="RAG Document QA System",
    description="Document question-answering using RAG with Gemini and ChromaDB",
    version="1.0.0"
)

# ================================================================
# CORS CONFIGURATION
# ================================================================

# CORS (Cross-Origin Resource Sharing) allows the frontend (running on a different port)
# to make requests to this backend API without being blocked by the browser.
# 
# allow_origins=["*"] means ANY domain can call this API.
# In production, you should restrict this to your specific frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ For production: replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# ================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ================================================================

# These models define the structure of request bodies and responses.
# FastAPI automatically validates incoming requests against these schemas.

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str  # The user's question


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    answer: str                    # The generated answer
    sources: Optional[list] = None # The chunks used to generate the answer


class UploadResponse(BaseModel):
    """Response body for the /upload endpoint."""
    message: str       # Success/status message
    document_id: str   # Unique ID for this document
    chunk_count: int   # Number of chunks the document was split into

# ================================================================
# RAG PIPELINE INITIALIZATION
# ================================================================

# Read the Gemini API key from environment variables
# This must be set in the .env file or as a Render environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # If no API key is found, the application fails fast with a clear error
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Get the singleton RAG pipeline instance
# This ensures we only initialize the pipeline once across all requests
# The pipeline includes: embeddings model, LLM, ChromaDB connection
pipeline = get_pipeline(API_KEY)

# Create the uploads directory if it doesn't exist
# This is where temporary files are stored during processing
os.makedirs("uploads", exist_ok=True)

# ================================================================
# ENDPOINT: GET / (Health Check)
# ================================================================

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    Returns basic service information.
    """
    return {
        "message": "RAG Document QA System API",
        "status": "running",
        "version": "1.0.0"
    }


# ================================================================
# ENDPOINT: POST /upload (Upload Document)
# ================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or text) for processing.
    
    The document goes through the full RAG pipeline:
    1. Loaded from the uploaded file
    2. Split into chunks (chunk_size=1000, overlap=200)
    3. Each chunk is converted to a vector using Gemini embeddings
    4. Vectors are stored in ChromaDB with HNSW indexing
    5. A QA chain is created for answering questions
    
    Args:
        file: The uploaded file (PDF or TXT)
    
    Returns:
        UploadResponse: Contains document_id and chunk_count
    
    Raises:
        HTTPException: If file type is invalid, empty, or processing fails
    """
    try:
        # --- Step 1: Validate file type ---
        # Only allow PDF and text files
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and text files are supported"
            )
        
        # --- Step 2: Read file content ---
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # --- Step 3: Generate a unique document ID ---
        # This ID is used as the collection name in ChromaDB
        doc_id = str(uuid.uuid4())[:8]
        collection_name = f"doc_{doc_id}"
        
        # --- Step 4: Load and chunk the document ---
        # The pipeline handles the actual loading and splitting
        logger.info(f"Processing document: {file.filename}")
        chunks = pipeline.load_and_chunk_document(content, file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the document"
            )
        
        # --- Step 5: Clear previous document ---
        # This removes the old document from ChromaDB before storing the new one
        # Only one document is stored at a time (single-document mode)
        pipeline.clear_vector_store()
        
        # --- Step 6: Create vector store ---
        # This embeds all chunks and stores them in ChromaDB
        # The QA chain is automatically created inside create_vector_store()
        pipeline.create_vector_store(chunks, collection_name=collection_name)
        
        logger.info(f"Document uploaded successfully: {file.filename} with {len(chunks)} chunks")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_id=doc_id,
            chunk_count=len(chunks)
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (they already have status codes)
        raise e
    except Exception as e:
        # Catch all other exceptions and return a 500 error
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# ================================================================
# ENDPOINT: POST /ask (Ask a Question)
# ================================================================

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about the uploaded document.
    
    The RAG pipeline performs the following steps:
    1. The question is converted to a vector using Gemini embeddings
    2. ChromaDB performs HNSW similarity search to find the top-k chunks
    3. The top chunks are combined with the question in a prompt
    4. Gemini generates an answer based on the context
    
    Args:
        request: AskRequest containing the question
    
    Returns:
        AskResponse: Contains the answer and source chunks
    
    Raises:
        HTTPException: If question is empty, no document is loaded, or processing fails
    """
    try:
        # --- Step 1: Validate the question ---
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # --- Step 2: Query the RAG pipeline ---
        # This performs the full retrieval + generation flow
        result = pipeline.query(request.question)
        
        # --- Step 3: Return the answer with sources ---
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except ValueError as e:
        # This happens when no document is loaded
        # (pipeline.query() raises ValueError if QA chain is None)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Question error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


# ================================================================
# ENDPOINT: POST /clear (Clear Document)
# ================================================================

@app.post("/clear")
async def clear_documents():
    """
    Clear all documents from the vector store.
    
    This deletes the current collection from ChromaDB and resets the pipeline state.
    After this, you must upload a new document before asking questions.
    
    Returns:
        dict: Success message
    """
    try:
        pipeline.clear_vector_store()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing vector store: {str(e)}"
        )


# ================================================================
# ENDPOINT: GET /status (Get System Status)
# ================================================================

@app.get("/status")
async def get_status():
    """
    Get the current status of the system.
    
    Returns information about:
    - Whether a document is loaded (has_document)
    - The current collection name in ChromaDB
    - Whether the API key is configured
    
    Returns:
        dict: System status information
    """
    return {
        "has_document": pipeline.vector_store is not None,
        "collection_name": pipeline.current_collection_name,
        "api_key_configured": API_KEY is not None
    }


# ================================================================
# MAIN ENTRY POINT (For Local Development)
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application with uvicorn
    # --host 0.0.0.0 means "listen on all network interfaces"
    # --port 8000 is the default port
    # --reload would auto-restart on code changes (not used here)
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000        # Default FastAPI port
    )