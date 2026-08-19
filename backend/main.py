import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from dotenv import load_dotenv

from rag_pipeline import get_pipeline, RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document QA System",
    description="Document question-answering using RAG with Gemini and ChromaDB",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: Optional[list] = None

class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunk_count: int

# Initialize pipeline
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

pipeline = get_pipeline(API_KEY)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "RAG Document QA System API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or text) for processing.
    The document will be chunked, embedded, and stored in ChromaDB.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and text files are supported"
            )
        
        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Generate document ID
        doc_id = str(uuid.uuid4())[:8]
        collection_name = f"doc_{doc_id}"
        
        # Load and chunk the document
        logger.info(f"Processing document: {file.filename}")
        chunks = pipeline.load_and_chunk_document(content, file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the document"
            )
        
        # Clear previous vector store
        pipeline.clear_vector_store()
        
        # Create vector store with the chunks
        pipeline.create_vector_store(chunks, collection_name=collection_name)
        
        logger.info(f"Document uploaded successfully: {file.filename} with {len(chunks)} chunks")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_id=doc_id,
            chunk_count=len(chunks)
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about the uploaded document.
    The system will retrieve relevant chunks and generate an answer.
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Query the RAG pipeline
        result = pipeline.query(request.question)
        
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except ValueError as e:
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

@app.post("/clear")
async def clear_documents():
    """Clear all documents from the vector store."""
    try:
        pipeline.clear_vector_store()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing vector store: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get the current status of the system."""
    return {
        "has_document": pipeline.vector_store is not None,
        "collection_name": pipeline.current_collection_name,
        "api_key_configured": API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)