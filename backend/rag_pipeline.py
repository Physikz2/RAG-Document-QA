"""
RAG Pipeline Module
====================
This module implements the core Retrieval-Augmented Generation (RAG) pipeline
using LangChain, ChromaDB, and Google's Gemini API.

The pipeline:
1. Loads documents (PDF or text)
2. Splits them into chunks
3. Creates embeddings using Gemini
4. Stores vectors in ChromaDB
5. Retrieves relevant chunks for questions
6. Generates answers using Gemini LLM
"""

import os
import tempfile
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline class that handles document processing, embedding,
    vector storage, and question answering.
    
    This class orchestrates the entire RAG workflow:
    - Document loading and chunking
    - Embedding generation with Gemini
    - Vector storage in ChromaDB
    - Similarity search and retrieval
    - Answer generation with Gemini LLM
    """
    
    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG pipeline with Gemini and ChromaDB.
        
        Args:
            api_key: Google Gemini API key
            persist_directory: Where to store ChromaDB data (persists between runs)
        """
        self.api_key = api_key
        self.persist_directory = persist_directory
        
        # These will be initialized in _initialize_models()
        self.embeddings = None      # Gemini embedding model
        self.llm = None             # Gemini chat model
        self.vector_store = None    # ChromaDB vector store
        self.qa_chain = None        # LangChain QA chain
        self.current_collection_name = None  # Name of current document collection
        
        # Initialize everything
        self._initialize_models()
        
    def _initialize_models(self):
        """
        Initialize Gemini models for embeddings and chat.
        
        IMPORTANT: Google recently deprecated 'models/embedding-001'.
        Use 'models/text-embedding-004' which is the current stable model.
        """
        try:
            # ============================================================
            # EMBEDDING MODEL - Converts text to vectors
            # ============================================================
            # text-embedding-004 is Google's latest embedding model
            # It creates 768-dimensional vectors that capture semantic meaning
            # task_type="retrieval_document" optimizes for document retrieval
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",  # ✅ FIXED: Was embedding-001 (deprecated)
                google_api_key=self.api_key,
                task_type="retrieval_document"
            )
            
            # ============================================================
            # LLM MODEL - Generates answers
            # ============================================================
            # 
            # temperature=0.3 makes output focused and deterministic
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,  # Lower = more focused, higher = more creative
                convert_system_message_to_human=True,
                max_output_tokens=8192  # ← MAXIMUM LENGTH!
            )
            logger.info("✅ Gemini models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini models: {e}")
            raise
        
    def load_and_chunk_document(self, file_content: bytes, filename: str) -> List[str]:
        """
        Load a document and split it into chunks for embedding.
        
        Steps:
        1. Save file content to a temporary file
        2. Load it using PyPDFLoader (PDF) or TextLoader (TXT)
        3. Split into chunks using RecursiveCharacterTextSplitter
        
        Args:
            file_content: Raw file bytes
            filename: Original filename (determines loader type)
            
        Returns:
            List of document chunks (each chunk is a Document object)
        """
        # Save to temp file (LangChain loaders need a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # ============================================================
            # STEP 1: Load the document
            # ============================================================
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)  # Extracts text from PDF
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')  # Reads text file
            
            documents = loader.load()  # Returns list of Document objects
            
            # ============================================================
            # STEP 2: Split into chunks
            # ============================================================
            # RecursiveCharacterTextSplitter tries to split on natural boundaries:
            # 1. "\n\n" (paragraphs) - best
            # 2. "\n" (lines) - good
            # 3. " " (words) - okay
            # 4. "" (characters) - last resort
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      # Each chunk ~1000 characters
                chunk_overlap=200,    # Overlap 200 chars to maintain context
                length_function=len,  # Count characters
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"📄 Document split into {len(chunks)} chunks")
            
            return chunks
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def create_vector_store(self, chunks: List, collection_name: str = "documents"):
        """
        Create a vector store from document chunks.
        
        This is where the magic happens:
        1. Each chunk is converted to a vector (embedding)
        2. Vectors are stored in ChromaDB with HNSW indexing
        3. The vector store enables fast similarity search
        
        Args:
            chunks: List of document chunks from load_and_chunk_document()
            collection_name: Name for this document collection
            
        Returns:
            True if successful
        """
        try:
            # ============================================================
            # Create ChromaDB vector store
            # ============================================================
            # Chroma.from_documents() automatically:
            # 1. Converts each chunk to a vector using the embedding model
            # 2. Stores vectors with HNSW indexing for fast search
            # 3. Persists to disk for future use
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,      # Gemini embedding model
                persist_directory=self.persist_directory,  # Save to disk
                collection_name=collection_name
            )
            
            # Save to disk so we don't have to re-embed every time
            self.vector_store.persist()
            self.current_collection_name = collection_name
            
            # Create the QA chain (retriever + LLM)
            self._create_qa_chain()
            
            logger.info(f"✅ Vector store created with collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            raise
    
    def _create_qa_chain(self):
        """
        Create the LangChain QA chain with custom prompt.
        
        This connects:
        - Retriever (ChromaDB similarity search)
        - LLM (Gemini)
        - Custom prompt template
        
        The chain:
        1. Takes a question
        2. Retrieves relevant chunks from ChromaDB
        3. Constructs a prompt with context + question
        4. Sends to Gemini
        5. Returns the answer
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # ============================================================
        # Custom prompt template
        # ============================================================
        # This tells the LLM how to use the retrieved context
        template = """You are a helpful assistant that answers questions based on the provided context.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        **IMPORTANT: Provide a THOROUGH and COMPREHENSIVE answer. Include specific details, examples, and explanations. Write at least 1-2 paragraphs explaining the answer in detail.**
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # ============================================================
        # Retriever - searches ChromaDB
        # ============================================================
        # k=4 means "return the top 4 most relevant chunks"
        # Uses HNSW algorithm for fast similarity search
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
        
        # ============================================================
        # QA Chain - combines retriever + LLM + prompt
        # ============================================================
        # chain_type="stuff" means "stuff all context into the prompt"
        # This works well for small numbers of chunks (like 4)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True  # Returns the chunks used
        )
        
        logger.info("✅ QA chain created successfully")
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        The flow:
        1. Question is converted to a vector
        2. ChromaDB finds the 4 most similar chunks
        3. Chunks + question go to Gemini
        4. Gemini generates an answer based on the chunks
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with 'answer' and 'sources' (the chunks used)
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please upload a document first.")
        
        try:
            # Invoke the chain
            result = self.qa_chain.invoke({"query": question})
            
            # Return answer and sources
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:500] + "...",  # First 500 chars
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
    
    def clear_vector_store(self, collection_name: str = "documents"):
        """
        Clear the vector store (delete all documents).
        
        Args:
            collection_name: Name of the collection to delete
        """
        try:
            if self.vector_store:
                # Use ChromaDB's persistent client to delete the collection
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_directory)
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"🗑️ Collection {collection_name} deleted")
                except Exception as e:
                    logger.warning(f"Collection {collection_name} may not exist: {e}")
                
                # Reset the pipeline state
                self.vector_store = None
                self.qa_chain = None
                self.current_collection_name = None
        except Exception as e:
            logger.error(f"❌ Failed to clear vector store: {e}")
            raise


# ============================================================
# SINGLETON INSTANCE
# ============================================================
# This ensures we only create one RAGPipeline instance
# across the entire application (shared state)

_pipeline_instance = None

def get_pipeline(api_key: str) -> RAGPipeline:
    """
    Get or create the singleton RAG pipeline instance.
    
    This function ensures we don't re-initialize the pipeline
    on every request, which would waste resources.
    
    Args:
        api_key: Google Gemini API key
        
    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(api_key)
    return _pipeline_instance