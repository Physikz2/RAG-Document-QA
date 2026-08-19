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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        """Initialize RAG pipeline with Gemini embeddings and ChromaDB."""
        self.api_key = api_key
        self.persist_directory = persist_directory
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        self.current_collection_name = None
        
        # Initialize embeddings and LLM
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize Gemini embeddings and LLM."""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key,
                task_type="retrieval_document"
            )
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            logger.info("Gemini models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            raise
        
    def load_and_chunk_document(self, file_content: bytes, filename: str) -> List[str]:
        """Load document and split into chunks."""
        # Save to temp file for loading
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Load document based on file type
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')
            
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            return chunks
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def create_vector_store(self, chunks: List, collection_name: str = "documents"):
        """Create vector store from document chunks."""
        try:
            # Delete existing collection if it exists
            if collection_name == "documents":
                # Only delete default collection to avoid accidental deletion
                pass
                
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            self.vector_store.persist()
            self.current_collection_name = collection_name
            
            # Create QA chain
            self._create_qa_chain()
            
            logger.info(f"Vector store created with collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Custom prompt template
        template = """You are a helpful assistant that answers questions based on the provided context.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and focused on the question.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info("QA chain created successfully")
    
    def query(self, question: str) -> dict:
        """Query the RAG system with a question."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please upload a document first.")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:500] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def clear_vector_store(self, collection_name: str = "documents"):
        """Clear the vector store."""
        try:
            if self.vector_store:
                # Delete the collection
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_directory)
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"Collection {collection_name} deleted")
                except Exception as e:
                    logger.warning(f"Collection {collection_name} may not exist: {e}")
                
                self.vector_store = None
                self.qa_chain = None
                self.current_collection_name = None
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise

# Singleton instance for the application
_pipeline_instance = None

def get_pipeline(api_key: str) -> RAGPipeline:
    """Get or create the singleton RAG pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(api_key)
    return _pipeline_instance