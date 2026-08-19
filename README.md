# 📄 RAG Document QA System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-00A67E?style=for-the-badge&logo=chromadb&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

A production-ready **Retrieval-Augmented Generation (RAG)** document question-answering system. Upload PDFs or text files, and ask questions about them using Google's Gemini LLM with vector-based semantic search.

---

## 🌐 Live Demo

- **Frontend**: [https://physikz2.github.io/RAG-Document-QA/](https://physikz2.github.io/RAG-Document-QA/)
- **Backend API**: [https://rag-document-qa-ty6i.onrender.com/](https://rag-document-qa-ty6i.onrender.com/)
- **API Docs (Swagger UI)**: [https://rag-document-qa-ty6i.onrender.com/docs](https://rag-document-qa-ty6i.onrender.com/docs)
- **GitHub Repository**: [https://github.com/Physikz2/RAG-Document-QA](https://github.com/Physikz2/RAG-Document-QA)

---

## 📖 Overview

This system implements a complete RAG pipeline:

1. **Document Upload**: User uploads a PDF or text file via the web interface or API.
2. **Chunking & Embedding**: The document is split into semantic chunks (1000 chars) and converted to vectors using Gemini embeddings.
3. **Vector Storage**: Chunks are stored in ChromaDB with HNSW indexing for fast similarity search.
4. **Question Answering**: User asks a question → ChromaDB retrieves top-k relevant chunks → Gemini generates an answer based on the context.
5. **Response**: Returns the answer with source citations showing which chunks were used.

---

## ✨ Features

- **Upload & Process**: Supports PDF and text files with automatic chunking
- **Semantic Search**: ChromaDB vector database with HNSW indexing
- **RAG Pipeline**: LangChain orchestration with Gemini LLM
- **Interactive API**: Swagger UI at `/docs` for testing
- **Clean Web Interface**: Drag-and-drop upload with real-time answers
- **Source Attribution**: Shows which chunks were used to generate answers
- **Deployable**: Ready for Render (free tier) and GitHub Pages

---

## 🛠️ Tech Stack & How It's Used

| Technology | Badge | Role in the Project |
| :--- | :--- | :--- |
| **Python** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Primary language. Implements the RAG pipeline, API, and all backend logic. |
| **FastAPI** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) | REST API framework. Provides `/upload`, `/ask`, `/clear`, and `/status` endpoints with auto-generated OpenAPI docs. |
| **LangChain** | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) | Orchestrates the RAG pipeline: document loading, text splitting, embeddings, vector store, and retrieval QA chain. |
| **ChromaDB** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-00A67E?style=for-the-badge&logo=chromadb&logoColor=white) | Local vector database. Stores embeddings and performs HNSW similarity search to retrieve relevant chunks. |
| **Google Gemini** | ![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white) | LLM provider. `gemini-embedding-001` for embeddings, `gemini-3.5-flash` for answer generation. |
| **HTML/CSS/JS** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) | Single-page frontend. Handles file uploads, question input, and displays answers with sources. |
| **Render** | ![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white) | Deployment platform. Hosts the backend API (free tier) and serves the frontend as a static site. |

---

## 📋 Prerequisites

- **Python 3.12**: This project is developed and tested with Python 3.12. Ensure you have it installed. **Render uses the `.python-version` file to enforce this.**
- **Git** (optional, for cloning)
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Physikz2/RAG-Document-QA.git
cd RAG-Document-QA
```

### 2. Create a Virtual Environment (Python 3.12)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. (Optional) Install Additional Dependencies

If you encounter issues with the embedding model:

```bash
pip install google-genai  # For the latest Gemini SDK
```

---

## 🚀 How to Use

### Run the Backend Locally

```bash
cd backend
python main.py
```

The API will be available at: `http://localhost:8000`

### Run the Frontend Locally

In a separate terminal:

```bash
cd frontend
python -m http.server 8001
```

Open: `http://localhost:8001`

### Test via Swagger UI

Visit: `http://localhost:8000/docs`

- **POST /upload**: Upload a PDF or text file
- **POST /ask**: Ask a question about the document
- **POST /clear**: Clear the current document
- **GET /status**: Check system status

---

## 📊 Sample Output

### Upload Response

```json
{
  "message": "Document uploaded and processed successfully",
  "document_id": "abc12345",
  "chunk_count": 11
}
```

### Ask Response

```json
{
  "answer": "Kevin Tamkei worked at Royal Bank of Canada (RBC) as a Lead Quality Engineer from April 2023 to February 2024, where they designed testing strategies for financial transaction workflows.",
  "sources": [
    {
      "content": "Lead Quality Engineer, RBC (via TCS), Toronto, ON, April 2023 – February 2024...",
      "metadata": { "source": "resume.pdf", "page": 1 }
    }
  ]
}
```

---

## 📂 Project Structure

```
RAG-Document-QA/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── rag_pipeline.py         # RAG pipeline implementation
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables (gitignored)
│   └── chroma_db/              # ChromaDB vector storage
├── frontend/
│   └── index.html              # Single-page web interface
├── .python-version             # Python version for Render (3.12)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🚀 Deployment

### Backend (Render)

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **"New +"** → **"Web Service"**
4. Connect your GitHub repository
5. Configure:

   - **Name**: `rag-document-qa-backend`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port 10000`

6. **Add Environment Variable**:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: `your_gemini_api_key_here` (set as **Secret**)

7. **Ensure `.python-version` file exists** in the root with `3.12` to enforce Python version compatibility.

8. Click **"Create Web Service"**

### Frontend (GitHub Pages)

1. In your repository → **Settings** → **Pages**
2. Set source to `main` branch and `/frontend` folder
3. Your frontend will be live at: `https://physikz2.github.io/RAG-Document-QA/`

---

## 🔧 Environment Variables

| Variable | Description | Required |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API key | ✅ Yes |

---

## 📝 License

This project is for portfolio and interview demonstration purposes.

---

## 🤝 Contributing

This is a personal portfolio project. For suggestions or issues, please open a GitHub issue.

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for RAG orchestration
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Google Gemini](https://ai.google.dev/gemini) for LLM and embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Render](https://render.com/) for free deployment