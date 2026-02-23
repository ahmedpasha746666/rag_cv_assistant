# 📄 RAG CV Assistant - Streamlit App

A complete Retrieval-Augmented Generation (RAG) web application for querying Ahmed Pasha's CV using semantic search and LLM-powered responses.

## 🎯 Features

### Left Sidebar
- **RAG System Information**
  - Embedding model name
  - LLM model name
  - Vector database type (ChromaDB)
  - Number of stored chunks
  - Chunk size configuration
  - Explanation of RAG pipeline workflow

### Main Section
- **Title**: RAG CV Assistant
- **Question Input**: Text box for user queries
- **Submit Button**: Process and generate responses
- **Retrieved Context**: Expandable section showing relevant CV chunks
- **Answer Display**: LLM-generated response based on context

### Example Questions
- Pre-configured clickable questions:
  - "How many years of experience does Ahmed have?"
  - "Which projects has Ahmed worked on?"
  - "What degree did Ahmed complete in 2024?"
  - "What technologies does Ahmed use?"
- Click any example to auto-populate input and trigger RAG pipeline

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Ollama running locally with required models:
  - `qwen3-embedding:8b` (embedding model)
  - `llama3.2:latest` (LLM model)

### Steps

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Setup Environment Variables**
Create a `.env` file in the project directory (if needed):
```bash
# Add any API keys or configuration here
```

3. **Ensure ChromaDB Exists**
Make sure you have already created the ChromaDB vector store by running the notebook `rag_cv.ipynb` first. The database should be located at `data_base/`.

## 🏃 Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 Configuration

Key settings in `app.py`:

```python
DB_NAME = "data_base"           # ChromaDB directory
COLLECTION_NAME = "docs"         # Collection name
EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "ollama/llama3.2:latest"
RETRIEVAL_K = 10                 # Number of chunks to retrieve
AVERAGE_CHUNK_SIZE = 450         # Average chunk size in chars
```

## 📋 RAG Pipeline Workflow

1. **Query Rewriting**: Optimizes user question for better retrieval
2. **Retrieval**: Semantic search using embeddings to find top-K chunks
3. **Reranking**: LLM-based reordering by true relevance
4. **Generation**: Grounded answer generation using retrieved context

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: Ollama (qwen3-embedding:8b)
- **LLM**: Ollama (llama3.2:latest)
- **Frameworks**: LiteLLM, Pydantic, OpenAI SDK

## 📝 Usage Tips

- Ask specific questions about Ahmed's experience, projects, education, or skills
- Use the example questions as templates
- The system only answers based on CV data (no hallucinations)
- If information isn't in the CV, the system will honestly say so

## 🔒 Error Handling

- Graceful error messages if Ollama isn't running
- Validates ChromaDB connection
- Displays friendly warnings for empty queries

## 📦 Project Structure

```
rag_cv-project/
├── app.py                 # Streamlit application
├── rag_cv.ipynb          # RAG pipeline notebook
├── requirements.txt       # Python dependencies
├── README_APP.md         # This file
├── data_base/            # ChromaDB vector store
└── .env                  # Environment variables (optional)
```

## 💡 Example Queries

Try questions like:
- "What is Ahmed's contact information?"
- "Tell me about Ahmed's education background"
- "What machine learning projects has Ahmed completed?"
- "Which programming languages does Ahmed know?"
- "What certifications does Ahmed have?"

## 🐛 Troubleshooting

**Error: Connection refused**
- Ensure Ollama is running: `ollama serve`

**Error: Model not found**
- Pull required models:
  ```bash
  ollama pull qwen3-embedding:8b
  ollama pull llama3.2:latest
  ```

**Error: Collection not found**
- Run the notebook first to create the vector database

## 📄 License

This is a personal CV assistant application built for demonstration purposes.

---

**Built with ❤️ using Streamlit and Ollama**
