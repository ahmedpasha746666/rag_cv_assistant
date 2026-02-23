# 🤖 Advanced RAG Pipeline for Personal CV/Resume

A production-ready Retrieval-Augmented Generation (RAG) system built from scratch for intelligent question-answering over personal CV and resume documents. This project demonstrates advanced RAG techniques including LLM-based semantic chunking, vector retrieval with reranking, and query optimization.

## 🌟 Features

- **🧠 LLM-Based Semantic Chunking**: Uses large language models to intelligently split documents into meaningful chunks with structured metadata (headline + summary + original text)
- **🔍 Vector Similarity Search**: ChromaDB-powered semantic search with Ollama embeddings (`qwen3-embedding:8b`)
- **🎯 LLM Reranking**: Two-stage retrieval with LLM-based reranking for improved relevance
- **✨ Query Rewriting**: Automatic query optimization to extract core intent from conversational questions
- **📊 Visualization**: Interactive 2D and 3D t-SNE plots for embedding quality analysis
- **🔒 Type-Safe Outputs**: Pydantic schemas for structured LLM responses
- **⚡ Flexible Architecture**: Works with both native tools and LangChain abstractions

## 🛠️ Tech Stack

### Core Dependencies
- **Python 3.10+**
- **LiteLLM** - Unified LLM interface
- **Ollama** - Local LLM backend (`llama3.2:latest`)
- **ChromaDB** - Vector database with persistent storage
- **Pydantic** - Data validation and structured outputs
- **NumPy** - Numerical operations
- **scikit-learn** - t-SNE dimensionality reduction
- **Plotly** - Interactive visualizations

### Optional LangChain Integration
- `langchain-community` - Document loaders
- `langchain-text-splitters` - Text splitting utilities
- `langchain-ollama` - Ollama embeddings wrapper

## 📁 Project Structure

```
rg_cv_project/
├── rag_cv.ipynb             # Main RAG pipeline notebook (fully documented)
├── data2/                   # CV/Resume markdown files 
│   ├── cv/
│   ├── education/
│   ├── goals/
│   ├── partime/
│   └── technical concetps/
├── data_base/               # ChromaDB persistent storage
└── README.md                # Project documentation
```

## 🚀 Installation

### Prerequisites

1. **Install Ollama** (if not already installed):
   ```bash
   # Visit https://ollama.ai to download and install
   ```

2. **Pull Required Models**:
   ```bash
   ollama pull llama3.2:latest
   ollama pull qwen3-embedding:8b
   ```

3. **Verify Models**:
   ```bash
   ollama list
   ```

### Python Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ahmedpasha746666/rag_cv_assistant.git
   cd rg_agri_project
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install openai litellm chromadb pydantic python-dotenv
   pip install numpy scikit-learn plotly
   pip install tqdm  # For progress bars
   
   # Optional: LangChain components
   pip install langchain-community langchain-text-splitters langchain-ollama
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   # Add any API keys or configuration here
   # Example:
   # OPENAI_API_KEY=your_key_here  # If using OpenAI models
   ```

## 📖 Usage

### Running the Complete Pipeline

1. **Open the Notebook**:
   ```bash
   jupyter notebook rag_cv.ipynb
   ```
   
2. **Execute Cells Sequentially**:
   - **Steps 1-8**: Document loading and preparation
   - **Steps 9-16**: LLM-based chunking (⚠️ May take time)
   - **Steps 17-22**: Embedding generation and visualization
   - **Steps 23-30**: Retrieval and reranking setup
   - **Steps 31-34**: Question answering system
   - **Steps 35-36**: Test queries

### Quick Start Example

```python
# Load environment and configuration
from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize OpenAI client (pointing to Ollama)
from openai import OpenAI
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Ask a question
answer, context = answer_question("What is Ahmed Pasha's work experience?", [])
print(answer)
```

### Key Functions

- **`fetch_documents()`** - Load all markdown files from `data2/`
- **`create_chunks(documents)`** - LLM-based semantic chunking
- **`create_embeddings(chunks)`** - Generate and store vector embeddings
- **`fetch_context(question)`** - Retrieve + rerank relevant chunks
- **`answer_question(question, history=[])`** - End-to-end Q&A with context

## 🏗️ Pipeline Architecture

```
┌─────────────────┐
│  Load Documents │
│   (Markdown)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Chunking   │
│ (Structured)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │
│   (ChromaDB)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ User Query      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Rewriting │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Retrieval│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Reranking  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer Generation│
│   (Grounded)    │
└─────────────────┘
```

## 🎨 Visualizations

The pipeline includes interactive visualizations:

- **2D t-SNE Plot**: Projects high-dimensional embeddings to 2D space
- **3D t-SNE Plot**: Interactive 3D scatter plot with rotation/zoom
- **Color-Coded Categories**: Documents colored by type (CV, education, goals, etc.)

These visualizations help validate:
- Similar documents cluster together
- Semantic relationships are preserved
- Embedding quality across categories

## ⚙️ Configuration

Key parameters in the notebook (cell 3):

```python
DB_NAME = "data_base"                      # ChromaDB storage path
collection_name = "docs"                    # Collection name
embedding_model = "qwen3-embedding:8b"      # Ollama embedding model
KNOWLEDGE_BASE_PATH = Path("data2/")        # Document directory
AVERAGE_CHUNK_SIZE = 450                    # Target chunk size (chars)
MODEl = "ollama/llama3.2:latest"           # LLM for chunking/reranking
RETRIEVAL_K = 10                            # Number of chunks to retrieve
```

### Customization

- **Change embedding model**: Update `embedding_model` variable
- **Adjust chunk size**: Modify `AVERAGE_CHUNK_SIZE` (default: 450 chars)
- **Change LLM**: Update `MODEl` to any LiteLLM-compatible model
- **Tune retrieval**: Adjust `RETRIEVAL_K` for more/fewer context chunks
- **Data source**: Point `KNOWLEDGE_BASE_PATH` to your markdown directory

## 🧪 Testing

The notebook includes test queries in Steps 35-36:

```python
# Example 1: Work experience
answer_question("i want to know ahmed pashas work experience", [])

# Example 2: Certificates
answer_question("tell me about certificates and links", [])
```

Add your own test cases at the end of the notebook.

## 📊 Performance Notes

- **Chunking**: ~5-10 seconds per document (depends on LLM speed)
- **Embedding**: ~1-2 seconds for 50-100 chunks
- **Retrieval**: < 1 second for similarity search
- **Reranking**: ~2-3 seconds for 10 chunks
- **Total Query Time**: ~3-5 seconds per question

## 🔧 Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Model not found**:
   ```bash
   # Verify models are installed
   ollama list
   # Pull missing models
   ollama pull llama3.2:latest
   ```

3. **ChromaDB errors**:
   - Delete the database folder and recreate embeddings
   - Check write permissions on `DB_NAME` directory

4. **Memory issues**:
   - Reduce `RETRIEVAL_K` to fetch fewer chunks
   - Process documents in smaller batches
   - Use a smaller embedding model

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for PDF/DOCX document loading
- [ ] Implement conversation memory for multi-turn Q&A
- [ ] Add evaluation metrics (precision, recall, NDCG)
- [ ] Create web UI with Streamlit/Gradio
- [ ] Add batch processing for large document sets
- [ ] Implement hybrid search (BM25 + vector)

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Ollama** - Local LLM infrastructure
- **ChromaDB** - Vector database
- **LangChain** - RAG framework inspiration
- **LiteLLM** - Unified LLM interface

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**⭐ If you find this project useful, please consider giving it a star!**
