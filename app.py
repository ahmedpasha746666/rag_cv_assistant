import streamlit as st
from openai import OpenAI
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Configuration
DB_NAME = "data_base"
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "ollama/llama3.2:latest"
RETRIEVAL_K = 10
AVERAGE_CHUNK_SIZE = 450

# Initialize OpenAI client for Ollama
openai_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Pydantic models
class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

# System prompt for QA
SYSTEM_PROMPT = """
You are a precise and professional AI assistant representing an individual's personal CV and profile.

You are answering questions about the person's background, education, projects, skills, experience, certifications, and achievements.

CRITICAL INSTRUCTIONS:

1. Use ONLY the provided context extracts from the personal Knowledge Base.
2. Do NOT use outside knowledge.
3. Do NOT assume missing details.
4. Do NOT fabricate experience, skills, or qualifications.
5. If the answer is not clearly supported by the provided context, respond with:
   "I don't have enough information in the provided CV data to answer that accurately."

ANSWER REQUIREMENTS:

- Be accurate and factual.
- Be directly relevant to the question.
- Be complete but concise.
- If the question involves dates, numbers, or counts, extract them carefully from the context.
- If multiple sections are relevant (e.g., projects + skills), combine them clearly.

PERSONAL KNOWLEDGE BASE CONTEXT:
------------------------------------------------------------
{context}
------------------------------------------------------------

Now answer the user's question using only the information above.
"""

# Initialize ChromaDB
@st.cache_resource
def init_chroma():
    """Initialize ChromaDB connection and collection"""
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    return collection

# RAG Pipeline Functions
def rewrite_query(question, history=[]):
    """Rewrite the user's question into a concise, highly specific search query"""
    message = f"""
You are assisting in retrieving information from a personal CV and profile Knowledge Base.

Your job is to transform the user's question into a short, precise, and retrieval-optimized search query.

CONTEXT:
The Knowledge Base contains structured information about:
- Education
- Technical skills
- Projects
- Work experience
- Certifications
- Tools and technologies
- Achievements

Conversation history:
{history}

User's current question:
{question}

INSTRUCTIONS:

1. Rewrite the question to maximize retrieval accuracy.
2. Make it very specific and focused.
3. Remove conversational words.
4. Keep important keywords (technologies, dates, project names, roles).
5. Do NOT add new information.
6. Do NOT answer the question.
7. Do NOT include explanations or extra text.

Respond ONLY with the refined Knowledge Base search query.
"""
    
    response = completion(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": message}]
    )
    
    return response.choices[0].message.content.strip()

def fetch_context_unranked(question, collection):
    """Retrieve top-K chunks without reranking"""
    query_embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, 
        input=[question]
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=RETRIEVAL_K
    )
    
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    
    return chunks

def rerank(question, chunks):
    """Rerank retrieved chunks by relevance using LLM"""
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = completion(model=LLM_MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    
    return [chunks[i - 1] for i in order]

def fetch_context(question, collection):
    """Retrieve and rerank relevant chunks"""
    chunks = fetch_context_unranked(question, collection)
    return rerank(question, chunks)

def make_rag_messages(question, history, chunks):
    """Build messages for RAG with context"""
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" 
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

def answer_question(question: str, collection, history: list = None):
    """Complete RAG pipeline: rewrite → retrieve → rerank → generate"""
    if history is None:
        history = []
    
    # Rewrite query
    rewritten_query = rewrite_query(question, history)
    
    # Fetch and rerank context
    chunks = fetch_context(rewritten_query, collection)
    
    # Generate answer
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=LLM_MODEL, messages=messages)
    answer = response.choices[0].message.content
    
    return {
        'answer': answer,
        'rewritten_query': rewritten_query,
        'chunks': chunks
    }

# Streamlit App
def main():
    st.set_page_config(
        page_title="RAG CV Assistant",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize ChromaDB collection
    collection = init_chroma()
    
    # Get chunk count
    try:
        chunk_count = collection.count()
    except:
        chunk_count = 0
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 System Info")
        
        st.markdown("### 🧠 Configuration")
        st.info(f"""
**Embedding:** `{EMBEDDING_MODEL.split(':')[0]}`  
**LLM:** `{LLM_MODEL.split('/')[-1].split(':')[0]}`  
**Database:** ChromaDB  
**Chunks:** {chunk_count}  
**Chunk Size:** ~{AVERAGE_CHUNK_SIZE} chars
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 RAG Pipeline")
        st.markdown("""
        1. **Query Rewrite** - Optimize search
        2. **Retrieval** - Find relevant chunks
        3. **Reranking** - Order by relevance
        4. **Generation** - Create answer
        
        All answers are grounded in Ahmed's CV data.
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit & Ollama")
    
    # Main section
    st.title("📄 RAG CV Assistant")
    st.markdown("Ask me anything about **Ahmed Pasha's** CV, experience, projects, and skills.")
    st.markdown("")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'process_question' not in st.session_state:
        st.session_state.process_question = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""
    
    # Example questions section
    st.markdown("### 💡 Try These Examples")
    
    example_questions = [
        "how many years of experience does ahmed have",
        "Ahmed's education?",
        "how many projects has ahmed worked on",
    ]
    
    # Create columns for example questions - compact layout
    cols = st.columns(4)
    for idx, question in enumerate(example_questions):
        with cols[idx]:
            if st.button(f"💬 {question}", key=f"example_{idx}", use_container_width=True):
                st.session_state.current_question = question
                st.session_state.process_question = True
                st.rerun()
    
    st.markdown("---")
    
    # Question input
    user_question = st.text_input(
        "🔍 Your Question:",
        value=st.session_state.current_question,
        placeholder="Ask about Ahmed's CV...",
        key="question_input"
    )
    
    # Buttons row
    col1, col2 = st.columns([3, 1])
    with col1:
        submit_button = st.button("🚀 Submit", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.current_question = ""
            st.session_state.last_result = None
            st.rerun()
    
    # Check if we should process the question (either Submit clicked or auto-process from example)
    should_process = (submit_button and user_question.strip()) or (st.session_state.process_question and user_question.strip())
    
    # Reset process flag
    if st.session_state.process_question:
        st.session_state.process_question = False
    
    # Process question
    if should_process:
        with st.spinner("🔍 Searching CV and generating response..."):
            try:
                # Get answer using RAG pipeline
                result = answer_question(user_question, collection)
                st.session_state.last_result = result
                st.session_state.last_question = user_question
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please ensure Ollama is running and the models are available.")
                st.session_state.last_result = None
    
    elif submit_button:
        st.warning("⚠️ Please enter a question.")
    
    # Display results if available
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        st.markdown("---")
        
        # Display answer prominently
        st.markdown("### ✨ Answer")
        st.success(result['answer'])
        
        # Display additional info in expanders
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🔄 Rewritten Query"):
                st.info(result['rewritten_query'])
        
        with col2:
            with st.expander(f"📚 Retrieved Context ({len(result['chunks'])} chunks)"):
                for idx, chunk in enumerate(result['chunks'][:3], 1):
                    st.markdown(f"**Chunk {idx}**")
                    st.caption(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                    st.text(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
                    if idx < 3:
                        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption("💬 Powered by RAG | All answers are grounded in Ahmed's CV data")

if __name__ == "__main__":
    main()
