import streamlit as st
from groq import Groq
from pathlib import Path
import PyPDF2
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Medical PDF Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None

# Load embedding model (cached)
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def create_vector_store(chunks: List[str], model):
    """Create FAISS vector store from text chunks"""
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, embeddings

def retrieve_relevant_chunks(query: str, model, index, chunks: List[str], k: int = 3) -> List[str]:
    """Retrieve top-k relevant chunks for a query"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_response(query: str, context_chunks: List[str], api_key: str) -> str:
    """Generate response using Groq (LLaMA3) with RAG context"""
    
    context = "\n\n".join(
        [f"Document excerpt {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical information assistant. Answer ONLY based on the provided document excerpts. If the information is not in the excerpts, say so clearly."
            },
            {
                "role": "user",
                "content": f"""
Medical Document Excerpts:
{context}

Question: {query}

Provide a clear and accurate answer strictly based on the excerpts above.
"""
            }
        ],
        temperature=0.2,
        max_tokens=1500
    )

    return completion.choices[0].message.content

# Main UI
st.title("🏥 Medical PDF Chatbot with RAG")
st.markdown("Upload a medical PDF document and ask questions about its content using AI-powered retrieval.")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("📄 Document Upload")
    
    # API Key input
    api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
    
    # PDF upload
    uploaded_file = st.file_uploader("Upload Medical PDF", type=['pdf'])
    
    if uploaded_file and api_key:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    text = extract_text_from_pdf(uploaded_file)
                    st.success(f"✅ Extracted {len(text)} characters")
                    
                    chunks = chunk_text(text)
                    st.success(f"✅ Created {len(chunks)} chunks")
                    
                    model = load_embedding_model()
                    index, embeddings = create_vector_store(chunks, model)
                    st.success("✅ Vector store created")
                    
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.pdf_processed = True
                    st.session_state.model = model
                    
                    st.success("🎉 PDF processed successfully! You can now ask questions.")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    if st.session_state.pdf_processed:
        st.success("📚 Document loaded and ready")
        st.metric("Chunks", len(st.session_state.chunks))
        
        if st.button("Clear Document"):
            st.session_state.pdf_processed = False
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ How to use")
    st.markdown("""
    1. Enter your Groq API key
    2. Upload a medical PDF
    3. Click 'Process PDF'
    4. Ask questions about the document
    """)

# Main chat interface
if not api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to begin.")
elif not st.session_state.pdf_processed:
    st.info("👈 Please upload and process a PDF document to start chatting.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View source chunks"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
    
    if prompt := st.chat_input("Ask a question about the medical document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating response..."):
                try:
                    model = load_embedding_model()
                    relevant_chunks = retrieve_relevant_chunks(
                        prompt, 
                        model, 
                        st.session_state.index, 
                        st.session_state.chunks,
                        k=3
                    )
                    
                    response = generate_response(prompt, relevant_chunks, api_key)
                    st.markdown(response)
                    
                    with st.expander("📚 View source chunks"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": relevant_chunks
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Medical PDF RAG Chatbot | Powered by Groq & Streamlit
</div>
""", unsafe_allow_html=True)
