from sentence_transformers import SentenceTransformer
import streamlit as st

# Load embedding model (cached)
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')
