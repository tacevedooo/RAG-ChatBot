import faiss
from typing import List

def create_vector_store(chunks: List[str], model):
    """Create FAISS vector store from text chunks"""
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, embeddings