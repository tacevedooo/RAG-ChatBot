from typing import List

def retrieve_relevant_chunks(query: str, model, index, chunks: List[str], k: int = 3) -> List[str]:
    """Retrieve top-k relevant chunks for a query"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks
