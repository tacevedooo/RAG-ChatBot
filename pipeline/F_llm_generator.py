from groq import Groq
from typing import List

def generate_response(query: str, context_chunks: List[str], api_key: str) -> str:
    """Generate response using Groq with RAG context"""

    context = "\n\n".join(
        [f"Document excerpt {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical information assistant. "
                           "Answer the question based ONLY on the provided medical document excerpts. "
                           "If the information is not in the excerpts, say so clearly."
            },
            {
                "role": "user",
                "content": f"""Medical Document Excerpts:
{context}

Question: {query}

Please provide a clear, accurate answer based on the medical information provided.
If you're making clinical statements, be precise and cite the relevant excerpt."""
            }
        ],
        temperature=0.2,
        max_tokens=2000
    )

    return completion.choices[0].message.content
