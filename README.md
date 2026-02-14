# 🤖 RAG Chatbot

A Retrieval-Augmented Generation (RAG) Chatbot that allows users to upload PDF documents and ask questions about their content using AI-powered retrieval. The chatbot combines document embeddings, vector search, and language generation to provide accurate and context-aware answers.

# 📄 Description

This project enables you to:

Upload PDF documents of any topic (research papers, reports, manuals, etc.).

Automatically extract text and split it into chunks for efficient retrieval.

Build a vector store using embeddings for fast semantic search.

Ask natural language questions about your document and get AI-generated answers.

View source chunks that contributed to the response for transparency.

It’s ideal for research, data exploration, and intelligent document Q&A.

# 🧰 Technology Stack

Frontend & UI: Streamlit
 – modern interactive web interface.

Language Model: OpenAI GPT / Groq API for text generation.

Embeddings: Custom embedding model for semantic search.

Vector Database: In-memory vector store for retrieval (can be extended to FAISS or Pinecone).

PDF Processing: PyPDF2 / pdfminer for text extraction.

Python Libraries: streamlit, numpy, pandas, scikit-learn, tqdm.

# 📂 Project Structure
RAG-Chatbot/
│
├─ pipeline/                # Core pipeline modules
│   ├─ A_pdf_loader.py       # Extracts text from PDF
│   ├─ B_text_chunker.py     # Splits text into chunks
│   ├─ C_embedding_model.py  # Loads embedding model
│   ├─ D_vector_store.py     # Creates vector store
│   ├─ E_retriever.py        # Retrieves relevant chunks
│   └─ F_llm_generator.py    # Generates AI response from chunks
│
├─ app.py                    # Main Streamlit application
├─ requirements.txt          # Python dependencies
└─ README.md                 # Project documentation

# ⚙️ How It Works

PDF Upload: User uploads a PDF document via the sidebar.

Text Extraction: Text is extracted and split into smaller chunks.

Embedding & Indexing: Each chunk is converted into a vector embedding and stored in a vector index.

Semantic Search: When a user asks a question, the most relevant chunks are retrieved based on similarity.

Response Generation: The AI model generates a coherent answer using the retrieved chunks.

Source Transparency: Users can view the chunks used for generating the response.

# 🚀 Running the App
1. Clone the repository
git clone https://github.com/tacevooo/
cd rag-chatbot

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py

5. Using the chatbot

Enter your Groq API key in the sidebar.

Upload one or multiple PDFs.

Click Process PDF.

Ask questions in the chat interface and view the generated answers.

🛠 Features

General-purpose chatbot for any PDF document.

Supports multiple PDFs in a session.

Shows source chunks used for transparency.


📜 License

MIT License – feel free to use, modify, and share.