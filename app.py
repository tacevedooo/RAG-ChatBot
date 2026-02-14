import streamlit as st
from pipeline.A_pdf_loader import extract_text_from_pdf
from pipeline.B_text_chunker import chunk_text
from pipeline.C_embedding_model import load_embedding_model
from pipeline.D_vector_store import create_vector_store
from pipeline.E_retriever import retrieve_relevant_chunks
from pipeline.F_llm_generator import generate_response

# -------------------- Page Config --------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 📄 Document Upload & Settings", unsafe_allow_html=True)
    st.write("---")

    api_key = st.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API key")

    uploaded_file = st.file_uploader("📂 Upload PDF", type=['pdf'])

    if uploaded_file and api_key:
        if st.button("🚀 Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    text = extract_text_from_pdf(uploaded_file)
                    st.success(f"✅ Extracted **{len(text)} characters**")
                    
                    chunks = chunk_text(text)
                    st.success(f"✅ Created **{len(chunks)} chunks**")
                    
                    model = load_embedding_model()
                    index, embeddings = create_vector_store(chunks, model)
                    st.success("✅ Vector store created")
                    
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.pdf_processed = True
                    st.session_state.model = model
                    
                    st.success("🎉 PDF processed successfully! You can now ask questions.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    if st.session_state.pdf_processed:
        st.info("📚 Document loaded and ready to chat")
        st.metric("Chunks Created", len(st.session_state.chunks))
        
        if st.button("🗑️ Clear Document"):
            st.session_state.pdf_processed = False
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.messages = []
            st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown("""
    1. Enter your Groq API key
    2. Upload PDF(s)
    3. Click **Process PDF**
    4. Ask questions about the document
    """)

# -------------------- Main Chat Interface --------------------
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🤖 AI RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask questions about your uploaded PDFs</p>", unsafe_allow_html=True)
st.write("---")

if not api_key:
    st.warning("👈 Please enter your Groq API key in the sidebar to begin.")
elif not st.session_state.pdf_processed:
    st.info("👈 Please upload and process a PDF document to start chatting.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        role_color = "#1E90FF" if message["role"] == "user" else "#32CD32"
        with st.chat_message(message["role"]):
            st.markdown(f"<div style='color:{role_color}'>{message['content']}</div>", unsafe_allow_html=True)
            if "sources" in message:
                with st.expander("📚 View source chunks"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)

    # Chat input
    if prompt := st.chat_input("💬 Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"<div style='color:#1E90FF'>{prompt}</div>", unsafe_allow_html=True)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document and generating response..."):
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
                    st.markdown(f"<div style='color:#32CD32'>{response}</div>", unsafe_allow_html=True)
                    
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
                    st.error(f"❌ Error generating response: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    RAG Chatbot | Powered by Groq & Streamlit
</div>
""", unsafe_allow_html=True)
