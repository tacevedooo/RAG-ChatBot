# Medical PDF RAG Chatbot 🏥

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows you to upload medical PDFs and ask questions about their content using Claude AI.

## Features

- 📄 **PDF Upload**: Upload any medical PDF document
- 🔍 **Smart Retrieval**: Uses semantic search to find relevant information
- 🤖 **AI-Powered**: Leverages Claude Sonnet 4 for accurate responses
- 💬 **Chat Interface**: User-friendly conversational interface
- 📚 **Source Citations**: View the document chunks used to generate each answer
- 🧠 **Vector Search**: FAISS-based similarity search for fast retrieval

## How It Works

1. **Document Processing**: PDF is uploaded and text is extracted
2. **Chunking**: Text is split into overlapping chunks for better context
3. **Embedding**: Each chunk is converted to a vector embedding using sentence-transformers
4. **Storage**: Embeddings are stored in a FAISS vector database
5. **Retrieval**: When you ask a question, the most relevant chunks are retrieved
6. **Generation**: Claude uses the retrieved chunks to generate an accurate answer

## Installation

### Prerequisites
- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Steps

1. **Clone or download this project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run medical_rag_chatbot.py
```

4. **Open your browser** to `http://localhost:8501`

## Usage

1. **Enter API Key**: Paste your Anthropic API key in the sidebar
2. **Upload PDF**: Select a medical PDF document
3. **Process**: Click "Process PDF" to create the vector database
4. **Ask Questions**: Type your questions in the chat interface
5. **View Sources**: Expand source chunks to see where answers came from

## Example Questions

- "What are the symptoms described in this document?"
- "What treatment options are mentioned?"
- "What are the contraindications?"
- "Summarize the key findings"
- "What dosage recommendations are provided?"

## Configuration

You can adjust these parameters in the code:

- `chunk_size`: Size of text chunks (default: 500 words)
- `overlap`: Overlap between chunks (default: 50 words)
- `k`: Number of relevant chunks to retrieve (default: 3)
- `model`: Claude model to use (default: claude-sonnet-4-20250514)

## Technologies Used

- **Streamlit**: Web application framework
- **Claude API**: AI language model
- **Sentence Transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **PyPDF2**: PDF text extraction

## File Structure

```
.
├── medical_rag_chatbot.py   # Main application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### PDF Not Processing
- Ensure the PDF contains extractable text (not scanned images)
- Try a different PDF if issues persist

### API Errors
- Verify your Anthropic API key is valid
- Check your API usage limits

### Memory Issues
- For large PDFs, consider reducing chunk_size
- Reduce the number of retrieved chunks (k parameter)

## Security Notes

⚠️ **Important**: 
- Never commit your API key to version control
- Keep medical documents confidential
- This is for educational/research purposes only
- Always consult qualified healthcare professionals for medical advice

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions, please check:
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Disclaimer**: This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
