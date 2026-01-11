
# Multi-File AI Chatbot with Memory and Document Summarization

A Streamlit app that lets you chat with the content of your uploaded files (PDF, DOCX, images) using AI, with conversational memory, document summaries, and retrieval-augmented generation. Now powered by GPT-4o-mini and LangGraph.

---

## 🚀 Features

- **Multi-file upload:** Supports PDF, DOCX, PNG, JPG, JPEG.
- **Conversational memory:** Remembers previous chat turns for context.
- **Retrieval-Augmented Generation (RAG):** Answers are based only on your uploaded files.
- **OCR for images:** Extracts text from images using the OCR.space API.
- **Document summarization for images:** Uses GPT-4o-mini to summarize and categorize each uploaded image file (not PDFs or DOCX).
- **Chunk filtering:** Ignores text chunks shorter than 50 characters for better relevance.
- **Chunk indexing:** Each chunk is indexed and shown in the preview for clarity.
- **Privacy:** No files or chat data are stored.
- **Modular architecture:** Code is organized into reusable modules for easy maintenance and extension.

---

## 🛠️ Tech Stack

- Python, Streamlit
- LangChain, LangGraph, OpenAI (GPT-4o-mini), FAISS
- PyPDF2, python-docx, Pillow, requests
- OCR.space API

---

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── chatbot.py         # LangGraph chatbot setup
│   ├── config.py          # Configuration and constants
│   ├── file_processing.py # Text extraction and OCR
│   ├── ui.py              # Streamlit UI components
│   └── vectorstore.py     # FAISS vectorstore creation
├── test_app.py            # Unit tests
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── data/                  # Sample data files
```

---

## ⚡ Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys to `.streamlit/secrets.toml`:
   ```toml
   openai_api_key = "sk-..."
   ocr_space_api_key = "your-ocr-space-key"
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 💡 Usage

- Upload one or more files (PDF, DOCX, PNG, JPG, JPEG).
- The app will extract and summarize the content of images (not PDFs or DOCX). Summarization uses GPT-4o-mini for images only.
- Ask questions in the chat box about your files.
- View extracted text and summaries in expandable sections, with chunk indices.

---


## ❓ Troubleshooting

- **API Key errors:** Make sure your keys are correct and active.
- **OCR issues:** Some images may not be readable; try higher quality scans.
- **File size:** Very large files may take longer to process.

---


## 🧪 Testing

Automated unit tests are provided in `test_app.py` for core functions (file extraction and document summarization).

To run the tests:
```bash
python -m unittest test_app.py
```

---

## 🔒 Privacy

> All files and chat data are processed in-memory and never stored.

---

## 📄 License

MIT License

---


## 🙏 Thanks & About

This script is part of a larger project focused on AI-powered document understanding and conversational interfaces. Now with LangGraph for advanced conversational memory and flow, and a modular architecture for better code organization.

Thank you for checking out this repository and exploring its features!

Made with ❤️ by mohammad007kh.

