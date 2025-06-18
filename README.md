# Multi-File AI Chatbot with Memory and Document Summarization

A Streamlit app that lets you chat with the content of your uploaded files (PDF, DOCX, images) using AI, with memory and document summaries.

---

## 🚀 Features

- **Multi-file upload:** Supports PDF, DOCX, PNG, JPG, JPEG.
- **Conversational memory:** Remembers previous chat turns for context.
- **Retrieval-Augmented Generation (RAG):** Answers are based only on your uploaded files.
- **OCR for images:** Extracts text from images using the OCR.space API.
- **Document summarization:** Uses GPT-4 to summarize and categorize each uploaded file.
- **Privacy:** No files or chat data are stored.

---

## 🛠️ Tech Stack

- Python, Streamlit
- LangChain, OpenAI, FAISS
- PyPDF2, python-docx, Pillow, requests
- OCR.space API

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
- The app will extract and summarize the content.
- Ask questions in the chat box about your files.
- View extracted text and summaries in expandable sections.

---

## ❓ Troubleshooting

- **API Key errors:** Make sure your keys are correct and active.
- **OCR issues:** Some images may not be readable; try higher quality scans.
- **File size:** Very large files may take longer to process.

---

## 🔒 Privacy

> All files and chat data are processed in-memory and never stored.

---

## 📄 License

MIT License

