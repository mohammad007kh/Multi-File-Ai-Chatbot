# 🧠 Multi-File Chatbot with Memory

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that understands and responds to questions based only on the **uploaded files**. It's built using **LangChain**, **OpenAI embeddings**, and **FAISS**, and supports multiple formats like PDFs, Word documents, and images.

---

## 🔍 Features

| Feature | Description |
|--------|-------------|
| 📂 Multi-File Upload | Supports `.pdf`, `.docx`, `.png`, `.jpg` |
| 🧠 Conversational Memory | Remembers previous messages |
| 🔍 RAG System | Retrieves relevant chunks before generating answers |
| 🖼️ OCR Support | Extracts text from image files using Tesseract |
| 🔐 Safe & Private | All files and chats are **not stored** or analyzed |
| 💬 Honest AI | Replies with “I don’t know” if an answer isn't found |

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI Embeddings
- FAISS
- PyPDF2, python-docx, pytesseract

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file with your OpenAI key:

```toml
openai_api_key = "sk-..."
```

Then run:

```bash
streamlit run app.py
```

## 📢 Disclaimer

> ✅ **All uploaded files and conversations are kept private and never stored.**

🙏 Wish me luck on my AI career journey!

