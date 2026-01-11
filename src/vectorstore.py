from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
from .file_processing import extract_text, describe_document

def create_vectorstore(uploaded_files):
    extracted_texts = {}
    ocr_flags = {}

    for f in uploaded_files:
        text, is_ocr = extract_text(f)
        extracted_texts[f.name] = text
        ocr_flags[f.name] = is_ocr

    # Split text and add metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_texts, metadatas = [], []

    for filename, doc_text in extracted_texts.items():
        chunks = splitter.split_text(doc_text)
        summary = describe_document(doc_text, filename) if ocr_flags.get(filename) else ""
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < MIN_CHUNK_LENGTH:
                continue
            split_texts.append(chunk)
            metadatas.append({
                "source": filename,
                "summary": summary,
                "chunk": i
            })

    # Vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vs = FAISS.from_texts(split_texts, embeddings, metadatas=metadatas)

    return vs, extracted_texts, ocr_flags, split_texts, metadatas