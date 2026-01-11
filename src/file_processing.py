import requests
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
from PIL import Image
from .config import OPENAI_API_KEY, OCR_SPACE_API_KEY

def ocr_space_text(img_bytes):
    try:
        payload = {"apikey": OCR_SPACE_API_KEY, "language": "eng"}
        files = {"file": img_bytes}
        r = requests.post("https://api.ocr.space/parse/image", files=files, data=payload, timeout=60)
        r.raise_for_status()
        result = r.json()
        return "\n".join(item["ParsedText"].strip() for item in result.get("ParsedResults", []))
    except requests.exceptions.RequestException as e:
        st.error(f"OCR request failed: {e}")
        return ""

def extract_text(f):
    if f.name.lower().endswith(".pdf"):
        reader = PdfReader(f)
        return "\n".join(p.extract_text() or "" for p in reader.pages), False
    if f.name.lower().endswith(".docx"):
        doc = docx.Document(f)
        return "\n".join(p.text for p in doc.paragraphs), False
    if f.name.lower().endswith((".png", "jpg", "jpeg")):
        return ocr_space_text(f), True
    return "", False

def describe_document(text, filename):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"Describe the content of the following document named '{filename}'. Give a short summary and possible categories:\n\n{text[:1500]}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Failed to describe document: {e}")
        return ""