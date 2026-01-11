import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import docx
from PyPDF2 import PdfWriter
from src.file_processing import extract_text, ocr_space_text, describe_document

class TestAppFunctions(unittest.TestCase):
    def test_extract_text_pdf(self):
        # Create a simple PDF in memory
        pdf_writer = PdfWriter()
        pdf_writer.add_blank_page(width=72, height=72)
        pdf_bytes = BytesIO()
        pdf_writer.write(pdf_bytes)
        pdf_bytes.seek(0)
        pdf_bytes.name = 'test.pdf'
        text, is_ocr = extract_text(pdf_bytes)
        self.assertIsInstance(text, str)
        self.assertFalse(is_ocr)

    def test_extract_text_docx(self):
        # Create a simple DOCX in memory
        doc = docx.Document()
        doc.add_paragraph("Hello World")
        docx_bytes = BytesIO()
        doc.save(docx_bytes)
        docx_bytes.seek(0)
        docx_bytes.name = 'test.docx'
        text, is_ocr = extract_text(docx_bytes)
        self.assertIn("Hello World", text)
        self.assertFalse(is_ocr)

    @patch('src.file_processing.requests.post')
    def test_extract_text_image(self, mock_post):
        # Mock OCR.space response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"ParsedResults": [{"ParsedText": "Image text"}]}
        )
        img_bytes = BytesIO(b"fakeimage")
        img_bytes.name = 'test.jpg'
        text, is_ocr = extract_text(img_bytes)
        self.assertEqual(text, "Image text")
        self.assertTrue(is_ocr)

    @patch('src.file_processing.OpenAI')
    def test_describe_document(self, mock_openai):
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Summary text"))]
        )
        mock_openai.return_value = mock_client
        summary = describe_document("Some text", "file.txt")
        self.assertEqual(summary, "Summary text")

if __name__ == '__main__':
    unittest.main()
