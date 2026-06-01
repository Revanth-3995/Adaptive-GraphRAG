import fitz  # PyMuPDF
import re
import json
import uuid
import os
from dotenv import load_dotenv
load_dotenv()
tesseract_path = os.getenv("TESSERACT_PATH")
if tesseract_path:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
from typing import List, Dict, Any

# OCR support — only imported if needed
def _ocr_page(page) -> str:
    """
    Fallback: render the page as an image and run Tesseract OCR on it.
    Only called when pymupdf returns no text (i.e. the page is image-based).
    
    Requires: pip install pytesseract pdf2image
    System:   sudo apt install tesseract-ocr poppler-utils  (Linux)
              brew install tesseract poppler              (Mac)
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page at 2x resolution for better OCR accuracy
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except ImportError:
        print("      ⚠ OCR skipped — install pytesseract & Pillow for image-based PDF support:")
        print("        pip install pytesseract Pillow")
        print("        sudo apt install tesseract-ocr   (Linux)")
        print("        brew install tesseract           (Mac)")
        return ""
    except Exception as e:
        print(f"      ⚠ OCR failed on page: {e}")
        return ""

class DocumentChunker:
    """
    DocumentChunker is responsible for extracting text from PDFs, cleaning it, 
    and splitting it into overlapping chunks. This is the foundation of our RAG pipeline.
    
    Why Chunking?
    Large Language Models (LLMs) and embedding models have sequence length limits (e.g., 512 tokens).
    If we feed an entire document, it exceeds this limit. By chunking, we create focused, 
    semantically coherent text pieces that can be individually embedded and retrieved.
    
    Why Overlap?
    If we chunk without overlap, we might split a sentence or a core concept in half, 
    causing a loss of semantic context. An overlap ensures that boundary information 
    is preserved across chunks, maintaining context continuity.
    """
    
    def __init__(self, chunk_size_words: int = 200, overlap_words: int = 50):
        self.chunk_size = chunk_size_words
        self.overlap = overlap_words
        self.chunks = []
        
    def _extract_tables_from_page(self, page) -> str:
        """
        Extracts tables from a PDF page and formats them as markdown tables.
        Uses PyMuPDF's built-in table detection.
        Falls back gracefully if no tables found.
        """
        try:
            tables = page.find_tables()
            if not tables or not tables.tables:
                return ""

            markdown_tables = []
            for table in tables.tables:
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue

                # Build markdown table
                md_rows = []
                header = rows[0]
                # Clean None values
                header = [str(cell).strip() if cell else "" for cell in header]
                md_rows.append("| " + " | ".join(header) + " |")
                md_rows.append("|" + "|".join(["---"] * len(header)) + "|")

                for row in rows[1:]:
                    row = [str(cell).strip() if cell else "" for cell in row]
                    md_rows.append("| " + " | ".join(row) + " |")

                markdown_tables.append('\n'.join(md_rows))

            return '\n\n'.join(markdown_tables)
        except Exception:
            return ""

    def _clean_text(self, text: str) -> str:
        """
        Cleans text but PRESERVES paragraph breaks (double newlines) for semantic chunking.
        Single newlines within a paragraph are collapsed to spaces.
        Double newlines (paragraph boundaries) are preserved.
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Preserve paragraph breaks
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for para in paragraphs:
            # Within each paragraph, collapse single newlines to spaces
            para = re.sub(r'\n', ' ', para)
            para = re.sub(r'\s+', ' ', para)
            para = para.strip()
            if para:
                cleaned_paragraphs.append(para)
        return '\n\n'.join(cleaned_paragraphs)

    def _make_chunk(self, text: str, page_number: int, source_filename: str) -> Dict[str, Any]:
        """Helper to create a chunk dict with metadata."""
        return {
            "chunk_id": str(uuid.uuid4()),
            "text": text,
            "source_filename": source_filename,
            "page_number": page_number
        }

    def _split_into_chunks(self, text: str, page_number: int, source_filename: str) -> List[Dict[str, Any]]:
        """
        Semantic chunking — splits at natural boundaries instead of fixed word count.
        Priority order: paragraph breaks → heading breaks → sentence breaks → word count
        """
        # Step 1: Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk_words = []
        current_word_count = 0

        for para in paragraphs:
            para_words = para.split()
            para_word_count = len(para_words)

            # If adding this paragraph would exceed chunk_size
            if current_word_count + para_word_count > self.chunk_size and current_chunk_words:
                # Save current chunk with overlap
                chunk_text = ' '.join(current_chunk_words)
                if len(current_chunk_words) >= 3:
                    chunks.append(self._make_chunk(chunk_text, page_number, source_filename))

                # Start new chunk with overlap from end of previous
                overlap_words = current_chunk_words[-self.overlap:] if self.overlap > 0 else []
                current_chunk_words = overlap_words + para_words
                current_word_count = len(current_chunk_words)
            else:
                current_chunk_words.extend(para_words)
                current_word_count += para_word_count

        # Don't forget the last chunk
        if len(current_chunk_words) >= 3:
            chunk_text = ' '.join(current_chunk_words)
            chunks.append(self._make_chunk(chunk_text, page_number, source_filename))

        return chunks

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF, extracts page-wise text, cleans it, and splits it into chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        filename = os.path.basename(file_path)
        
        # PyMuPDF is fast and reliable for text extraction
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {file_path}: {e}")
            
        total_pages = len(doc)
        ocr_pages = 0
        all_chunks = []
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")

            # Extract tables separately
            table_text = self._extract_tables_from_page(page)

            if not raw_text.strip():
                # Page has no digital text — likely scanned or image-based
                # Fall back to OCR instead of skipping
                print(f"      → Page {page_num}/{total_pages}: no text layer, running OCR...")
                raw_text = _ocr_page(page)
                if raw_text.strip():
                    ocr_pages += 1

            if not raw_text.strip() and not table_text:
                print(f"      → Page {page_num}/{total_pages}: skipped (blank or unreadable)")
                continue

            # Combine regular text with table text
            combined_text = raw_text
            if table_text:
                combined_text = raw_text + "\n\n" + table_text

            clean_text = self._clean_text(combined_text)
            page_chunks = self._split_into_chunks(clean_text, page_num, filename)
            all_chunks.extend(page_chunks)

        if ocr_pages > 0:
            print(f"      ✓ OCR extracted text from {ocr_pages} image-based page(s)")
            
        self.chunks.extend(all_chunks)
        return all_chunks

    def save_chunks(self, output_path: str = "graph/chunk_store.json"):
        """
        Saves the processed chunks to a JSON file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully saved {len(self.chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str = "graph/chunk_store.json") -> List[Dict[str, Any]]:
        """
        Loads chunks from a JSON file.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Chunk store not found: {input_path}")
            
        with open(input_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
        return self.chunks

if __name__ == "__main__":
    # Example usage for testing
    # chunker = DocumentChunker()
    # chunker.process_pdf("sample.pdf")
    # chunker.save_chunks()
    pass