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
        
    def _clean_text(self, text: str) -> str:
        """
        Cleans the extracted text by removing excessive whitespaces and newlines.
        
        Why Clean Text?
        Raw PDF extraction often contains weird formatting, arbitrary line breaks, 
        and multiple spaces. These artifacts can negatively impact the quality of 
        the embeddings generated later.
        """
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _split_into_chunks(self, text: str, page_number: int, source_filename: str) -> List[Dict[str, Any]]:
        """
        Splits text into chunks of specified word count with overlap.
        
        Time Complexity: O(N) where N is the number of words.
        Space Complexity: O(N) to store the chunks in memory.
        """
        words = text.split()
        page_chunks = []
        
        # We step through the words array. The step size is (chunk_size - overlap)
        step_size = self.chunk_size - self.overlap
        if step_size <= 0:
            raise ValueError("Overlap must be strictly less than chunk_size")
            
        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
                
            chunk_text = ' '.join(chunk_words)
            # Create a unique ID for each chunk for future retrieval and deduplication
            chunk_id = str(uuid.uuid4())
            
            # Important: Check if chunk has substantial content (e.g., > 3 words)
            if len(chunk_words) < 3:
                continue

            chunk_metadata = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_filename": source_filename,
                "page_number": page_number
            }
            page_chunks.append(chunk_metadata)
            
        return page_chunks

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

            if not raw_text.strip():
                # Page has no digital text — likely scanned or image-based
                # Fall back to OCR instead of skipping
                print(f"      → Page {page_num}/{total_pages}: no text layer, running OCR...")
                raw_text = _ocr_page(page)
                if raw_text.strip():
                    ocr_pages += 1

            if not raw_text.strip():
                print(f"      → Page {page_num}/{total_pages}: skipped (blank or unreadable)")
                continue

            clean_text = self._clean_text(raw_text)
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