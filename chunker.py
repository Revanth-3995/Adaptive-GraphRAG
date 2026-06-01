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
        self._nlp = None
        self._spacy_failed = False
        
    def _extract_tables_from_page(self, page) -> List[str]:
        """
        Extracts tables from a PDF page and formats them as markdown tables.
        Uses PyMuPDF's built-in table detection.
        Falls back gracefully if no tables found.
        Returns a list of markdown table strings.
        """
        try:
            tables = page.find_tables()
            if not tables or not tables.tables:
                return []

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

            return markdown_tables
        except Exception:
            return []

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

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extracts metadata using spaCy if available, fallback to regex, or empty.
        Returns a dict with entities, keywords, noun_phrases and metadata_extractor.
        """
        meta = {
            "entities": [],
            "keywords": [],
            "noun_phrases": [],
            "metadata_extractor": "empty"
        }
        if not text.strip():
            return meta

        if not self._spacy_failed:
            try:
                import spacy
                if self._nlp is None:
                    self._nlp = spacy.load("en_core_web_sm")

                doc = self._nlp(text)

                entities = list(set([ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON", "GPE", "PRODUCT", "EVENT")]))
                noun_phrases = list(set([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]))
                keywords = list(set([token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ("NOUN", "PROPN", "VERB")]))

                meta["entities"] = entities
                meta["noun_phrases"] = noun_phrases
                meta["keywords"] = keywords[:20]  # limit to top keywords
                meta["metadata_extractor"] = "spacy"
                return meta

            except Exception:
                self._spacy_failed = True
                # Fallback to regex
                pass

        # Regex fallback
        try:
            # Capitalized terms (2 or more words) for entities/noun phrases
            cap_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
            meta["entities"] = list(set(cap_terms))

            # Simple keyword extraction (words > 5 chars, start with uppercase)
            keywords = re.findall(r'\b[A-Z][A-Za-z]{5,}\b', text)
            meta["keywords"] = list(set(keywords))

            meta["metadata_extractor"] = "regex"
            return meta

        except Exception:
            meta["metadata_extractor"] = "empty"
            return meta

    def _make_chunk(self, text: str, page_number: int, source_filename: str, chunk_type: str = "text", heading: str = "", section_path: List[str] = None) -> Dict[str, Any]:
        """Helper to create a chunk dict with metadata."""
        if section_path is None:
            section_path = []

        word_count = len(text.split())
        metadata = self._extract_metadata(text)

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": text,
            "source_filename": source_filename,
            "page_number": page_number,
            "heading": heading,
            "section_path": section_path,
            "word_count": word_count,
            "chunk_type": chunk_type,
            "entities": metadata["entities"],
            "keywords": metadata["keywords"],
            "noun_phrases": metadata["noun_phrases"],
            "metadata_extractor": metadata["metadata_extractor"]
        }

        return chunk

    def _is_heading(self, text: str) -> bool:
        """
        Detects if a given string looks like a heading.
        """
        text = text.strip()
        if not text:
            return False

        # ALL CAPS headings (allowing numbers and spaces)
        if text.isupper() and len(text) > 3 and len(text) < 100:
            return True

        # Numbered headings (e.g., "1. Introduction", "5.1 Distance Vector Routing")
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text):
            return True

        # Chapter/Section titles
        if text.lower().startswith(('chapter ', 'section ')):
            return True

        # Lines ending with ':'
        if text.endswith(':') and len(text.split()) < 10:
            return True

        return False

    def _split_into_chunks(self, text: str, page_number: int, source_filename: str) -> List[Dict[str, Any]]:
        """
        Semantic chunking — splits at natural boundaries instead of fixed word count.
        Priority order: heading boundaries -> section boundaries -> paragraph boundaries -> sentence boundaries -> word count limits
        """
        # Step 1: Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk_text = []
        current_word_count = 0
        current_heading = ""
        current_section_path = []

        def save_current_chunk():
            nonlocal current_chunk_text, current_word_count
            if current_word_count >= 3:
                chunk_text = ' '.join(current_chunk_text)
                chunks.append(self._make_chunk(
                    chunk_text,
                    page_number,
                    source_filename,
                    chunk_type="text",
                    heading=current_heading,
                    section_path=list(current_section_path)
                ))

            if self.overlap > 0 and current_word_count >= 3:
                # Retain overlap from end of current chunk
                words = ' '.join(current_chunk_text).split()
                overlap_words = words[-self.overlap:]
                current_chunk_text = [' '.join(overlap_words)]
                current_word_count = len(overlap_words)
            else:
                current_chunk_text = []
                current_word_count = 0

        for para in paragraphs:
            para_words = para.split()
            para_word_count = len(para_words)

            is_heading = self._is_heading(para)

            # If it's a heading, we likely want to start a new chunk
            if is_heading:
                if current_word_count > 0:
                    save_current_chunk()
                current_heading = para
                # Update section path: limit to last 3 levels to avoid endless growth
                current_section_path.append(para)
                if len(current_section_path) > 3:
                    current_section_path.pop(0)

            # If adding this paragraph exceeds limit, OR it's a heading (concept boundary)
            if current_word_count + para_word_count > self.chunk_size and current_word_count > 0:

                # We need to split further if a single paragraph is too long (sentence split)
                if para_word_count > self.chunk_size:
                    save_current_chunk()

                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        sentence_words = sentence.split()
                        sentence_word_count = len(sentence_words)

                        if current_word_count + sentence_word_count > self.chunk_size and current_word_count > 0:
                            save_current_chunk()

                        current_chunk_text.append(sentence)
                        current_word_count += sentence_word_count
                else:
                    save_current_chunk()
                    current_chunk_text.append(para)
                    current_word_count += para_word_count
            else:
                current_chunk_text.append(para)
                current_word_count += para_word_count

        # Save remaining
        if current_word_count >= 3:
            chunk_text = ' '.join(current_chunk_text)
            chunks.append(self._make_chunk(
                chunk_text,
                page_number,
                source_filename,
                chunk_type="text",
                heading=current_heading,
                section_path=list(current_section_path)
            ))

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
            table_texts = self._extract_tables_from_page(page)

            for md_table in table_texts:
                all_chunks.append(self._make_chunk(
                    md_table,
                    page_num,
                    filename,
                    chunk_type="table",
                    heading="",
                    section_path=[]
                ))

            if not raw_text.strip():
                # Page has no digital text — likely scanned or image-based
                # Fall back to OCR instead of skipping
                print(f"      → Page {page_num}/{total_pages}: no text layer, running OCR...")
                raw_text = _ocr_page(page)
                if raw_text.strip():
                    ocr_pages += 1

            if not raw_text.strip() and not table_texts:
                print(f"      → Page {page_num}/{total_pages}: skipped (blank or unreadable)")
                continue

            # Text processing
            if raw_text.strip():
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