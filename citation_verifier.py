# NOTE: This module contains Experimental / Advanced Features that are currently disabled from the standard user-facing product experience.
import re
import os
import json
from typing import List, Dict, Any, Tuple

class CitationVerifier:
    """
    CitationVerifier parses inline citations (e.g. [Source 1], [filename.pdf, Page X])
    and verifies if they correspond to actual retrieved evidence.
    """

    def __init__(self):
        # Match patterns like [Source X], [Source X: filename.pdf, Page Y], or [filename.pdf, Page Y]
        self.citation_pattern = re.compile(r'\[([^\]]+)\]')

    def is_likely_citation(self, content: str, chunks: List[Dict[str, Any]]) -> bool:
        """Determines if the bracket content is likely a citation or array/math index."""
        content_clean = content.strip()
        # 1. Contains "Source" followed by a number
        if re.search(r'Source\s+\d+', content_clean, re.IGNORECASE):
            return True
        # 2. Contains a file extension like .pdf or .txt
        if re.search(r'\.(pdf|PDF|txt|TXT|docx|DOCX|xlsx|XLSX|csv|CSV|json|JSON|pkl|PKL)', content_clean):
            return True
        # 3. Contains "Page" followed by a number
        if re.search(r'Page\s*\d+', content_clean, re.IGNORECASE):
            return True
        # 4. Matches any of the active chunk source filenames
        content_lower = content_clean.lower()
        for chunk in chunks:
            filename = chunk.get("source_filename", "")
            if filename:
                filename_lower = filename.lower()
                if filename_lower in content_lower or content_lower in filename_lower:
                    if len(content_lower) > 3:
                        return True
        return False

    def parse_citations(self, text: str) -> List[str]:
        """Extracts all inline bracketed citations from the text."""
        return self.citation_pattern.findall(text)

    def verify_citations(
        self,
        answer: str,
        retrieved_chunks: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """
        Verifies all citation brackets in the answer against retrieved chunks.
        
        Args:
            answer: Raw text response from LLM
            retrieved_chunks: List of (chunk_dict, score) retrieved from database
            
        Returns:
            Dict containing:
                "verified_answer": The answer text with verification marks (✓ / ⚠️) added inline
                "citations": List of citation audit dicts
                "verified_count": Number of successfully verified citations
                "failed_count": Number of unverified citations
        """
        citations = []
        verified_count = 0
        failed_count = 0
        
        # Flatten the list of tuples to just chunks
        chunks = [c for c, _ in retrieved_chunks]
        
        def check_citation(match):
            nonlocal verified_count, failed_count
            full_content = match.group(1).strip()
            full_bracket = match.group(0)
            
            # Skip if this is a math/array index rather than a source citation
            if not self.is_likely_citation(full_content, chunks):
                return full_bracket
            
            is_verified = False
            ref_details = {}
            
            try:
                # 1. Try parsing index-based pattern: "Source X"
                idx_match = re.search(r'Source\s+(\d+)', full_content, re.IGNORECASE)
                if idx_match:
                    src_idx = int(idx_match.group(1)) - 1  # 0-indexed
                    if 0 <= src_idx < len(chunks):
                        target_chunk = chunks[src_idx]
                        ref_details = {
                            "type": "index",
                            "index": src_idx + 1,
                            "filename": target_chunk.get("source_filename", ""),
                            "page": target_chunk.get("page_number", 0)
                        }
                        
                        # Extra validation: if filename or page is specified, cross-reference it
                        file_match = re.search(r'Source\s+\d+:\s*([^,\]]+)', full_content, re.IGNORECASE)
                        page_match = re.search(r'Page\s*(\d+)', full_content, re.IGNORECASE)
                        
                        file_ok = True
                        page_ok = True
                        
                        if file_match:
                            claimed_file = file_match.group(1).strip().lower()
                            actual_file = target_chunk.get("source_filename", "").lower()
                            if claimed_file not in actual_file:
                                file_ok = False
                                
                        if page_match:
                            claimed_page = int(page_match.group(1))
                            actual_page = int(target_chunk.get("page_number", -999))
                            if claimed_page != actual_page:
                                page_ok = False
                                
                        if file_ok and page_ok:
                            is_verified = True
                
                # 2. Try parsing file/page-based pattern directly: "filename.pdf, Page X"
                if not is_verified:
                    parts = [p.strip() for p in full_content.split(',')]
                    if len(parts) >= 2:
                        claimed_file = parts[0].lower()
                        page_part = parts[1]
                        page_match = re.search(r'Page\s*(\d+)', page_part, re.IGNORECASE)
                        
                        if page_match:
                            claimed_page = int(page_match.group(1))
                            
                            # Search in all chunks
                            for idx, chunk in enumerate(chunks):
                                actual_file = chunk.get("source_filename", "").lower()
                                actual_page = int(chunk.get("page_number", -999))
                                
                                # If filename matches and page matches
                                if (claimed_file in actual_file or actual_file in claimed_file) and claimed_page == actual_page:
                                    is_verified = True
                                    ref_details = {
                                        "type": "file_page",
                                        "index": idx + 1,
                                        "filename": chunk.get("source_filename", ""),
                                        "page": actual_page
                                    }
                                    break
                
                # 3. Last fallback: Check if filename is mentioned and matches any retrieved chunk
                # ONLY if the citation does not explicitly specify a page number
                if not is_verified and "page" not in full_content.lower():
                    for idx, chunk in enumerate(chunks):
                        actual_file = chunk.get("source_filename", "").lower()
                        if actual_file and actual_file in full_content.lower():
                            is_verified = True
                            ref_details = {
                                "type": "file_match",
                                "index": idx + 1,
                                "filename": chunk.get("source_filename", ""),
                                "page": chunk.get("page_number", 0)
                            }
                            break
                            
            except Exception as e:
                # If anything fails during parsing, treat as unverified
                pass
                
            if is_verified:
                verified_count += 1
                citations.append({
                    "citation": full_bracket,
                    "verified": True,
                    "details": ref_details
                })
                # Use ASCII safe marker, will be rendered beautifully in Streamlit
                return f"{full_bracket} [Verified]"
            else:
                failed_count += 1
                citations.append({
                    "citation": full_bracket,
                    "verified": False,
                    "details": ref_details
                })
                return f"{full_bracket} [Unverified]"

        # Substitute each pattern with its verified or unverified counterpart
        verified_answer = self.citation_pattern.sub(check_citation, answer)
        
        return {
            "verified_answer": verified_answer,
            "citations": citations,
            "verified_count": verified_count,
            "failed_count": failed_count
        }

if __name__ == "__main__":
    # Quick self-test
    verifier = CitationVerifier()
    test_answer = "Bellman-Ford was introduced by Richard Bellman [test.pdf, Page 3] and Lester Ford [Source 2]."
    test_chunks = [
        ({"source_filename": "dijkstra.pdf", "page_number": 1, "text": "Dijkstra algorithm..."}, 0.9),
        ({"source_filename": "test.pdf", "page_number": 3, "text": "Bellman-Ford algorithm details..."}, 0.8),
    ]
    res = verifier.verify_citations(test_answer, test_chunks)
    print("Test Answer:", test_answer)
    print("Verified Answer:", res["verified_answer"])
    print("Citations Report:", json.dumps(res["citations"], indent=2))
