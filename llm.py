import os
from typing import List, Dict, Any, Tuple
from groq import Groq

# Model fallback chain — ordered by quality
# Each model has its own separate daily quota on Groq free tier
MODELS = [
    "llama-3.3-70b-versatile",   # best quality, 100k tokens/day
    "llama-3.1-8b-instant",      # fast, 1M tokens/day — main fallback
    "gemma2-9b-it",              # Google model, separate quota
    "mixtral-8x7b-32768",        # Mixtral, separate quota
]

class LLMGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODELS[0]
        self.client = Groq()

    def build_context(
        self,
        retrieved_chunks: List[Tuple[Dict[str, Any], float]],
        document_summaries: List[str] = None
    ) -> str:
        if not retrieved_chunks:
            chunk_context = "No relevant context found."
        else:
            context_parts = []
            for i, (chunk, _) in enumerate(retrieved_chunks):
                source = chunk.get("source_filename", "Unknown")
                page = chunk.get("page_number", "Unknown")
                text = chunk.get("text", "")
                part = f"[Source {i+1}: {source}, Page {page}]\n{text}\n"
                context_parts.append(part)
            chunk_context = "\n".join(context_parts)

        # Prepend document summaries if available
        if document_summaries:
            summary_header = "=== DOCUMENT SUMMARIES ===\n"
            summary_content = "\n\n".join(document_summaries)
            summary_footer = "\n==========================\n\n"
            return f"{summary_header}{summary_content}{summary_footer}=== RETRIEVED EVIDENCE CHUNKS ===\n{chunk_context}"
            
        return chunk_context

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict[str, Any], float]],
        chat_history: List[Dict[str, str]] = None,
        document_summaries: List[str] = None
    ) -> str:
        # If document_summaries is not provided, attempt to dynamically load or generate them
        if document_summaries is None and retrieved_chunks:
            document_summaries = []
            seen_files = set()
            for chunk, _ in retrieved_chunks:
                source_file = chunk.get("source_filename")
                if source_file and source_file not in seen_files:
                    seen_files.add(source_file)
                    import re
                    doc_name = re.sub(r'[^a-zA-Z0-9]', '_', source_file.replace(".pdf", "").replace(".PDF", "")).lower()
                    meta_path = f"doc_store/{doc_name}/meta.json"
                    if os.path.exists(meta_path):
                        try:
                            import json
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            if "document_summary" in meta:
                                document_summaries.append(f"Document: {source_file}\nSummary: {meta['document_summary']}")
                            else:
                                # Dynamically generate summary for backward compatibility
                                chunks_path = f"doc_store/{doc_name}/chunks.json"
                                if os.path.exists(chunks_path):
                                    with open(chunks_path, "r", encoding="utf-8") as cf:
                                        file_chunks = json.load(cf)
                                    doc_text = "\n\n".join([c.get("text", "") for c in file_chunks])
                                    summary_text = self.summarize_document(doc_text)
                                    
                                    # Try to generate vector embedding using Embedder
                                    try:
                                        from embedder import Embedder
                                        emb = Embedder().generate_query_embedding(summary_text)
                                        meta["summary_embedding"] = emb.tolist()
                                    except Exception:
                                        pass
                                        
                                    meta["document_summary"] = summary_text
                                    with open(meta_path, "w", encoding="utf-8") as mf:
                                        json.dump(meta, mf)
                                    document_summaries.append(f"Document: {source_file}\nSummary: {summary_text}")
                        except Exception:
                            pass

        context = self.build_context(retrieved_chunks, document_summaries)

        system_prompt = """You are a precise Q&A assistant. Answer ONLY from the provided Context.

STRICT RULES:
- Only use information from the Context. If not found, say "I cannot answer this based on the provided documents."
- Cite sources inline using the source filename and page number like [filename.pdf, Page X].
- NEVER repeat a sentence, phrase, or conclusion you have already written.
- NEVER write filler like "In conclusion", "Overall", "In summary" more than once.
- NEVER pad the answer. Each sentence must add NEW information.
- If the user asks for a specific length, cover MORE topics and details — never repeat points.
- If the user asks a follow-up question, use the conversation history to understand what they are referring to.
- Stop writing the moment you have no new information to add.

FORMAT RULES:
- Always answer using markdown formatting.
- Start with a one-line definition as plain text.
- Then use markdown bullet points (start each point with "- ").
- Each bullet = one distinct fact with its source citation.
- If the user asks for code or an algorithm, present it in a clean code block (```).
  Use the pseudocode/algo notation from the source — do not convert to Python.
  If the OCR text looks garbled or corrupted, use your understanding of the algorithm
  to present a clean readable version in the same pseudocode style.
- Keep each bullet concise — one idea per bullet, max 2 lines.

After your answer, on a NEW LINE, output exactly one of these — nothing else on that line:
CONFIDENCE: HIGH
CONFIDENCE: MEDIUM
CONFIDENCE: LOW

Use HIGH if the context directly and fully answers the question.
Use MEDIUM if the context partially answers or required some inference.
Use LOW if the answer is not clearly in the context or you said you cannot answer."""

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] in ("user", "assistant"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg.get("content", "")
                    })

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        })

        # Try each model in fallback chain
        for model in MODELS:
            try:
                if self.client:
                    self.client.api_key = os.environ.get("GROQ_API_KEY") or self.client.api_key
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1500
                )
                from performance_tracker import PerformanceTracker
                PerformanceTracker().increment_llm_calls()
                self.model_name = model
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    continue  # try next model
                elif "decommissioned" in error_str.lower():
                    continue  # skip decommissioned models
                else:
                    return (
                        f"I cannot answer this based on the provided documents.\n"
                        f"CONFIDENCE: LOW\n"
                        f"Error: {error_str}"
                    )

    def summarize_document(self, doc_text: str) -> str:
        """Generates a concise summary of a document based on its initial text."""
        prompt = f"""You are an expert document summarizer. 
Write a highly concise summary (3-5 sentences) describing the core topics, goals, and content of this document.
Write directly as the summary, do not include intro like 'Here is a summary'.

Document Text:
{doc_text[:6000]}  # Keep it safe for token limits

Concise Summary:"""
        try:
            # We'll use llama-3.1-8b-instant for fast, cheap summarization
            if self.client:
                self.client.api_key = os.environ.get("GROQ_API_KEY") or self.client.api_key
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a professional technical document summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            from performance_tracker import PerformanceTracker
            PerformanceTracker().increment_llm_calls()
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Document summarization failed ({e}). Returning fallback summary.")
            return "A technical document covering networking, protocols, or algorithm design."

if __name__ == "__main__":
    pass