import os
from typing import List, Dict, Any, Tuple
from groq import Groq

class LLMGenerator:
    """
    LLMGenerator is responsible for taking the user's query and the retrieved context 
    and formulating a final answer.

    Now uses Groq (https://groq.com) — free tier, no credit card required.
    Groq runs open-source models (llama-3.3-70b-versatile) at very high speed.

    Setup:
        1. Sign up at https://console.groq.com
        2. Create a free API key
        3. Set the environment variable:
               export GROQ_API_KEY="gsk_..."
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        # Expects GROQ_API_KEY environment variable to be set
        self.client = Groq()
        
    def build_context(self, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Formats the retrieved chunks into a single string for the LLM prompt.
        Includes metadata so the LLM can optionally cite sources.
        """
        if not retrieved_chunks:
            return "No relevant context found."
            
        context_parts = []
        for i, (chunk, _) in enumerate(retrieved_chunks):
            source = chunk.get("source_filename", "Unknown")
            page = chunk.get("page_number", "Unknown")
            text = chunk.get("text", "")
            
            part = f"[Source {i+1}: {source}, Page {page}]\n{text}\n"
            context_parts.append(part)
            
        return "\n".join(context_parts)

    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Sends the prompt to the LLM and streams or returns the response.
        """
        context = self.build_context(retrieved_chunks)
        
        system_prompt = """You are an expert Q&A system. Your task is to answer the user's question based ONLY on the provided Context.
        
Rules:
1. If the answer is not contained in the Context, say "I cannot answer this based on the provided documents."
2. Do not use outside knowledge.
3. If possible, briefly cite the source number (e.g., [Source 1]) when stating a fact."""

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0 # Keep it deterministic for factual Q&A
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

if __name__ == "__main__":
    pass
