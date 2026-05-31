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

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict[str, Any], float]],
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Sends the prompt to the LLM and streams or returns the response.
        """
        context = self.build_context(retrieved_chunks)
        
        system_prompt = """You are a precise Q&A assistant. Answer ONLY from the provided Context.

STRICT RULES:
- Only use information from the Context. If not found, say "I cannot answer this based on the provided documents."
- Cite sources inline like [Source 1].
- NEVER repeat a sentence, phrase, or conclusion you have already written.
- NEVER write filler like "In conclusion", "Overall", "In summary" more than once.
- NEVER pad the answer. Each sentence must add NEW information.
- If the user asks for a specific length, cover MORE topics and details — never repeat points.
- If the user asks a follow-up question, use the conversation history to understand what they are referring to.
- Stop writing the moment you have no new information to add."""

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add last 4 messages from history (2 exchanges) for context
        # Skip the system message, only include user/assistant turns
        if chat_history:
            recent = chat_history[-4:]  # last 4 messages = 2 Q&A pairs
            for msg in recent:
                if msg["role"] in ("user", "assistant"):
                    # For assistant messages, strip sources metadata, just keep content
                    content = msg.get("content", "")
                    messages.append({"role": msg["role"], "content": content})

        # Add current question with context
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

if __name__ == "__main__":
    pass
