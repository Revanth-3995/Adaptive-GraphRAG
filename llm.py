import os
from typing import List, Dict, Any, Tuple
from groq import Groq

# Model fallback chain — ordered by quality, falls back on rate limit
MODELS = [
    "llama-3.3-70b-versatile",   # best quality, 100k TPD free
    "llama-3.1-8b-instant",      # fast, 1M TPD free — main fallback
    "gemma2-9b-it",              # Google model, separate quota
    "mixtral-8x7b-32768",        # Mixtral, separate quota
]

class LLMGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODELS[0]
        self.client = Groq()

    def build_context(self, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
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
        context = self.build_context(retrieved_chunks)

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
- Always answer in bullet points (use • or -).
- Start with a one-line definition, then break the rest into bullets.
- Each bullet = one distinct fact or concept with its source citation.
- If the user asks for a summary, provide a concise bullet-point summary of the key points.
- If the user asks for algorithm, reproduce it EXACTLY as it appears 
  in the source document — do not convert to Python or any programming language.
-ANY algorithm, pseudocode, or code MUST be wrapped in a markdown 
  code block using triple backticks (```). This is mandatory — never 
  output an algorithm as a numbered list.
- If the user asks for code or an algorithm, use a code block instead of bullets.
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

        # Try each model in the fallback chain
        last_error = None
        for model in MODELS:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1500
                )
                # If successful, update current model for next call
                self.model_name = model
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    # Rate limited — try next model
                    last_error = f"⚠️ {model} rate limited, trying next..."
                    continue
                else:
                    # Different error (auth, network etc) — don't retry
                    return (
                        f"I cannot answer this based on the provided documents.\n"
                        f"CONFIDENCE: LOW\n"
                        f"Error: {error_str}"
                    )

        # All models exhausted
        return (
            "I cannot answer this based on the provided documents.\n"
            "CONFIDENCE: LOW\n"
            "⏳ All models are rate limited. Please wait ~30 minutes or "
            "add a fresh Groq API key in the sidebar."
        )

if __name__ == "__main__":
    pass