import os
import time
import json
import httpx
from typing import List, Dict, Any, Generator, Optional, Tuple
from embedder import Embedder

# Live monitor state for tracking provider health
PROVIDER_HEALTH = {
    "groq": {"success_count": 0, "failure_count": 0, "latencies": [], "last_error": None},
    "openai": {"success_count": 0, "failure_count": 0, "latencies": [], "last_error": None},
    "gemini": {"success_count": 0, "failure_count": 0, "latencies": [], "last_error": None},
    "claude": {"success_count": 0, "failure_count": 0, "latencies": [], "last_error": None},
}

def get_provider_health_stats() -> List[Dict[str, Any]]:
    stats = []
    for provider, data in PROVIDER_HEALTH.items():
        success = data["success_count"]
        fail = data["failure_count"]
        total = success + fail
        rate = (success / total * 100) if total > 0 else 100.0
        avg_lat = (sum(data["latencies"]) / len(data["latencies"])) if data["latencies"] else 0.0
        stats.append({
            "provider": provider,
            "success_rate": round(rate, 1),
            "avg_latency": round(avg_lat, 1),
            "failures": fail,
            "successes": success,
            "last_error": data["last_error"]
        })
    return stats

class BaseProvider:
    def __init__(self, provider_name: str, api_key: str):
        self.provider_name = provider_name
        self.api_key = api_key

    def record_success(self, latency: float):
        PROVIDER_HEALTH[self.provider_name]["success_count"] += 1
        PROVIDER_HEALTH[self.provider_name]["latencies"].append(latency)
        # Keep last 20 latencies
        PROVIDER_HEALTH[self.provider_name]["latencies"] = PROVIDER_HEALTH[self.provider_name]["latencies"][-20:]

    def record_failure(self, err_msg: str):
        PROVIDER_HEALTH[self.provider_name]["failure_count"] += 1
        PROVIDER_HEALTH[self.provider_name]["last_error"] = f"{datetime_now_str()}: {err_msg}"

def datetime_now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

PROVIDER_MODEL_FALLBACKS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "claude": ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"]
}

def get_candidate_models(provider: str, requested_model: Optional[str] = None) -> List[str]:
    fallbacks = PROVIDER_MODEL_FALLBACKS.get(provider, [])
    if not fallbacks:
        return [requested_model] if requested_model else []
    if not requested_model:
        return fallbacks
    if requested_model in fallbacks:
        idx = fallbacks.index(requested_model)
        return [requested_model] + fallbacks[:idx] + fallbacks[idx+1:]
    else:
        return [requested_model] + fallbacks

def is_auth_error(e: Exception) -> bool:
    err_str = str(e).lower()
    status_code = getattr(getattr(e, "response", None), "status_code", None)
    if status_code in [401, 403]:
        return True
    if "401" in err_str or "403" in err_str:
        return True
    if "unauthorized" in err_str or "invalid api key" in err_str or "invalid_api_key" in err_str:
        return True
    return False

# 1. Groq Provider
class GroqProvider(BaseProvider):
    def generate(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> str:
        t_start = time.perf_counter()
        model = model or "llama-3.3-70b-versatile"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in chat_history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1500),
            "stream": False
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                res = r.json()
                self.record_success(time.perf_counter() - t_start)
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def stream(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> Generator[str, None, None]:
        model = model or "llama-3.3-70b-versatile"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in chat_history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1500),
            "stream": True
        }

        t_start = time.perf_counter()
        try:
            with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            delta = json.loads(data_str)["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            pass
            self.record_success(time.perf_counter() - t_start)
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        # Groq currently does not host a general embedding model, fallback to local SentenceTransformers
        embedder = Embedder()
        embeddings = embedder.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

# 2. OpenAI Provider
class OpenAIProvider(BaseProvider):
    def generate(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> str:
        t_start = time.perf_counter()
        model = model or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in chat_history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1500),
            "stream": False
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                res = r.json()
                self.record_success(time.perf_counter() - t_start)
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def stream(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> Generator[str, None, None]:
        model = model or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in chat_history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1500),
            "stream": True
        }

        t_start = time.perf_counter()
        try:
            with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            delta = json.loads(data_str)["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            pass
            self.record_success(time.perf_counter() - t_start)
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        t_start = time.perf_counter()
        model = model or "text-embedding-3-small"
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"input": texts, "model": model}
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                res = r.json()
                self.record_success(time.perf_counter() - t_start)
                return [d["embedding"] for d in res["data"]]
        except Exception as e:
            self.record_failure(str(e))
            # Fallback to local SentenceTransformers
            embedder = Embedder()
            embeddings = embedder.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

# 3. Gemini Provider
class GeminiProvider(BaseProvider):
    def generate(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> str:
        t_start = time.perf_counter()
        model = model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        for msg in chat_history or []:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.1),
                "maxOutputTokens": kwargs.get("max_tokens", 1500)
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                res = r.json()
                self.record_success(time.perf_counter() - t_start)
                return res["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def stream(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> Generator[str, None, None]:
        model = model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        for msg in chat_history or []:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.1),
                "maxOutputTokens": kwargs.get("max_tokens", 1500)
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        t_start = time.perf_counter()
        try:
            with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        try:
                            chunk_data = json.loads(data_str)
                            text_piece = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                            yield text_piece
                        except Exception:
                            pass
            self.record_success(time.perf_counter() - t_start)
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        # For multiple texts, we embed one-by-one (keeping it simple and error-proof)
        model = model or "text-embedding-004"
        headers = {"Content-Type": "application/json"}
        embeddings = []
        t_start = time.perf_counter()
        try:
            with httpx.Client(timeout=15.0) as client:
                for text in texts:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={self.api_key}"
                    payload = {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text}]}
                    }
                    r = client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    embeddings.append(r.json()["embedding"]["values"])
            self.record_success(time.perf_counter() - t_start)
            return embeddings
        except Exception as e:
            self.record_failure(str(e))
            # Fallback to local
            embedder = Embedder()
            local_embs = embedder.model.encode(texts, convert_to_numpy=True)
            return local_embs.tolist()

# 4. Claude Provider (Anthropic)
class ClaudeProvider(BaseProvider):
    def generate(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> str:
        t_start = time.perf_counter()
        model = model or "claude-3-5-sonnet-latest"
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        messages = []
        for msg in chat_history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1500),
            "temperature": kwargs.get("temperature", 0.1),
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                res = r.json()
                self.record_success(time.perf_counter() - t_start)
                return res["content"][0]["text"]
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def stream(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, model: str = None, **kwargs) -> Generator[str, None, None]:
        model = model or "claude-3-5-sonnet-latest"
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        messages = []
        for msg in chat_history or []:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1500),
            "temperature": kwargs.get("temperature", 0.1),
            "stream": True
        }
        if system_prompt:
            payload["system"] = system_prompt

        t_start = time.perf_counter()
        try:
            with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "content_block_delta":
                                delta_text = event.get("delta", {}).get("text", "")
                                yield delta_text
                        except Exception:
                            pass
            self.record_success(time.perf_counter() - t_start)
        except Exception as e:
            self.record_failure(str(e))
            raise e

    def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        # Anthropic has no native embedding API, fallback to local SentenceTransformers
        embedder = Embedder()
        embeddings = embedder.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

# 5. Multi-Provider Router Manager
class MultiProviderManager:
    def __init__(self, db_keys: List[Dict[str, Any]]):
        """
        db_keys is a list of decrypted API keys from the DB.
        Structure: [{"provider": "groq", "decrypted_key": "gsk_...", "is_active": 1}]
        """
        self.keys = db_keys

    def _get_provider_instances(self, target_provider: str) -> List[BaseProvider]:
        """Creates provider instances for all active keys matching the target provider."""
        active_keys = [k for k in self.keys if k["provider"] == target_provider and k.get("is_active", 1) == 1]
        providers = []
        for key in active_keys:
            key_val = key["decrypted_key"]
            if target_provider == "groq":
                providers.append(GroqProvider(target_provider, key_val))
            elif target_provider == "openai":
                providers.append(OpenAIProvider(target_provider, key_val))
            elif target_provider == "gemini":
                providers.append(GeminiProvider(target_provider, key_val))
            elif target_provider == "claude":
                providers.append(ClaudeProvider(target_provider, key_val))
        return providers

    def _get_fallback_providers(self, target_provider: str) -> List[Tuple[str, BaseProvider]]:
        """Finds any fallback providers (ordered: gemini -> openai -> claude -> groq) if primary provider runs out of keys."""
        order = ["gemini", "openai", "claude", "groq"]
        fallbacks = []
        for p in order:
            if p == target_provider:
                continue
            p_instances = self._get_provider_instances(p)
            for inst in p_instances:
                fallbacks.append((p, inst))
        return fallbacks

    def generate(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, target_provider: str = "groq", model: str = None, **kwargs) -> Tuple[str, str, str]:
        """
        Routes the generate request.
        Returns Tuple[response_content, final_provider, final_model].
        """
        # Primary providers list
        providers = self._get_provider_instances(target_provider)
        
        # 1. Try primary provider keys
        for i, prov in enumerate(providers):
            candidate_models = get_candidate_models(target_provider, model)
            for m in candidate_models:
                try:
                    res = prov.generate(prompt, system_prompt, chat_history, model=m, **kwargs)
                    return res, target_provider, m
                except Exception as e:
                    if is_auth_error(e):
                        print(f"Warning: Primary Provider {target_provider} key {i} auth failure. Skipping to next key. Error: {e}")
                        break  # skip this key entirely
                    print(f"Warning: Primary Provider {target_provider} key {i} model {m} failed. Trying next model. Error: {e}")
                    continue

        # 2. Try Fallbacks
        print(f"Warning: All primary provider keys failed for {target_provider}. Initiating dynamic fallback chain...")
        fallbacks = self._get_fallback_providers(target_provider)
        for f_provider, f_inst in fallbacks:
            candidate_models = get_candidate_models(f_provider, None)
            for m in candidate_models:
                try:
                    res = f_inst.generate(prompt, system_prompt, chat_history, model=m, **kwargs)
                    return res, f_provider, m
                except Exception as e:
                    if is_auth_error(e):
                        print(f"Warning: Fallback Provider {f_provider} auth failure. Skipping. Error: {e}")
                        break
                    print(f"Warning: Fallback Provider {f_provider} model {m} failed. Trying next. Error: {e}")
                    continue

        # 3. Final Fallback - check environment variables directly (in case DB keys failed but .env holds a key)
        env_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "claude": os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        }
        for provider, key in env_keys.items():
            if not key:
                continue
            candidate_models = get_candidate_models(provider, None)
            for m in candidate_models:
                try:
                    if provider == "groq":
                        inst = GroqProvider(provider, key)
                    elif provider == "openai":
                        inst = OpenAIProvider(provider, key)
                    elif provider == "gemini":
                        inst = GeminiProvider(provider, key)
                    else:
                        inst = ClaudeProvider(provider, key)
                        
                    res = inst.generate(prompt, system_prompt, chat_history, model=m, **kwargs)
                    return res, provider, m
                except Exception as e:
                    if is_auth_error(e):
                        print(f"Warning: Env Provider {provider} auth failure. Skipping. Error: {e}")
                        break
                    print(f"Warning: Env Provider {provider} model {m} failed. Error: {e}")
                    continue

        raise RuntimeError("Failover error: All primary keys, fallback keys, and environment keys failed to execute.")

    def stream(self, prompt: str, system_prompt: str = None, chat_history: List[Dict[str, str]] = None, target_provider: str = "groq", model: str = None, **kwargs) -> Generator[Tuple[str, str, str], None, None]:
        """
        Streams response, falling back on connection failures.
        Yields Tuple[chunk_text, provider_name, model_name].
        """
        providers = self._get_provider_instances(target_provider)
        
        # Try primary
        for i, prov in enumerate(providers):
            candidate_models = get_candidate_models(target_provider, model)
            for m in candidate_models:
                try:
                    gen = prov.stream(prompt, system_prompt, chat_history, model=m, **kwargs)
                    try:
                        first_chunk = next(gen)
                    except StopIteration:
                        return
                    yield first_chunk, target_provider, m
                    for chunk in gen:
                        yield chunk, target_provider, m
                    return
                except Exception as e:
                    if is_auth_error(e):
                        print(f"Warning: Streaming key {i} for {target_provider} auth failure. Skipping to next key. Error: {e}")
                        break
                    print(f"Warning: Streaming key {i} model {m} for {target_provider} failed. Error: {e}")
                    continue

        # Try Fallbacks
        print(f"Warning: Streaming fallback chain initiated...")
        fallbacks = self._get_fallback_providers(target_provider)
        for f_provider, f_inst in fallbacks:
            candidate_models = get_candidate_models(f_provider, None)
            for m in candidate_models:
                try:
                    gen = f_inst.stream(prompt, system_prompt, chat_history, model=m, **kwargs)
                    try:
                        first_chunk = next(gen)
                    except StopIteration:
                        return
                    yield first_chunk, f_provider, m
                    for chunk in gen:
                        yield chunk, f_provider, m
                    return
                except Exception as e:
                    if is_auth_error(e):
                        print(f"Warning: Streaming fallback {f_provider} auth failure. Skipping. Error: {e}")
                        break
                    print(f"Warning: Streaming fallback {f_provider} model {m} failed. Error: {e}")
                    continue

        # Final Fallback directly to env keys
        env_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "claude": os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        }
        for provider, key in env_keys.items():
            if not key:
                continue
            candidate_models = get_candidate_models(provider, None)
            for m in candidate_models:
                try:
                    if provider == "groq":
                        inst = GroqProvider(provider, key)
                    elif provider == "openai":
                        inst = OpenAIProvider(provider, key)
                    elif provider == "gemini":
                        inst = GeminiProvider(provider, key)
                    else:
                        inst = ClaudeProvider(provider, key)
                        
                    gen = inst.stream(prompt, system_prompt, chat_history, model=m, **kwargs)
                    try:
                        first_chunk = next(gen)
                    except StopIteration:
                        return
                    yield first_chunk, provider, m
                    for chunk in gen:
                        yield chunk, provider, m
                    return
                except Exception as e:
                    if is_auth_error(e):
                        print(f"Warning: Streaming env provider {provider} auth failure. Skipping. Error: {e}")
                        break
                    print(f"Warning: Streaming env provider {provider} model {m} failed. Error: {e}")
                    continue

        raise RuntimeError("Failover streaming error: All connections failed.")

    def embed(self, texts: List[str], target_provider: str = "groq", model: str = None) -> List[List[float]]:
        providers = self._get_provider_instances(target_provider)
        for prov in providers:
            try:
                cur_model = model or self.get_default_embedding_model(target_provider)
                return prov.embed(texts, model=cur_model)
            except Exception:
                continue

        # Try environment fallback or local sentence-transformer
        env_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "claude": os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        }
        key = env_keys.get(target_provider)
        if key:
            try:
                if target_provider == "openai":
                    return OpenAIProvider(target_provider, key).embed(texts, model)
                elif target_provider == "gemini":
                    return GeminiProvider(target_provider, key).embed(texts, model)
            except Exception:
                pass

        # absolute fallback
        embedder = Embedder()
        embeddings = embedder.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def get_default_model(provider: str) -> str:
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-1.5-flash",
            "claude": "claude-3-5-sonnet-latest"
        }
        return defaults.get(provider, "llama-3.3-70b-versatile")

    @staticmethod
    def get_default_embedding_model(provider: str) -> str:
        defaults = {
            "openai": "text-embedding-3-small",
            "gemini": "text-embedding-004",
        }
        return defaults.get(provider, "all-MiniLM-L6-v2")
