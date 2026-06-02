"""
metadata_extractor.py — NLP metadata extraction for RAG chunks.

Extracts entities, keywords, and noun phrases from chunk text.
Supports spaCy ('en_core_web_sm') and features a robust, zero-dependency 
Regex-based fallback system when spaCy is unavailable or fails.
"""
import re
from typing import Dict, List, Set

# Standard English Stopwords list for the keyword fallback
STOPWORDS: Set[str] = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', 
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'is', 'used',
    'using', 'use', 'are', 'was'
}


class MetadataExtractor:
    """
    Extracts semantic metadata (entities, keywords, noun phrases) from text.
    """

    def __init__(self, force_fallback: bool = False):
        self.nlp = None
        self.use_fallback = force_fallback

        if not force_fallback:
            try:
                import spacy
                # Try loading the standard spaCy model
                self.nlp = spacy.load("en_core_web_sm")
                print("      [OK] Loaded spaCy en_core_web_sm successfully.")
            except Exception as e:
                print(f"      [WARN] spaCy unavailable or failed to load model ({e}). Using Regex-based fallback.")
                self.use_fallback = True

    def _extract_via_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extracts metadata using loaded spaCy pipeline."""
        if not self.nlp:
            raise RuntimeError("spaCy model is not loaded.")
        
        doc = self.nlp(text)
        
        # 1. Extract Entities
        entities = []
        for ent in doc.ents:
            val = ent.text.strip()
            if len(val) > 2 and val not in entities:
                entities.append(val)
                
        # 2. Extract Keywords (nouns, proper nouns, adjectives; length > 2; non-stopwords)
        keywords = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop:
                val = token.text.strip().lower()
                # Clean punctuation from tokens
                val = re.sub(r'[^\w-]', '', val)
                if len(val) > 2 and val not in keywords and val not in STOPWORDS:
                    keywords.append(val)
                    
        # 3. Extract Noun Phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            val = chunk.text.strip()
            # Clean and filter noun phrases
            if len(val) > 2 and val not in noun_phrases:
                noun_phrases.append(val)
                
        return {
            "entities": entities,
            "keywords": keywords,
            "noun_phrases": noun_phrases
        }

    def _extract_via_regex(self, text: str) -> Dict[str, List[str]]:
        """
        Regex-based metadata extraction fallback.
        Completely self-contained, lightweight, and robust.
        """
        try:
            # 1. Extract Entities (Words starting with a capital letter, possibly hyphenated, grouped in sequences)
            # Matches proper nouns e.g. "Bellman-Ford", "Distance Vector Routing"
            entity_pattern = r'\b[A-Z][a-zA-Z0-9-]*(?:\s+[A-Z][a-zA-Z0-9-]*)*\b'
            raw_entities = re.findall(entity_pattern, text)
            
            entities = []
            for ent in raw_entities:
                cleaned = ent.strip()
                # Skip trivial single-letter capitals or common sentence starts
                if len(cleaned) > 2 and cleaned not in entities:
                    # Filter out purely generic words like 'The', 'A' at start of entities
                    if cleaned.startswith("The ") and len(cleaned) > 7:
                        cleaned = cleaned[4:]
                    entities.append(cleaned)

            # 2. Extract Keywords
            # Tokenize words, lowercase, filter punctuation and stopwords
            words = re.findall(r'\b[a-zA-Z-]{3,}\b', text)
            keywords = []
            for w in words:
                w_lower = w.lower()
                if w_lower not in STOPWORDS and w_lower not in keywords:
                    keywords.append(w_lower)
            
            # Keep top keywords to avoid cluttering
            keywords = keywords[:15]

            # 3. Extract Noun Phrases
            # A simplistic heuristic: Adjective(s) + Noun(s) sequence or Capitalized phrases
            phrase_pattern = r'\b(?:[a-zA-Z-]+(?:ing|ed|al|ic|ous|ble)?\s+)*(?:[a-zA-Z-]+(?:tion|ment|ness|ity|er|or|ist|ing|s)?\b)'
            raw_phrases = re.findall(phrase_pattern, text)
            
            noun_phrases = []
            for phrase in raw_phrases:
                cleaned = phrase.strip()
                words_in_phrase = cleaned.split()
                if len(words_in_phrase) >= 2 and len(cleaned) > 5:
                    if cleaned.lower() not in STOPWORDS and cleaned not in noun_phrases:
                        noun_phrases.append(cleaned)
            
            # Keep entities themselves as part of noun phrases if not there
            for ent in entities:
                if ent not in noun_phrases:
                    noun_phrases.append(ent)
                    
            return {
                "entities": entities,
                "keywords": keywords,
                "noun_phrases": noun_phrases[:12]
            }
        except Exception:
            # Absolute foolproof fallback
            return {
                "entities": [],
                "keywords": [],
                "noun_phrases": []
            }

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts entities, keywords, and noun phrases from the text.
        Degrades gracefully on any failure.
        """
        if not text or not isinstance(text, str):
            return {"entities": [], "keywords": [], "noun_phrases": []}
            
        if self.use_fallback or not self.nlp:
            return self._extract_via_regex(text)
            
        try:
            return self._extract_via_spacy(text)
        except Exception as e:
            # Fallback on runtime spaCy error
            print(f"Warning: spaCy runtime extraction failed ({e}). Falling back to Regex.")
            return self._extract_via_regex(text)


if __name__ == "__main__":
    # Test cases
    extractor = MetadataExtractor(force_fallback=True)
    sample_text = "Bellman-Ford Algorithm is used in Distance Vector Routing."
    meta = extractor.extract(sample_text)
    print("Testing MetadataExtractor Fallback...")
    print(f"Text: '{sample_text}'")
    print(f"Entities: {meta['entities']}")
    print(f"Keywords: {meta['keywords']}")
    print(f"Noun Phrases: {meta['noun_phrases']}")
