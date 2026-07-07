import os
import sqlite3
import json
import uuid
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Attempt to import cryptography, fallback to base64 obfuscation if unavailable
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    import base64

# Ensure encryption secret key is set
def get_encryption_key() -> str:
    env_path = ".env"
    secret = os.getenv("ENCRYPTION_SECRET")
    if secret:
        return secret
        
    # Auto-generate if missing
    if HAS_CRYPTOGRAPHY:
        new_secret = Fernet.generate_key().decode()
    else:
        new_secret = "fallback-insecure-graphrag-secret-key-32bytes="
        
    # Write to .env
    if os.path.exists(env_path):
        try:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\nENCRYPTION_SECRET={new_secret}\n")
        except Exception:
            pass
    else:
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"ENCRYPTION_SECRET={new_secret}\n")
        except Exception:
            pass
            
    os.environ["ENCRYPTION_SECRET"] = new_secret
    return new_secret

# Encryption / Decryption helpers
class EncryptionHelper:
    def __init__(self):
        self.secret = get_encryption_key()
        if HAS_CRYPTOGRAPHY:
            try:
                self.cipher = Fernet(self.secret.encode())
            except Exception:
                # Key might be malformed, regenerate
                fallback_secret = Fernet.generate_key().decode()
                self.cipher = Fernet(fallback_secret.encode())
        else:
            self.cipher = None

    def encrypt(self, plain_text: str) -> str:
        if not plain_text:
            return ""
        if HAS_CRYPTOGRAPHY and self.cipher:
            return self.cipher.encrypt(plain_text.encode()).decode()
        # Fallback obfuscation
        return base64.b64encode(plain_text.encode()).decode()

    def decrypt(self, cipher_text: str) -> str:
        if not cipher_text:
            return ""
        if HAS_CRYPTOGRAPHY and self.cipher:
            try:
                return self.cipher.decrypt(cipher_text.encode()).decode()
            except Exception:
                return ""
        # Fallback obfuscation
        try:
            return base64.b64decode(cipher_text.encode()).decode()
        except Exception:
            return ""

# Database Abstraction Interface
class DatabaseInterface(ABC):
    @abstractmethod
    def create_user(self, google_id: str, email: str, display_name: str, profile_picture: Optional[str], role: str = "user") -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_last_login(self, user_id: str) -> bool:
        pass

    @abstractmethod
    def get_user_by_google_id(self, google_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def create_chat(self, user_id: str, title: str, provider: str, model: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_chats(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def rename_chat(self, chat_id: str, new_title: str) -> bool:
        pass

    @abstractmethod
    def delete_chat(self, chat_id: str) -> bool:
        pass

    @abstractmethod
    def duplicate_chat(self, chat_id: str, new_title: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def archive_chat(self, chat_id: str, is_archived: bool = True) -> bool:
        pass

    @abstractmethod
    def update_chat_model(self, chat_id: str, provider: str, model: str) -> bool:
        pass

    @abstractmethod
    def save_message(self, chat_id: str, role: str, content: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_message_content(self, message_id: str, content: str) -> bool:
        pass

    @abstractmethod
    def delete_message(self, message_id: str) -> bool:
        pass

    @abstractmethod
    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear_chat_history(self, chat_id: str) -> bool:
        pass

    @abstractmethod
    def search_message_history(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def add_api_key(self, user_id: str, provider: str, key_val: str, nickname: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_api_key(self, key_id: str) -> bool:
        pass

    @abstractmethod
    def toggle_api_key(self, key_id: str, is_active: bool) -> bool:
        pass

    @abstractmethod
    def add_document(self, chat_id: str, user_id: str, filename: str, doc_name: str, page_count: int, chunk_count: int, status: str, document_summary: str = "", summary_embedding: Optional[List[float]] = None, metadata: Optional[str] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_documents(self, chat_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        pass

# SQLite Implementation
class SQLiteDatabase(DatabaseInterface):
    def __init__(self, db_path: str = "workspace.db"):
        self.db_path = db_path
        self.encryptor = EncryptionHelper()
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                google_id TEXT UNIQUE,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL,
                profile_picture TEXT,
                role TEXT DEFAULT 'user',
                created_at TEXT NOT NULL,
                last_login TEXT
            )
            """)

            # Chats table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                is_archived INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
            """)

            # Messages table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sources TEXT,
                confidence TEXT,
                confidence_emoji TEXT,
                grounding_score REAL,
                trust_level TEXT,
                hallucination_risk TEXT,
                claims TEXT,
                trace TEXT,
                performance TEXT,
                FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
            )
            """)

            # API Keys table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                nickname TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
            """)

            # Documents table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                document_summary TEXT,
                summary_embedding TEXT,
                metadata TEXT,
                FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
            )
            """)
            conn.commit()

            # Schema Migration Check for role, last_login, user_id, metadata
            cursor.execute("PRAGMA table_info(users)")
            user_cols = [r[1] for r in cursor.fetchall()]
            if "role" not in user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
            if "last_login" not in user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN last_login TEXT")
            
            cursor.execute("PRAGMA table_info(documents)")
            doc_cols = [r[1] for r in cursor.fetchall()]
            if "user_id" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN user_id TEXT DEFAULT ''")
            if "metadata" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN metadata TEXT")
            conn.commit()

    # User CRUD
    def create_user(self, google_id: str, email: str, display_name: str, profile_picture: Optional[str], role: str = "user") -> Dict[str, Any]:
        user_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        last_login = created_at
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO users (user_id, google_id, email, display_name, profile_picture, role, created_at, last_login) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, google_id, email, display_name, profile_picture, role, created_at, last_login)
            )
            conn.commit()
        return {
            "user_id": user_id,
            "google_id": google_id,
            "email": email,
            "display_name": display_name,
            "profile_picture": profile_picture,
            "role": role,
            "created_at": created_at,
            "last_login": last_login
        }

    def update_last_login(self, user_id: str) -> bool:
        last_login = datetime.now().isoformat()
        with self._get_connection() as conn:
            res = conn.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (last_login, user_id))
            conn.commit()
            return res.rowcount > 0

    def get_user_by_google_id(self, google_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()
            return dict(row) if row else None

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    # Chat CRUD
    def create_chat(self, user_id: str, title: str, provider: str, model: str) -> Dict[str, Any]:
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO chats (chat_id, user_id, title, model_provider, model_name, is_archived, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (chat_id, user_id, title, provider, model, now, now)
            )
            conn.commit()
        return {
            "chat_id": chat_id,
            "user_id": user_id,
            "title": title,
            "model_provider": provider,
            "model_name": model,
            "is_archived": 0,
            "created_at": now,
            "updated_at": now
        }

    def list_chats(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if include_archived:
                rows = conn.execute("SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM chats WHERE user_id = ? AND is_archived = 0 ORDER BY updated_at DESC", (user_id,)).fetchall()
            return [dict(r) for r in rows]

    def rename_chat(self, chat_id: str, new_title: str) -> bool:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            res = conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?", (new_title, now, chat_id))
            conn.commit()
            return res.rowcount > 0

    def delete_chat(self, chat_id: str) -> bool:
        with self._get_connection() as conn:
            # Foreign keys constraint should cascade, but we make sure
            res = conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            conn.commit()
            return res.rowcount > 0

    def duplicate_chat(self, chat_id: str, new_title: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            chat = conn.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
            if not chat:
                return None
            
            new_chat_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Insert duplicated chat
            conn.execute(
                "INSERT INTO chats (chat_id, user_id, title, model_provider, model_name, is_archived, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (new_chat_id, chat["user_id"], new_title, chat["model_provider"], chat["model_name"], now, now)
            )
            
            # Duplicate messages
            msgs = conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,)).fetchall()
            for msg in msgs:
                new_msg_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO messages (message_id, chat_id, role, content, timestamp, sources, confidence, confidence_emoji, grounding_score, trust_level, hallucination_risk, claims, trace, performance) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (new_msg_id, new_chat_id, msg["role"], msg["content"], msg["timestamp"], msg["sources"], msg["confidence"], msg["confidence_emoji"], msg["grounding_score"], msg["trust_level"], msg["hallucination_risk"], msg["claims"], msg["trace"], msg["performance"])
                )
            
            # Duplicate documents metadata (indices are copied physically in files)
            docs = conn.execute("SELECT * FROM documents WHERE chat_id = ?", (chat_id,)).fetchall()
            for doc in docs:
                new_doc_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO documents (document_id, chat_id, user_id, filename, doc_name, upload_date, page_count, chunk_count, status, document_summary, summary_embedding, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (new_doc_id, new_chat_id, doc["user_id"], doc["filename"], doc["doc_name"], doc["upload_date"], doc["page_count"], doc["chunk_count"], doc["status"], doc["document_summary"], doc["summary_embedding"], doc["metadata"])
                )
                
            conn.commit()
            
        return {
            "chat_id": new_chat_id,
            "user_id": chat["user_id"],
            "title": new_title,
            "model_provider": chat["model_provider"],
            "model_name": chat["model_name"],
            "is_archived": 0,
            "created_at": now,
            "updated_at": now
        }

    def archive_chat(self, chat_id: str, is_archived: bool = True) -> bool:
        now = datetime.now().isoformat()
        val = 1 if is_archived else 0
        with self._get_connection() as conn:
            res = conn.execute("UPDATE chats SET is_archived = ?, updated_at = ? WHERE chat_id = ?", (val, now, chat_id))
            conn.commit()
            return res.rowcount > 0

    def update_chat_model(self, chat_id: str, provider: str, model: str) -> bool:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            res = conn.execute("UPDATE chats SET model_provider = ?, model_name = ?, updated_at = ? WHERE chat_id = ?", (provider, model, now, chat_id))
            conn.commit()
            return res.rowcount > 0

    # Message CRUD
    def save_message(self, chat_id: str, role: str, content: str, **kwargs) -> Dict[str, Any]:
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Serialize dicts/lists to JSON strings
        sources = json.dumps(kwargs.get("sources")) if "sources" in kwargs else None
        claims = json.dumps(kwargs.get("claims")) if "claims" in kwargs else None
        trace = json.dumps(kwargs.get("trace")) if "trace" in kwargs else None
        performance = json.dumps(kwargs.get("performance")) if "performance" in kwargs else None
        
        confidence = kwargs.get("confidence")
        confidence_emoji = kwargs.get("confidence_emoji")
        grounding_score = kwargs.get("grounding_score")
        trust_level = kwargs.get("trust_level")
        hallucination_risk = kwargs.get("hallucination_risk")

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO messages (message_id, chat_id, role, content, timestamp, sources, confidence, confidence_emoji, grounding_score, trust_level, hallucination_risk, claims, trace, performance) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (message_id, chat_id, role, content, timestamp, sources, confidence, confidence_emoji, grounding_score, trust_level, hallucination_risk, claims, trace, performance)
            )
            # Update chat updated_at
            conn.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (timestamp, chat_id))
            conn.commit()
            
        return {
            "message_id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
            **kwargs
        }

    def update_message_content(self, message_id: str, content: str) -> bool:
        with self._get_connection() as conn:
            res = conn.execute("UPDATE messages SET content = ? WHERE message_id = ?", (content, message_id))
            conn.commit()
            return res.rowcount > 0

    def delete_message(self, message_id: str) -> bool:
        with self._get_connection() as conn:
            res = conn.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))
            conn.commit()
            return res.rowcount > 0

    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,)).fetchall()
            history = []
            for r in rows:
                item = dict(r)
                # Deserialize
                if item.get("sources"):
                    item["sources"] = json.loads(item["sources"])
                if item.get("claims"):
                    item["claims"] = json.loads(item["claims"])
                if item.get("trace"):
                    item["trace"] = json.loads(item["trace"])
                if item.get("performance"):
                    item["performance"] = json.loads(item["performance"])
                history.append(item)
            return history

    def clear_chat_history(self, chat_id: str) -> bool:
        with self._get_connection() as conn:
            res = conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.commit()
            return res.rowcount > 0

    def search_message_history(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        # Searches across all chats of a user
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, c.title as chat_title 
                   FROM messages m 
                   JOIN chats c ON m.chat_id = c.chat_id 
                   WHERE c.user_id = ? AND m.content LIKE ? 
                   ORDER BY m.timestamp DESC""",
                (user_id, f"%{query}%")
            ).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("sources"):
                    item["sources"] = json.loads(item["sources"])
                results.append(item)
            return results

    # API Keys CRUD
    def add_api_key(self, user_id: str, provider: str, key_val: str, nickname: str) -> Dict[str, Any]:
        key_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        encrypted = self.encryptor.encrypt(key_val)
        
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, user_id, provider, encrypted_key, nickname, is_active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                (key_id, user_id, provider, encrypted, nickname, created_at)
            )
            conn.commit()
            
        return {
            "key_id": key_id,
            "user_id": user_id,
            "provider": provider,
            "nickname": nickname,
            "is_active": 1,
            "created_at": created_at,
            "display_key": f"{key_val[:4]}_xxxxxxxx...{key_val[-4:]}" if len(key_val) > 8 else "xxxxxxxx"
        }

    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM api_keys WHERE user_id = ?", (user_id,)).fetchall()
            keys = []
            for r in rows:
                item = dict(r)
                decrypted = self.encryptor.decrypt(item["encrypted_key"])
                item["decrypted_key"] = decrypted
                item["display_key"] = f"{decrypted[:4]}_xxxxxxxx...{decrypted[-4:]}" if len(decrypted) > 8 else "xxxxxxxx"
                keys.append(item)
            return keys

    def delete_api_key(self, key_id: str) -> bool:
        with self._get_connection() as conn:
            res = conn.execute("DELETE FROM api_keys WHERE key_id = ?", (key_id,))
            conn.commit()
            return res.rowcount > 0

    def toggle_api_key(self, key_id: str, is_active: bool) -> bool:
        val = 1 if is_active else 0
        with self._get_connection() as conn:
            res = conn.execute("UPDATE api_keys SET is_active = ? WHERE key_id = ?", (val, key_id))
            conn.commit()
            return res.rowcount > 0

    # Documents CRUD
    def add_document(self, chat_id: str, user_id: str, filename: str, doc_name: str, page_count: int, chunk_count: int, status: str, document_summary: str = "", summary_embedding: Optional[List[float]] = None, metadata: Optional[str] = None) -> Dict[str, Any]:
        document_id = str(uuid.uuid4())
        upload_date = datetime.now().isoformat()
        embedding_str = json.dumps(summary_embedding) if summary_embedding else None
        
        if not metadata:
            metadata_dict = {
                "doc_name": doc_name,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "status": status,
                "document_summary": document_summary
            }
            metadata = json.dumps(metadata_dict)

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO documents (document_id, chat_id, user_id, filename, doc_name, upload_date, page_count, chunk_count, status, document_summary, summary_embedding, metadata) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document_id, chat_id, user_id, filename, doc_name, upload_date, page_count, chunk_count, status, document_summary, embedding_str, metadata)
            )
            conn.commit()
            
        return {
            "document_id": document_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "filename": filename,
            "doc_name": doc_name,
            "upload_date": upload_date,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "status": status,
            "document_summary": document_summary,
            "metadata": metadata
        }

    def list_documents(self, chat_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents WHERE chat_id = ?", (chat_id,)).fetchall()
            docs = []
            for r in rows:
                item = dict(r)
                if item.get("summary_embedding"):
                    item["summary_embedding"] = json.loads(item["summary_embedding"])
                docs.append(item)
            return docs

    def delete_document(self, document_id: str) -> bool:
        with self._get_connection() as conn:
            res = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return res.rowcount > 0

# Database Factory Singleton
class Database:
    _instance = None
    
    @classmethod
    def get_db(cls, db_type: str = "sqlite", **kwargs) -> DatabaseInterface:
        if cls._instance is None:
            if db_type == "sqlite":
                cls._instance = SQLiteDatabase(**kwargs)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        return cls._instance
