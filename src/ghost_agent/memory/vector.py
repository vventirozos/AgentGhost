import hashlib
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from ..utils.logging import Icons, pretty_log
from ..utils.helpers import get_utc_timestamp

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

logger = logging.getLogger("GhostAgent")

class GhostEmbeddingFunction(EmbeddingFunction):
    """
    Custom robust embedding function that uses the upstream LLM.
    Handles proxy bypass and retries.
    """
    def __init__(self, upstream_url: str, tor_proxy: str = None):
        import httpx
        self.url = f"{upstream_url}/v1/embeddings"
        
        # Determine proxy usage
        proxy_url = None
        if "127.0.0.1" not in upstream_url and "localhost" not in upstream_url and tor_proxy:
             proxy_url = tor_proxy.replace("socks5://", "socks5h://")

        self.client = httpx.Client(timeout=60.0, proxy=proxy_url, http2=False)

    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB expects a List of Embeddings
        import time
        for attempt in range(3):
            try:
                resp = self.client.post(self.url, json={"input": input, "model": "default"})
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Embedding failed after 3 attempts: {e}")
                    raise

class VectorMemory:
    def __init__(self, memory_dir: Path, upstream_url: str, tor_proxy: str = None):
        """
        Robust Initialization with Explicit Settings.
        """
        self.chroma_dir = memory_dir
        if not self.chroma_dir.exists():
            self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.library_file = self.chroma_dir / "library_index.json"
        if not self.library_file.exists():
            self.library_file.write_text("[]")

        # --- GRANITE4 STYLE: LOCAL EMBEDDINGS ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from chromadb.utils import embedding_functions
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                break  # Success, exit the retry loop
            except Exception as e:
                logger.warning(f"Error loading embedding model (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(6) # Wait before retry
                else:
                    logger.error(f"Failed to load embedding model after {max_retries} attempts.")
                    sys.exit(1)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            
            # Switch back to 'agent_memory' to match standard naming
            collection_name = "agent_memory"
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            
            pretty_log("Memory System", f"Initialized [{collection_name}] ({self.collection.count()} items)", icon=Icons.SYSTEM_READY)
            
        except Exception as e:
            if "already exists" in str(e) or "Embedding function conflict" in str(e):
                pretty_log("Memory Conflict", "Embedding provider mismatch. Resetting collection for new provider...", level="WARNING", icon="⚠️")
                # Fallback: if 'v2' also conflicts (unlikely), we'd need to reset. 
                # For now, renaming to v2 is the safest non-destructive path.
                sys.exit(1)
            logger.error(f"CRITICAL DB ERROR: {e}")
            self.collection = None

    def search_advanced(self, query: str, limit: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        parsed_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                parsed_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i]
                })
        
        return parsed_results

    def _update_library_index(self, filename: str, action: str):
        try:
            data = json.loads(self.library_file.read_text())
            if action == "add" and filename not in data:
                data.append(filename)
            elif action == "remove" and filename in data:
                data.remove(filename)
            self.library_file.write_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Library index error: {e}")

    def get_library(self):
        if not self.library_file.exists():
            return []
        try:
            data = json.loads(self.library_file.read_text())
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []
    
    def add(self, text: str, meta: dict = None):
        if len(text) < 5: return
        mem_id = hashlib.md5(text.encode("utf-8")).hexdigest()

        existing = self.collection.get(ids=[mem_id])
        if existing and existing['ids']:
            return

        metadata = meta or {"timestamp": get_utc_timestamp(), "type": "auto"}
        self.collection.add(documents=[text], metadatas=[metadata], ids=[mem_id])
        pretty_log("Memory Save", text, icon=Icons.MEM_SAVE)

    def smart_update(self, text: str, type_label: str = "auto"):
        try:
            results = self.collection.query(query_texts=[text], n_results=1)
            if results['ids'] and results['ids'][0]:
                dist = results['distances'][0][0]
                existing_id = results['ids'][0][0]

                if dist < 0.20:
                    self.collection.delete(ids=[existing_id])
                    pretty_log("Memory Update", f"Refining existing entry (Sim={dist:.2f})", icon=Icons.RETRY)

            self.add(text, {"timestamp": get_utc_timestamp(), "type": type_label})
        except Exception as e:
            logger.error(f"Smart Update Error: {e}")

    def ingest_document(self, filename: str, chunks: List[str]):
        try:
            # ENRICH CHUNKS WITH SOURCE CONTEXT
            enriched_chunks = [f"[Source: {filename}]\n{chunk}" for chunk in chunks]
            
            ids = [hashlib.md5(f"{filename}_{i}_{chunk[:20]}".encode()).hexdigest() for i, chunk in enumerate(chunks)]
            metadatas = [{"timestamp": get_utc_timestamp(), "type": "document", "source": filename} for _ in range(len(chunks))]

            batch_size = 25
            for i in range(0, len(enriched_chunks), batch_size):
                self.collection.upsert(
                    documents=enriched_chunks[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
                if i % 10 == 0:
                    pretty_log("Memory Ingest", f"{filename} ({i+1}/{len(chunks)})", icon=Icons.MEM_INGEST)

            self._update_library_index(filename, "add")
            return True, f"Successfully ingested {len(chunks)} chunks from {filename}."
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            return False, str(e)

    def search(self, query: str, inject_identity: bool = True):
            try:
                search_queries = [query]
                
                # CONDITIONAL IDENTITY INJECTION
                # Only inject identity context if the query actually asks for it.
                # This prevents "pollution" where asking about Python code retrieves "My name is Bob".
                identity_triggers = ["who", "my ", " i ", "profile", "preference", "remember"]
                should_inject_identity = inject_identity and any(t in query.lower() for t in identity_triggers)
                
                if should_inject_identity:
                    search_queries.insert(0, "User's profile. User's name. User preferences.")

                results = self.collection.query(
                    query_texts=search_queries,
                    n_results=10,
                )

                candidates = []
                seen_docs = set()

                def process_batch(batch_idx, is_identity_batch):
                    if not results['documents'] or len(results['documents']) <= batch_idx:
                        return

                    for doc, meta, dist in zip(
                        results['documents'][batch_idx],
                        results['metadatas'][batch_idx],
                        results['distances'][batch_idx]
                    ):
                        if doc in seen_docs: continue

                        m_type = meta.get('type', 'auto')
                        doc_lower = doc.lower()
                        timestamp = meta.get('timestamp', '0000-00-00')

                        is_summary = m_type == "document_summary"

                        is_name_memory = (
                            "name is" in doc_lower or
                            "call me" in doc_lower or
                            "user's" in doc_lower or
                            "user is" in doc_lower
                        )

                        if is_name_memory:
                            threshold = 1.0  # Tightened from 1.2
                        elif is_summary:
                            threshold = 0.75 # Tightened from 0.85
                        elif is_identity_batch:
                            threshold = 0.8 if m_type == 'manual' else 0.65
                        else:
                            # General technical memory needs strict relevance
                            if m_type == 'document':
                                threshold = 1.25 # Relaxed for Asymmetric QA (Short Query vs Long Document Chunk)
                            else:
                                threshold = 0.65 if m_type == 'manual' else 0.55

                        if dist < threshold or is_name_memory or is_summary:
                            priority_score = 1

                            if is_name_memory: priority_score = -20
                            elif is_summary: priority_score = -15
                            elif is_identity_batch: priority_score = -10
                            elif m_type == 'document': priority_score = -5 # Elevate document priority above general manual/auto
                            elif m_type == 'manual': priority_score = 0

                            candidates.append({
                                "doc": doc,
                                "meta": meta,
                                "dist": dist,
                                "type": m_type,
                                "p_score": priority_score,
                                "timestamp": timestamp
                            })
                            seen_docs.add(doc)

                if should_inject_identity:
                    process_batch(0, is_identity_batch=True)
                    process_batch(1, is_identity_batch=False)
                else:
                    process_batch(0, is_identity_batch=False)

                # 1. Sort by timestamp descending (newest first)
                candidates.sort(key=lambda x: x['timestamp'], reverse=True)
                # 2. Sort by priority score ascending (lower is better)
                candidates.sort(key=lambda x: x['p_score'])

                final_selection = candidates[:12]
                if not final_selection: return ""

                output = []
                for item in final_selection:
                    ts = item['meta'].get('timestamp', '?')
                    m_type = item['meta'].get('type', 'auto').upper()
                    doc_text = item['doc']

                    prefix = ""
                    if item['p_score'] <= -15: prefix = "**[MASTER SUMMARY]** "
                    elif item['p_score'] <= -10: prefix = "**[IDENTITY]** "
                    elif item['p_score'] == 0: prefix = "**[USER PRIORITY]** "
                    elif item['p_score'] == 2: prefix = "**[DOCUMENT SOURCE]** "

                    output.append(f"[{ts}] ({m_type}) {prefix}{doc_text}")

                return "\n---\n".join(output)

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return ""

    def delete_document_by_name(self, filename: str):
        self.collection.delete(where={"source": filename})
        self._update_library_index(filename, "remove")
        return True, "Deleted"

    def delete_by_query(self, query: str):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
                where={"type": {"$ne": "document"}}
            )
            if not results['ids'] or not results['ids'][0]:
                return False, "Memory not found."

            dist = results['distances'][0][0]
            doc_text = results['documents'][0][0]
            mem_id = results['ids'][0][0]

            if dist > 0.5:
                return False, f"Best match was '{doc_text}' but score ({dist:.2f}) was too low."

            self.collection.delete(ids=[mem_id])
            pretty_log("Memory Wipe", doc_text, icon=Icons.MEM_WIPE)
            return True, f"Successfully forgot: [[{doc_text}]]"
        except Exception as e:
            return False, f"Error: {e}"