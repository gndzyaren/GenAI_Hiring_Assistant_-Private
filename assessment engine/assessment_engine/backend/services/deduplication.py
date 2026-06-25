"""
Deduplication Service
Prevents question repetition using semantic similarity.
Uses FAISS + MiniLM if available, falls back to simple text hashing if not.
"""

import os
import json
import logging
import hashlib
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
DB_PATH = Path("data/questions_db")
META_FILE = DB_PATH / "questions_meta.json"

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    SEMANTIC_AVAILABLE = True
    logger.info("Semantic deduplication enabled (FAISS + MiniLM)")
except ImportError as e:
    SEMANTIC_AVAILABLE = False
    logger.warning(f"FAISS/sentence-transformers not found. Using hash-based dedup. ({e})")


class DeduplicationService:
    def __init__(self):
        self._model = None
        self._index = None
        self._metadata: List[dict] = []
        self._seen_hashes: set = set()
        DB_PATH.mkdir(parents=True, exist_ok=True)
        self._load_persisted()

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def _get_model(self):
        if not SEMANTIC_AVAILABLE:
            return None
        if self._model is None:
            try:
                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("MiniLM embedding model loaded.")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                return None
        return self._model

    def _get_index(self):
        if not SEMANTIC_AVAILABLE:
            return None
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(384)
        return self._index

    def _load_persisted(self):
        try:
            if META_FILE.exists():
                with open(META_FILE, "r") as f:
                    self._metadata = json.load(f)
                self._seen_hashes = {m.get("hash", "") for m in self._metadata}
                logger.info(f"Loaded {len(self._metadata)} existing questions.")
        except Exception as e:
            logger.warning(f"Could not load persisted metadata: {e}")

    def _persist(self):
        try:
            with open(META_FILE, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist metadata: {e}")

    def is_duplicate(self, question_text: str, threshold: float = SIMILARITY_THRESHOLD) -> Tuple[bool, float]:
        h = self._text_hash(question_text)
        if h in self._seen_hashes:
            return True, 1.0

        if not SEMANTIC_AVAILABLE:
            return False, 0.0

        model = self._get_model()
        index = self._get_index()
        if model and index and index.ntotal > 0:
            try:
                import numpy as np
                emb = model.encode([question_text], normalize_embeddings=True).astype(np.float32)
                k = min(5, index.ntotal)
                sims, _ = index.search(emb, k)
                max_sim = float(sims[0].max()) if sims[0].size > 0 else 0.0
                if max_sim >= threshold:
                    return True, max_sim
            except Exception as e:
                logger.warning(f"Semantic similarity check failed: {e}")

        return False, 0.0

    def add_question(self, question_id: str, question_text: str, section: str, topic: str) -> List[float]:
        h = self._text_hash(question_text)
        self._seen_hashes.add(h)

        embedding = []
        if SEMANTIC_AVAILABLE:
            model = self._get_model()
            index = self._get_index()
            if model and index:
                try:
                    import numpy as np
                    emb = model.encode([question_text], normalize_embeddings=True).astype(np.float32)
                    index.add(emb)
                    embedding = emb[0].tolist()
                except Exception as e:
                    logger.warning(f"Embedding add failed: {e}")

        self._metadata.append({
            "id": question_id,
            "hash": h,
            "question_text": question_text[:200],
            "section": section,
            "topic": topic
        })
        self._persist()
        return embedding

    def filter_unique(self, questions: list) -> list:
        unique = []
        seen_in_batch: set = set()

        for q in questions:
            text = q.question_text
            h = self._text_hash(text)

            if h in seen_in_batch:
                logger.debug(f"Batch duplicate (hash): {text[:60]}")
                continue

            is_dup, score = self.is_duplicate(text)
            if is_dup:
                logger.debug(f"Duplicate detected (score={score:.2f}): {text[:60]}")
                continue

            unique.append(q)
            seen_in_batch.add(h)

        for q in unique:
            emb = self.add_question(q.id, q.question_text, q.section, q.topic)
            if emb:
                q.embedding = emb

        return unique

    def get_stats(self) -> dict:
        return {
            "total_indexed": len(self._metadata),
            "semantic_enabled": SEMANTIC_AVAILABLE,
            "sections": self._count_by("section"),
            "topics": self._count_by("topic"),
        }

    def _count_by(self, key: str) -> dict:
        counts = {}
        for m in self._metadata:
            k = m.get(key, "unknown")
            counts[k] = counts.get(k, 0) + 1
        return counts


dedup_service = DeduplicationService()
