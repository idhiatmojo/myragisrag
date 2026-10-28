"""
Modul Vector Store & Hybrid Retriever (Dense Gemini Embeddings + Lexical BM25/TF-IDF)
Mendukung Metadata Filtering per artikel PDF agar informasi antar jurnal tidak tercampur.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
from src.pdf_loader import DocumentChunk


class AcademicVectorStore:
    """
    Vector Database ringan dengan persistensi disk yang mendukung:
    1. Dense Embedding menggunakan Google Gemini Embedding API (jika API Key aktif)
    2. Lexical/TF-IDF Vectorizer (sebagai hybrid booster & fallback otomatis)
    3. Metadata Filtering berdasarkan `filename` (Per-Paper Retrieval)
    """

    def __init__(self, persist_dir: Path, use_gemini_embeddings: bool = True):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[DocumentChunk] = []
        self.dense_embeddings: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.sparse_matrix: Optional[np.ndarray] = None

        self.use_gemini = bool(
            use_gemini_embeddings
            and GEMINI_API_KEY
            and GEMINI_API_KEY != "masukkan_api_key_gemini_anda_disini"
        )
        self._genai_client = None
        if self.use_gemini:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as exc:
                print(f"[INFO] Gagal inisialisasi Gemini Embedding Client ({exc}), beralih ke Hybrid Local Vectorizer.")
                self.use_gemini = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]{2,}", text.lower())

    def _build_sparse_index(self) -> None:
        """Membangun matriks TF-IDF untuk pencarian leksikal/hybrid."""
        tokenized_docs = [self._tokenize(c.text) for c in self.chunks]
        df_counter: Counter = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                df_counter[token] += 1

        # Ambil top 4000 istilah akademik paling informatif
        most_common = [w for w, _ in df_counter.most_common(4000)]
        self.vocab = {w: idx for idx, w in enumerate(most_common)}

        n_docs = len(self.chunks)
        vocab_size = len(self.vocab)
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for word, idx in self.vocab.items():
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + df_counter[word])) + 1.0

        matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for doc_idx, tokens in enumerate(tokenized_docs):
            tf_counts = Counter(tokens)
            total_terms = max(len(tokens), 1)
            for word, count in tf_counts.items():
                if word in self.vocab:
                    w_idx = self.vocab[word]
                    matrix[doc_idx, w_idx] = (count / total_terms) * self.idf[w_idx]

        # Normalisasi L2
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.sparse_matrix = matrix / norms

    def _encode_sparse_query(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf_counts = Counter(tokens)
        total_terms = max(len(tokens), 1)
        for word, count in tf_counts.items():
            if word in self.vocab and self.idf is not None:
                w_idx = self.vocab[word]
                vec[w_idx] = (count / total_terms) * self.idf[w_idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _embed_texts_gemini(self, texts: List[str]) -> Optional[np.ndarray]:
        """Menghasilkan dense vector menggunakan Google Gemini Embedding API."""
        if not self._genai_client:
            return None
        try:
            vectors = []
            batch_size = 25
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self._genai_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=batch,
                )
                for emb in resp.embeddings:
                    vectors.append(emb.values)
            arr = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        except Exception as exc:
            print(f"[WARN] Gemini Embedding gagal ({exc}). Menggunakan Hybrid TF-IDF Retriever.")
            self.use_gemini = False
            return None

    def index_chunks(self, all_chunks: List[DocumentChunk]) -> None:
        """Mengindeks seluruh chunk dokumen ke dalam Vector Store."""
        self.chunks = all_chunks
        self._build_sparse_index()

        if self.use_gemini:
            print(f"[RAG] Menghitung Dense Embeddings ({GEMINI_EMBEDDING_MODEL}) untuk {len(all_chunks)} chunks...")
            self.dense_embeddings = self._embed_texts_gemini([c.text for c in all_chunks])

        self.save_index()

    def save_index(self) -> None:
        """Menyimpan metadata chunk ke disk."""
        meta_path = self.persist_dir / "chunks_metadata.json"
        data = [c.to_dict() for c in self.chunks]
        meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.dense_embeddings is not None:
            np.save(self.persist_dir / "dense_vectors.npy", self.dense_embeddings)

    def retrieve(
        self,
        query: str,
        filename_filter: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Melakukan Hybrid Retrieval (Dense + Sparse Cosine Similarity)
        dengan dukungan filter nama file PDF (`filename_filter`).
        """
        if not self.chunks or self.sparse_matrix is None:
            return []

        candidate_indices = [
            idx
            for idx, c in enumerate(self.chunks)
            if (filename_filter is None or c.filename == filename_filter)
        ]
        if not candidate_indices:
            return []

        sparse_q = self._encode_sparse_query(query)
        sparse_scores = self.sparse_matrix[candidate_indices] @ sparse_q

        if self.use_gemini and self.dense_embeddings is not None:
            dense_q = self._embed_texts_gemini([query])
            if dense_q is not None:
                dense_scores = (self.dense_embeddings[candidate_indices] @ dense_q[0])
                # Hybrid Score: 70% Dense Semantic + 30% Lexical Keyword
                final_scores = 0.7 * dense_scores + 0.3 * sparse_scores
            else:
                final_scores = sparse_scores
        else:
            final_scores = sparse_scores

        ranked_order = np.argsort(final_scores)[::-1][:top_k]
        results: List[Tuple[DocumentChunk, float]] = []
        for pos in ranked_order:
            chunk_idx = candidate_indices[int(pos)]
            results.append((self.chunks[chunk_idx], float(final_scores[int(pos)])))

        return results
