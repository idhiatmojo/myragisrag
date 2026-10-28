"""
Konfigurasi Pipeline RAG untuk Literature Gap Matrix
Domain: Edukasi & Kesehatan Remaja (Kecanduan Smartphone & Rokok/Vape)
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


def load_dotenv_simple(env_path: Path) -> None:
    """Membaca file .env secara mandiri tanpa bergantung pada library eksternal."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv_simple(BASE_DIR / ".env")

PAPERS_DIR = BASE_DIR / "data" / "papers"
OUTPUTS_DIR = BASE_DIR / "outputs"
VECTOR_STORE_DIR = OUTPUTS_DIR / "vector_store"
CACHE_DIR = OUTPUTS_DIR / "cache"
EXTRACTION_CACHE_FILE = CACHE_DIR / "extraction_cache.json"

# Konfigurasi Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite").strip()
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
USE_GEMINI_EMBEDDINGS = (
    os.environ.get("USE_GEMINI_EMBEDDINGS", "false").lower() in ("true", "1", "yes")
)

# Konfigurasi Chunking & Retrieval
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
TOP_K_PER_ASPECT = 3

# Konteks Domain Tugas S2 AI
DOMAIN_TOPIC = (
    "Edukasi dan Kesehatan Remaja (Adolescent Health & Education) "
    "terkait Pencegahan/Intervensi Kecanduan Smartphone (Smartphone/Screen Addiction) "
    "dan Bahaya Merokok/Vaping (Smoking/E-Cigarette Cessation)"
)


class PaperGapRow(BaseModel):
    """Skema terstruktur (Structured Output) untuk satu baris Literature Gap Matrix."""

    no: int = Field(description="Nomor urut artikel")
    filename: str = Field(description="Nama file PDF sumber")
    penulis_tahun: str = Field(description="Nama Penulis dan Tahun Terbit (contoh: Zhang et al., 2024)")
    judul_artikel: str = Field(description="Judul lengkap artikel ilmiah")
    publisher_jurnal: str = Field(
        description="Nama Publisher / Jurnal (contoh: Elsevier / IEEE / Springer / Taylor & Francis / SAGE / JMIR)"
    )
    domain_subtopik: str = Field(
        description="Fokus sub-domain (contoh: Edukasi Kesehatan Remaja - Kecanduan Smartphone / Berhenti Merokok)"
    )
    masalah_penelitian: str = Field(
        description="Masalah utama (Problem Statement) yang ingin diselesaikan oleh paper tersebut"
    )
    arsitektur_metode_rag: str = Field(
        description="Komponen & Arsitektur RAG yang digunakan (Retriever, Embedding, Vector DB, LLM, Knowledge Graph/Reranker)"
    )
    dataset_dan_evaluasi: str = Field(
        description="Dataset yang digunakan, partisipan (remaja/siswa), serta metrik evaluasi (RAGAS, Recall@K, F1, klinis/edukasi)"
    )
    hasil_utama: str = Field(
        description="Temuan utama dan performa sistem RAG yang dilaporkan"
    )
    research_gap_keterbatasan: str = Field(
        description="Keterbatasan penelitian (Limitations) dan peluang riset lanjutan (Future Work / Research Gap)"
    )


class SynthesisAnalysis(BaseModel):
    """Skema terstruktur untuk Sintesis Paragraf Analisis Gap & Usulan Paper Kelompok."""

    paragraf_1_lanskap_dan_tren: str = Field(
        description="Paragraf 1: Analisis tren pemanfaatan RAG pada domain edukasi & kesehatan remaja (kecanduan smartphone & rokok) dari matriks."
    )
    paragraf_2_sintesis_gap_metodologis: str = Field(
        description="Paragraf 2: Sintesis kelemahan/gap metodologis dan teknis dari penelitian-penelitian terdahulu (misal: Naive RAG, kurangnya personalisasi psikologis remaja, belum adanya guardrails empatik atau hybrid retrieval)."
    )
    paragraf_3_sintesis_gap_domain_dan_evaluasi: str = Field(
        description="Paragraf 3: Sintesis gap pada aspek domain edukasi kesehatan remaja (konteks bahasa lokal/slang remaja, multimodalitas, dan evaluasi dampak perubahan perilaku)."
    )
    paragraf_4_usulan_paper_kelompok: str = Field(
        description="Paragraf 4: Usulan konkrit paper kelompok (Judul Usulan, Arsitektur RAG yang diusulkan, Novelty/Kebaruan, dan Kontribusi untuk menutup gap di atas)."
    )
    usulan_judul_paper: str = Field(
        description="Rekomendasi judul paper ilmiah kelompok S2 AI yang spesifik, kuat, dan siap diajukan."
    )


class TokenTracker(BaseModel):
    """Pencatat statistik penggunaan token dan efisiensi caching."""

    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    cached_hits: int = 0
    api_calls: int = 0
    estimated_saved_tokens: int = 0
