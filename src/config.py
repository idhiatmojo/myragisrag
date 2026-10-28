"""
Konfigurasi Pipeline RAG untuk Literature Gap Matrix
Domain: Edukasi & Kesehatan Remaja (Kecanduan Smartphone & Rokok/Vape)
Mendukung Multi-Model LLM: Google Gemini, xAI Grok, Hugging Face, OpenAI, Ollama, DeepSeek, dll.
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

# ==============================================================================
# KONFIGURASI PILIHAN MODEL (MULTI-MODEL LLM)
# ==============================================================================
# Pilihan default provider: gemini | grok | huggingface | openai | ollama | deepseek | openrouter
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

# 1. Google Gemini (https://aistudio.google.com/app/apikey)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite").strip()
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
USE_GEMINI_EMBEDDINGS = (
    os.environ.get("USE_GEMINI_EMBEDDINGS", "false").lower() in ("true", "1", "yes")
)

# 2. xAI Grok (https://console.x.ai/)
GROK_API_KEY = os.environ.get("GROK_API_KEY", os.environ.get("XAI_API_KEY", "")).strip()
GROK_LLM_MODEL = os.environ.get("GROK_LLM_MODEL", "grok-2-latest").strip()
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1").strip()

# 3. Hugging Face (https://huggingface.co/settings/tokens)
HUGGINGFACE_API_KEY = os.environ.get(
    "HUGGINGFACE_API_KEY", os.environ.get("HF_TOKEN", "")
).strip()
HUGGINGFACE_LLM_MODEL = os.environ.get(
    "HUGGINGFACE_LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"
).strip()
HUGGINGFACE_BASE_URL = os.environ.get(
    "HUGGINGFACE_BASE_URL", "https://router.huggingface.co/hf-inference/v1"
).strip()

# 4. OpenAI (https://platform.openai.com/api-keys)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

# 5. Local Ollama (https://ollama.com)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.1").strip()

# 6. DeepSeek (https://platform.deepseek.com)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_LLM_MODEL = os.environ.get("DEEPSEEK_LLM_MODEL", "deepseek-chat").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()

# 7. OpenRouter (https://openrouter.ai/keys)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_LLM_MODEL = os.environ.get(
    "OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash"
).strip()
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()

# 8. Custom / Generic OpenAI-Compatible Endpoint
CUSTOM_LLM_API_KEY = os.environ.get("CUSTOM_LLM_API_KEY", "").strip()
CUSTOM_LLM_MODEL = os.environ.get("CUSTOM_LLM_MODEL", "").strip()
CUSTOM_LLM_BASE_URL = os.environ.get("CUSTOM_LLM_BASE_URL", "").strip()


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
    """
    Skema terstruktur Literature Gap Matrix sesuai template spreadsheet:
    Name Paper | DOI | Problem | Metode | Dataset | Temuan | Keterbatasan | Future work
    """

    no: int = Field(default=1, description="Nomor urut artikel")
    filename: str = Field(default="", description="Nama file PDF sumber")
    name_paper: str = Field(
        description="Nama paper / judul lengkap artikel ilmiah beserta penulis dan tahun (contoh: Charkazi et al. (2026) - Agreement and classification performance...)"
    )
    doi: str = Field(
        description="Digital Object Identifier (DOI) artikel, contoh: https://doi.org/10.1016/j.abrep.2026.100740"
    )
    problem: str = Field(
        description="Masalah utama (Problem Statement) yang ingin diselesaikan oleh penelitian"
    )
    metode: str = Field(
        description="Metode, arsitektur, algoritma, atau pendekatan analisis yang digunakan"
    )
    dataset: str = Field(
        description="Dataset yang digunakan, jumlah partisipan/sampel, instrumen asesmen, dan metrik evaluasi"
    )
    temuan: str = Field(
        description="Temuan utama dan hasil signifikan yang dilaporkan (Key Findings)"
    )
    keterbatasan: str = Field(
        description="Keterbatasan penelitian (Limitations / threats to validity)"
    )
    future_work: str = Field(
        description="Peluang penelitian lanjutan (Future work / research directions)"
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

    provider_name: str = ""
    model_name: str = ""
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    cached_hits: int = 0
    api_calls: int = 0
    estimated_saved_tokens: int = 0
