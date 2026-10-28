"""
Main Entrypoint Pipeline RAG untuk Literature Gap Matrix (Tugas S2 AI)
Domain: Edukasi & Kesehatan Remaja (Kecanduan Smartphone & Merokok/Vaping)
Mendukung Multi-Model LLM: Google Gemini, xAI Grok, Hugging Face, OpenAI, Ollama, DeepSeek, dll.

Cara Menjalankan:
    # 1. Menjalankan sesuai pengaturan .env (default: LLM_PROVIDER di .env)
    python main.py

    # 2. Menjalankan dengan override provider dari CLI
    python main.py --provider grok
    python main.py --provider huggingface --model meta-llama/Llama-3.1-70B-Instruct
    python main.py --provider gemini --model gemini-2.5-flash
    python main.py --provider openai --model gpt-4o-mini
    python main.py --provider ollama --model llama3.1

    # 3. Menjalankan ekstraksi ulang paksa (bypass cache)
    python main.py --force-refresh
"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    PAPERS_DIR,
    OUTPUTS_DIR,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOMAIN_TOPIC,
    USE_GEMINI_EMBEDDINGS,
)
from src.sample_papers_generator import ensure_sample_papers_if_empty
from src.pdf_loader import load_all_papers
from src.vector_store import AcademicVectorStore
from src.llm_provider import get_llm_client
from src.gap_extractor import LiteratureGapRAGPipeline
from src.report_exporter import export_all_outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline RAG Literature Gap Matrix Multi-Model (S2 AI)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Pilih LLM Provider: gemini | grok | huggingface | openai | ollama | deepseek | openrouter | custom",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nama model yang akan digunakan (misal: grok-2-latest, gemini-2.5-flash, meta-llama/Llama-3.1-70B-Instruct)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Paksa ekstraksi ulang PDF tanpa menggunakan cache lama",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    llm_client = get_llm_client(provider=args.provider, model=args.model)

    print("=" * 78)
    print(" PIPELINE RAG - GENERATOR LITERATURE GAP MATRIX & ANALISIS PAPER (S2 AI)")
    print("=" * 78)
    print(f"Domain Fokus : {DOMAIN_TOPIC}")
    print(f"Pilihan LLM  : {llm_client.provider_name.upper()} ({llm_client.model_name})")
    print(f"Status Model : {llm_client.status_message}")
    if not llm_client.is_configured:
        print("              [INFO] Berjalan dalam mode Offline Fallback Heuristic Extractor.")
        print("                     Setel API Key di file .env untuk hasil LLM generasi penuh.")
    if args.force_refresh:
        print("Mode Cache   : FORCE REFRESH (Mengabaikan cache hasil sebelumnya)")
    print("-" * 78)

    # 1. Pastikan ada PDF di folder data/papers/
    used_samples = ensure_sample_papers_if_empty(PAPERS_DIR)
    if not used_samples:
        print(f"[1/4] Membaca artikel PDF milik pengguna dari: {PAPERS_DIR}")
    else:
        print(f"[1/4] Membaca 5 contoh artikel PDF ilmiah dari: {PAPERS_DIR}")

    # 2. Load & Chunking PDF
    papers_dict = load_all_papers(
        PAPERS_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    if not papers_dict:
        print("[ERROR] Tidak ada teks PDF yang ditemukan di folder data/papers/.")
        sys.exit(1)

    all_chunks = [c for chunks in papers_dict.values() for c in chunks]
    print(
        f"      Berhasil mengekstrak {len(papers_dict)} dokumen PDF "
        f"menjadi total {len(all_chunks)} chunks."
    )

    # 3. Indexing ke Vector Store
    print("[2/4] Membangun Index Vector Database & Metadata Tagging per Artikel...")
    vector_store = AcademicVectorStore(
        persist_dir=VECTOR_STORE_DIR, use_gemini_embeddings=USE_GEMINI_EMBEDDINGS
    )
    vector_store.index_chunks(all_chunks)

    # 4. Multi-Aspect Retrieval & Ekstraksi Gap Matrix per Paper
    print(
        f"[3/4] Menjalankan Multi-Aspect RAG Retrieval & Ekstraksi via {llm_client.provider_name.upper()}..."
    )
    pipeline = LiteratureGapRAGPipeline(vector_store=vector_store, llm_client=llm_client)
    gap_rows = []
    for idx, (filename, p_chunks) in enumerate(papers_dict.items(), start=1):
        pdf_path = PAPERS_DIR / filename
        row = pipeline.extract_gap_row(
            idx=idx,
            filename=filename,
            paper_chunks=p_chunks,
            pdf_path=pdf_path,
            force_refresh=args.force_refresh,
        )
        gap_rows.append(row)

    # 5. Sintesis Analisis Paragraf untuk Usulan Paper Kelompok
    print(
        f"[4/4] Menghasilkan Sintesis Analisis Paragraf & Usulan Paper Kelompok ({llm_client.provider_name.upper()})..."
    )
    synthesis = pipeline.generate_synthesis_analysis(gap_rows)

    # 6. Ekspor semua hasil (termasuk metrik penggunaan token & cache)
    exported = export_all_outputs(
        outputs_dir=OUTPUTS_DIR,
        gap_rows=gap_rows,
        synthesis=synthesis,
        token_tracker=pipeline.token_tracker,
    )

    tracker = pipeline.token_tracker
    print("\n" + "=" * 78)
    print(" RINGKASAN PENGGUNAAN TOKEN & SMART CACHING")
    print("=" * 78)
    print(f" LLM Provider Digunakan   : {tracker.provider_name.upper()} ({tracker.model_name})")
    print(f" Total Artikel Diproses   : {len(gap_rows)} dokumen PDF")
    print(f" Cache Hit (Hasil Lama)   : {tracker.cached_hits} dokumen (0 token baru terpakai)")
    print(f" Panggilan API Baru       : {tracker.api_calls} request API")
    print(f" Token Input (Prompt)     : {tracker.prompt_tokens:,} tokens")
    print(f" Token Output (Generasi)  : {tracker.candidate_tokens:,} tokens")
    print(f" Total Token Run Ini      : {tracker.total_tokens:,} tokens")
    print(f" Estimasi Token Dihemat   : ~{tracker.estimated_saved_tokens:,} tokens (via Smart Caching)")
    print("=" * 78)
    print(" HASIL LENGKAP TERSIMPAN DI:")
    print("=" * 78)
    print(f" 1. Spreadsheet Excel (.xlsx)    : {exported['xlsx']}")
    print(f" 2. Spreadsheet CSV (.csv)       : {exported['csv']}")
    print(f" 3. Data Terstruktur (JSON)      : {exported['json']}")
    print(f" 4. Laporan & Analisis Paragraf  : {exported['markdown']}")
    print("-" * 78)
    print(f" Rekomendasi Judul Usulan Paper Kelompok:\n   \"{synthesis.usulan_judul_paper}\"")
    print("=" * 78)


if __name__ == "__main__":
    main()
