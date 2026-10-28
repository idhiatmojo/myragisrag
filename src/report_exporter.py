"""
Modul Ekspor Hasil Literature Gap Matrix & Analisis Paragraf
Menghasilkan:
1. outputs/literature_gap_matrix.csv (Dapat dibuka langsung di Microsoft Excel / Google Sheets)
2. outputs/literature_gap_matrix.json (Data terstruktur)
3. outputs/laporan_gap_matrix_dan_analisis.md (Laporan lengkap Tabel + Analisis Paragraf untuk Usulan Paper Kelompok)
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
from src.config import DOMAIN_TOPIC, PaperGapRow, SynthesisAnalysis, TokenTracker


def export_all_outputs(
    outputs_dir: Path,
    gap_rows: List[PaperGapRow],
    synthesis: SynthesisAnalysis,
    token_tracker: Optional[TokenTracker] = None,
) -> Dict[str, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / "literature_gap_matrix.csv"
    json_path = outputs_dir / "literature_gap_matrix.json"
    md_path = outputs_dir / "laporan_gap_matrix_dan_analisis.md"

    # 1. Simpan CSV (UTF-8 with BOM agar langsung rapi saat dibuka di Microsoft Excel Windows)
    fieldnames = [
        "No",
        "Nama File PDF",
        "Penulis & Tahun",
        "Judul Artikel",
        "Publisher / Jurnal",
        "Domain & Sub-Topik",
        "Masalah Penelitian (Problem Statement)",
        "Arsitektur & Metode RAG",
        "Dataset & Metrik Evaluasi",
        "Hasil Utama (Key Findings)",
        "Research Gap / Keterbatasan (Limitations)",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in gap_rows:
            writer.writerow(
                {
                    "No": r.no,
                    "Nama File PDF": r.filename,
                    "Penulis & Tahun": r.penulis_tahun,
                    "Judul Artikel": r.judul_artikel,
                    "Publisher / Jurnal": r.publisher_jurnal,
                    "Domain & Sub-Topik": r.domain_subtopik,
                    "Masalah Penelitian (Problem Statement)": r.masalah_penelitian,
                    "Arsitektur & Metode RAG": r.arsitektur_metode_rag,
                    "Dataset & Metrik Evaluasi": r.dataset_dan_evaluasi,
                    "Hasil Utama (Key Findings)": r.hasil_utama,
                    "Research Gap / Keterbatasan (Limitations)": r.research_gap_keterbatasan,
                }
            )

    # 2. Simpan JSON
    payload = {
        "domain": DOMAIN_TOPIC,
        "token_usage_stats": token_tracker.model_dump() if token_tracker else {},
        "gap_matrix": [r.model_dump() for r in gap_rows],
        "synthesis_analysis": synthesis.model_dump(),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3. Simpan Laporan Markdown Lengkap (Tabel Gap Matrix + Analisis Paragraf Usulan Paper)
    def esc_md(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    md_lines = [
        "# Hasil Literature Gap Matrix & Analisis Usulan Paper Kelompok (S2 AI)",
        "",
        f"**Domain Fokus:** {DOMAIN_TOPIC}",
        "",
        "---",
        "",
        "## 1. Tabel Literature Gap Matrix (Hasil Ekstraksi Pipeline RAG)",
        "",
        "| No | Penulis & Tahun | Judul Artikel | Publisher / Jurnal | Masalah Penelitian | Arsitektur & Metode RAG | Dataset & Evaluasi | Hasil Utama | Research Gap / Keterbatasan |",
        "|:---:|---|---|---|---|---|---|---|---|",
    ]

    for r in gap_rows:
        md_lines.append(
            f"| {r.no} | **{esc_md(r.penulis_tahun)}** | {esc_md(r.judul_artikel)} | "
            f"{esc_md(r.publisher_jurnal)} | {esc_md(r.masalah_penelitian)} | "
            f"{esc_md(r.arsitektur_metode_rag)} | {esc_md(r.dataset_dan_evaluasi)} | "
            f"{esc_md(r.hasil_utama)} | {esc_md(r.research_gap_keterbatasan)} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Analisis Sintesis Literature Gap & Usulan Paper Kelompok (Bentuk Paragraf)",
            "",
            "### A. Lanskap dan Tren Pemanfaatan RAG pada Edukasi & Kesehatan Remaja",
            synthesis.paragraf_1_lanskap_dan_tren,
            "",
            "### B. Sintesis Kesenjangan Metodologis dan Arsitektur RAG (Technical Gap)",
            synthesis.paragraf_2_sintesis_gap_metodologis,
            "",
            "### C. Sintesis Kesenjangan Domain Edukasi Adiksi Remaja & Evaluasi (Domain & Evaluation Gap)",
            synthesis.paragraf_3_sintesis_gap_domain_dan_evaluasi,
            "",
            "### D. Usulan Riset & Novelty Paper Kelompok",
            synthesis.paragraf_4_usulan_paper_kelompok,
            "",
            "---",
            "",
            "## 3. Rekomendasi Judul Usulan Paper Kelompok",
            "",
            f"> **\"{synthesis.usulan_judul_paper}\"**",
            "",
        ]
    )

    if token_tracker:
        md_lines.extend(
            [
                "---",
                "",
                "## 4. Statistik Penggunaan Token & Efisiensi Caching",
                "",
                "| Metrik Evaluasi | Nilai | Penjelasan Teknis |",
                "|---|---|---|",
                f"| **Total Artikel Ilmiah** | {len(gap_rows)} dokumen | Seluruh PDF di folder `data/papers/` |",
                f"| **Cache Hit (Pakai Ulang Hasil Sebelumnya)** | {token_tracker.cached_hits} dokumen | Diambil dari cache instan (0 token baru) |",
                f"| **Panggilan API Baru** | {token_tracker.api_calls} panggilan | Request baru ke Google Gemini API |",
                f"| **Token Input (Prompt)** | {token_tracker.prompt_tokens:,} tokens | Konteks chunks RAG & prompt skema |",
                f"| **Token Output (Generasi)** | {token_tracker.candidate_tokens:,} tokens | Hasil ekstraksi JSON & analisis sintesis |",
                f"| **Total Token Terpakai Run Ini** | **{token_tracker.total_tokens:,} tokens** | Total kuota API yang dikonsumsi |",
                f"| **Estimasi Token Berhasil Dihemat** | **~{token_tracker.estimated_saved_tokens:,} tokens** | Berkat mekanisme Smart Fingerprint Caching |",
                "",
            ]
        )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": md_path,
    }
