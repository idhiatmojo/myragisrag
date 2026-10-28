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


"""
Modul Ekspor Hasil Literature Gap Matrix & Analisis Paragraf
Menghasilkan:
1. outputs/literature_gap_matrix.xlsx (File Excel resmi dengan styling, auto-wrap & link DOI)
2. outputs/literature_gap_matrix.csv  (Format CSV UTF-8 BOM siap buka di spreadsheet)
3. outputs/literature_gap_matrix.json (Data terstruktur JSON)
4. outputs/laporan_gap_matrix_dan_analisis.md (Laporan lengkap Tabel + 4 Paragraf Sintesis)
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
from src.config import DOMAIN_TOPIC, PaperGapRow, SynthesisAnalysis, TokenTracker


def _export_to_excel(xlsx_path: Path, gap_rows: List[PaperGapRow]) -> Path:
    """Mengekspor matriks ke format Microsoft Excel (.xlsx) dengan styling profesional."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Literature Gap Matrix"
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "No",
            "Name Paper",
            "DOI",
            "Problem",
            "Metode",
            "Dataset",
            "Temuan",
            "Keterbatasan",
            "Future work",
        ]
        ws.append(headers)

        header_fill = PatternFill(
            start_color="1F4E79", end_color="1F4E79", fill_type="solid"
        )
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        thin_side = Side(border_style="thin", color="D9D9D9")
        cell_border = Border(
            left=thin_side, right=thin_side, top=thin_side, bottom=thin_side
        )

        data_font = Font(name="Calibri", size=10)
        data_align = Alignment(horizontal="left", vertical="top", wrap_text=True)
        link_font = Font(name="Calibri", size=10, color="0563C1", underline="single")

        ws.row_dimensions[1].height = 28
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = cell_border

        for r_idx, r in enumerate(gap_rows, start=2):
            row_vals = [
                r.no,
                r.name_paper,
                r.doi,
                r.problem,
                r.metode,
                r.dataset,
                r.temuan,
                r.keterbatasan,
                r.future_work,
            ]
            ws.append(row_vals)
            ws.row_dimensions[r_idx].height = 80

            for c_idx, val in enumerate(row_vals, start=1):
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.border = cell_border
                cell.font = data_font
                cell.alignment = data_align

                if c_idx == 3 and str(val).startswith("http"):
                    cell.hyperlink = str(val)
                    cell.font = link_font

                if c_idx == 1:
                    cell.alignment = Alignment(horizontal="center", vertical="top")

        col_widths = {
            1: 6,   # No
            2: 36,  # Name Paper
            3: 32,  # DOI
            4: 45,  # Problem
            5: 42,  # Metode
            6: 38,  # Dataset
            7: 45,  # Temuan
            8: 45,  # Keterbatasan
            9: 45,  # Future work
        }
        for col_idx, width in col_widths.items():
            ws.column_dimensions[get_column_letter(col_idx)].width = width

        ws.freeze_panes = "A2"
        target_path = xlsx_path
        try:
            wb.save(str(target_path))
        except PermissionError:
            target_path = xlsx_path.with_name(f"{xlsx_path.stem}_baru{xlsx_path.suffix}")
            print(f"[INFO] {xlsx_path.name} sedang dibuka di Excel. Menyimpan ke: {target_path.name}")
            wb.save(str(target_path))
        return target_path
    except Exception as exc:
        print(f"[WARN] Gagal membuat file XLSX via openpyxl: {exc}")
        return xlsx_path


def _export_to_csv(csv_path: Path, gap_rows: List[PaperGapRow]) -> Path:
    """Mengekspor matriks ke format CSV dengan penanganan file lock jika dibuka di Excel."""
    fieldnames = [
        "No",
        "Name Paper",
        "DOI",
        "Problem",
        "Metode",
        "Dataset",
        "Temuan",
        "Keterbatasan",
        "Future work",
    ]
    target_path = csv_path
    rows = [
        {
            "No": r.no,
            "Name Paper": r.name_paper,
            "DOI": r.doi,
            "Problem": r.problem,
            "Metode": r.metode,
            "Dataset": r.dataset,
            "Temuan": r.temuan,
            "Keterbatasan": r.keterbatasan,
            "Future work": r.future_work,
        }
        for r in gap_rows
    ]

    try:
        with target_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target_path
    except PermissionError:
        target_path = csv_path.with_name(f"{csv_path.stem}_baru{csv_path.suffix}")
        print(f"[INFO] {csv_path.name} sedang dibuka di aplikasi lain. Menyimpan ke: {target_path.name}")
        with target_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target_path


def export_all_outputs(
    outputs_dir: Path,
    gap_rows: List[PaperGapRow],
    synthesis: SynthesisAnalysis,
    token_tracker: Optional[TokenTracker] = None,
) -> Dict[str, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = outputs_dir / "literature_gap_matrix.xlsx"
    csv_path = outputs_dir / "literature_gap_matrix.csv"
    json_path = outputs_dir / "literature_gap_matrix.json"
    md_path = outputs_dir / "laporan_gap_matrix_dan_analisis.md"

    # 1. Simpan XLSX Microsoft Excel resmi
    actual_xlsx_path = _export_to_excel(xlsx_path, gap_rows)

    # 2. Simpan CSV (UTF-8 with BOM agar langsung rapi saat dibuka di Microsoft Excel / Google Sheets)
    actual_csv_path = _export_to_csv(csv_path, gap_rows)

    # 3. Simpan JSON
    payload = {
        "domain": DOMAIN_TOPIC,
        "token_usage_stats": token_tracker.model_dump() if token_tracker else {},
        "gap_matrix": [r.model_dump() for r in gap_rows],
        "synthesis_analysis": synthesis.model_dump(),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. Simpan Laporan Markdown Lengkap (Tabel Gap Matrix + Analisis Paragraf Usulan Paper)
    def esc_md(text: str) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    md_lines = [
        "# Hasil Literature Gap Matrix & Analisis Usulan Paper Kelompok (S2 AI)",
        "",
        f"**Domain Fokus:** {DOMAIN_TOPIC}",
        "",
        "---",
        "",
        "## 1. Tabel Literature Gap Matrix",
        "",
        "| No | Name Paper | DOI | Problem | Metode | Dataset | Temuan | Keterbatasan | Future work |",
        "|:---:|---|---|---|---|---|---|---|---|",
    ]

    for r in gap_rows:
        md_lines.append(
            f"| {r.no} | **{esc_md(r.name_paper)}** | [{esc_md(r.doi)}]({esc_md(r.doi)}) | "
            f"{esc_md(r.problem)} | {esc_md(r.metode)} | {esc_md(r.dataset)} | "
            f"{esc_md(r.temuan)} | {esc_md(r.keterbatasan)} | {esc_md(r.future_work)} |"
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
        "xlsx": actual_xlsx_path,
        "csv": actual_csv_path,
        "json": json_path,
        "markdown": md_path,
    }
