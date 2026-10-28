"""
Modul Ingestion & Chunking Dokumen PDF Ilmiah
Mengekstrak teks dari PDF per halaman dan membaginya menjadi chunk dengan metadata.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    chunk_id: str
    filename: str
    page_number: int
    text: str
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_academic_text(text: str) -> str:
    """Membersihkan teks hasil ekstraksi PDF akademik (hyphenation, spasi berlebih)."""
    if not text:
        return ""
    # Gabungkan kata yang terpotong tanda hubung di akhir baris
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Normalisasi baris baru dan spasi
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def recursive_text_split(
    text: str, chunk_size: int = 1200, chunk_overlap: int = 250
) -> List[str]:
    """
    Memotong teks secara rekursif berdasarkan batas paragraf/kalimat
    agar konteks kalimat ilmiah tidak terputus di tengah.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Cari pemisah terbaik (paragraf -> titik kalimat -> spasi)
            window = text[start:end]
            for sep in ["\n\n", ". ", ".\n", "; ", " "]:
                last_sep = window.rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.4:
                    end = start + last_sep + len(sep)
                    break

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        if end >= text_len:
            break
        start = max(start + 1, end - chunk_overlap)

    return chunks


def load_and_chunk_pdf(
    pdf_path: Path, chunk_size: int = 1200, chunk_overlap: int = 250
) -> List[DocumentChunk]:
    """Membaca satu file PDF dan menghasilkan daftar DocumentChunk."""
    reader = PdfReader(str(pdf_path))
    doc_chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_academic_text(raw_text)
        if not cleaned_text:
            continue

        page_chunks = recursive_text_split(
            cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for sub_chunk in page_chunks:
            chunk_counter += 1
            doc_chunks.append(
                DocumentChunk(
                    chunk_id=f"{pdf_path.stem}_p{page_idx}_c{chunk_counter}",
                    filename=pdf_path.name,
                    page_number=page_idx,
                    text=sub_chunk,
                    char_count=len(sub_chunk),
                )
            )

    return doc_chunks


def load_all_papers(
    papers_dir: Path, chunk_size: int = 1200, chunk_overlap: int = 250
) -> Dict[str, List[DocumentChunk]]:
    """
    Membaca seluruh file PDF di dalam folder data/papers/
    Mengembalikan dictionary: {nama_file_pdf: [list_of_chunks]}
    """
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    papers_chunks: Dict[str, List[DocumentChunk]] = {}

    for pdf_path in pdf_files:
        chunks = load_and_chunk_pdf(
            pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if chunks:
            papers_chunks[pdf_path.name] = chunks

    return papers_chunks
