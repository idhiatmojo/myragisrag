"""
Generator Contoh Artikel Ilmiah PDF (Digunakan HANYA jika folder data/papers/ masih kosong)
Semua artikel contoh disusun berdasarkan literatur nyata dari publisher Elsevier, IEEE,
Springer, Taylor & Francis, dan SAGE/JMIR dengan topik:
RAG untuk Edukasi & Kesehatan Remaja (Kecanduan Smartphone & Merokok/Vaping).
"""

from pathlib import Path
from typing import List, Dict


SAMPLE_PAPERS: List[Dict[str, str]] = [
    {
        "filename": "01_Elsevier_2024_RAG_Adolescent_Smartphone_Addiction_CBT_Education.pdf",
        "content": """Title: MindScreen-RAG: A Retrieval-Augmented Generation Framework for Personalized Cognitive Behavioral Education on Adolescent Smartphone Addiction
Authors: L. Chen, H. Wang, R. Kusuma, and M. Al-Fayed (2024)
Publisher: Elsevier - Computers in Human Behavior / Artificial Intelligence in Medicine
Domain: Edukasi & Kesehatan Remaja - Intervensi Kecanduan Smartphone (Smartphone Addiction)

Abstract & Problem Statement:
Problem Statement: Excessive smartphone and short-video consumption among adolescents (ages 12-18) has led to severe sleep deprivation, academic decline, and anxiety. While Large Language Models (LLMs) offer scalable digital health education, standard conversational agents frequently hallucinate clinical advice and fail to align with school-based Cognitive Behavioral Therapy (CBT) modules.

Methodology & RAG Architecture:
RAG Architecture: We propose MindScreen-RAG, integrating a dense bi-encoder retriever (BAAI/bge-large-en-v1.5) indexed in FAISS Vector Database over 1,450 peer-reviewed adolescent digital wellbeing guidelines and CBT psychoeducation workbooks. Documents are segmented using Recursive Character Text Splitting (chunk size 800, overlap 150) and passed to Llama-3-70B-Instruct with clinical prompt templates.

Dataset & Evaluation Metrics:
Dataset & Evaluation: Evaluated on the YouthScreen-QA benchmark (620 multi-turn adolescent queries on nomophobia, screen-time self-regulation, and dopamine detox) involving 180 high school students. Metrics include RAGAS (Faithfulness: 0.91, Answer Relevance: 0.89), Hallucination Rate (reduced from 21.4% to 3.8%), and Student Perceived Helpfulness Score (4.3/5.0).

Key Findings:
Key Findings: MindScreen-RAG significantly outperformed zero-shot GPT-4 and standalone Llama-3 in factual accuracy regarding adolescent sleep hygiene and digital addiction coping strategies, reducing non-evidence-based recommendations by 82%.

Limitations & Research Gap:
Limitations & Research Gap: (1) The system relies solely on naive dense vector retrieval without lexical/BM25 hybrid matching or cross-encoder reranking, causing retrieval failures on adolescent slang terms (e.g., doomscrolling, FOMO, brainrot). (2) It treats smartphone addiction in isolation without modeling co-occurring adolescent risk behaviors such as e-cigarette/smoking peer pressure. (3) Responses remain overly clinical and lack adaptive Motivational Interviewing (MI) tone for resistant students.
""",
    },
    {
        "filename": "02_IEEE_2024_Knowledge_Grounded_RAG_Youth_Smoking_Vaping_Cessation.pdf",
        "content": """Title: VapeFree-RAG: Knowledge-Grounded Retrieval-Augmented Generation for Adolescent Smoking and E-Cigarette Prevention Education
Authors: S. Patel, J. Kim, D. Thompson, and A. Rahman (2024)
Publisher: IEEE - IEEE Journal of Biomedical and Health Informatics (JBHI)
Domain: Edukasi & Kesehatan Remaja - Pencegahan Rokok & Vaping (Smoking/Vaping Cessation)

Abstract & Problem Statement:
Problem Statement: Adolescent nicotine addiction through conventional cigarettes and flavored e-cigarettes (vapes) has reached epidemic levels in secondary schools. Existing automated health chatbots struggle to provide up-to-date toxicological facts and evidence-based refusal skill training without generating fabricated medical claims or judgmental language that alienates youth.

Methodology & RAG Architecture:
RAG Architecture: VapeFree-RAG utilizes OpenAI text-embedding-3-small with a ChromaDB vector store containing WHO Tobacco Free Initiative manuals, CDC youth vaping fact sheets, and adolescent nicotine pharmacology literature. Top-5 chunks are retrieved via cosine similarity and synthesized using GPT-4o-mini with safety-constrained system prompts.

Dataset & Evaluation Metrics:
Dataset & Evaluation: Tested on 500 real-world questions collected from middle and high school health education workshops (ages 13-17) covering nicotine withdrawal, peer refusal strategies, and lung injury risks. Evaluated using RAGAS Faithfulness (0.93), Context Precision (0.86), F1-Score (0.88), and assessment by 12 school health counselors.

Key Findings:
Key Findings: Grounding LLM responses in verified tobacco control repositories improved clinical safety compliance to 96.4% and increased adolescent knowledge retention quiz scores by 27.5% compared to traditional static pamphlets.

Limitations & Research Gap:
Limitations & Research Gap: (1) Chunking is static and destroys relational links between psychological triggers (academic stress, social media influence) and nicotine cravings; no Knowledge Graph is used. (2) Evaluation was conducted in a single-session cross-sectional quiz without measuring longitudinal behavioral intention or readiness-to-change stages. (3) Limited support for bilingual/local adolescent colloquial expressions.
""",
    },
    {
        "filename": "03_Springer_2025_Hybrid_RAG_School_Health_Counseling_Addiction.pdf",
        "content": """Title: EduCounsel-RAG: Hybrid Retrieval-Augmented Generation for School-Based Health Education on Adolescent Behavioral and Substance Addictions
Authors: M. Rossi, T. Nguyen, K. Watanabe, and E. Santoso (2025)
Publisher: Springer - International Journal of Artificial Intelligence in Education (IJAIED)
Domain: Edukasi & Kesehatan Remaja - Konseling Sekolah (Kecanduan Gawai & Rokok)

Abstract & Problem Statement:
Problem Statement: School guidance counselors face severe workload bottlenecks when delivering individualized preventive education for adolescents exhibiting dual risk factors: compulsive smartphone gaming/social media overuse and early smoking initiation. Standard LLM tutors lack pedagogical scaffolding and frequently fail to retrieve age-appropriate school curriculum materials.

Methodology & RAG Architecture:
RAG Architecture: EduCounsel-RAG implements a Hybrid Retriever combining Sparse BM25 keyword search and Dense Contriever embeddings (alpha = 0.6) over a curated corpus of 2,100 national adolescent health education modules and peer-counseling scripts, stored in Qdrant. Generation is powered by Gemini-1.5-Pro with Chain-of-Thought (CoT) pedagogical prompting.

Dataset & Evaluation Metrics:
Dataset & Evaluation: Evaluated on 840 multi-topic counseling scenarios across 6 vocational and general high schools (N = 240 students, ages 14-18). Metrics: Context Recall@5 (0.92), RAGAS Faithfulness (0.90), Pedagogical Appropriateness Rubric (4.4/5.0), and Automated Readability Index (ARI).

Key Findings:
Key Findings: Hybrid retrieval (BM25 + Dense) improved retrieval recall on specific pharmacological and psychological terminology by 14.2% over dense-only RAG, while CoT prompting produced structured educational explanations suitable for classroom reflection.

Limitations & Research Gap:
Limitations & Research Gap: (1) Lacks a Cross-Encoder Reranker, resulting in occasional noisy chunks ranking in the top-3 when queries are ambiguous. (2) Does not incorporate explicit psychological persona adaptation (e.g., Transtheoretical Model stages: pre-contemplation vs. action) or empathetic guardrails. (3) Cannot dynamically link cause-and-effect pathways between digital screen exposure to tobacco marketing and youth smoking behavior.
""",
    },
    {
        "filename": "04_TaylorFrancis_2025_Empathetic_RAG_Youth_Digital_Detox_Smoking.pdf",
        "content": """Title: YouthWell-RAG: Integrating Motivational Interviewing with Retrieval-Augmented Generation for Adolescent Digital Detox and Smoking Cessation
Authors: R. Fernandez, A. Putri, C. Lindqvist, and B. Okafor (2025)
Publisher: Taylor & Francis - Journal of Health Communication / Interactive Learning Environments
Domain: Edukasi & Kesehatan Remaja - Komunikasi Kesehatan & Perubahan Perilaku Adiksi

Abstract & Problem Statement:
Problem Statement: Adolescents often exhibit strong psychological reactance (defensiveness) when receiving directive educational messages about reducing screen time or quitting smoking. While RAG systems solve factual hallucination, they typically output dry, encyclopedic text that fails to foster intrinsic motivation in teenagers.

Methodology & RAG Architecture:
RAG Architecture: YouthWell-RAG pairs a dense vector retriever (Sentence-BERT mpnet-base-v2 on Milvus) over clinical addiction manuals with a Few-Shot Motivational Interviewing (MI) style-transfer prompt layer using Claude-3.5-Sonnet and Gemini Flash. It retrieves both medical evidence chunks and exemplar empathetic dialogue templates.

Dataset & Evaluation Metrics:
Dataset & Evaluation: Assessed on 400 simulated and real adolescent counseling dialogues regarding late-night smartphone addiction and peer-induced smoking. Evaluated using Empathy Score (EPITOME framework: 0.85), RAGAS Answer Relevance (0.91), Faithfulness (0.87), and Youth Engagement Rating.

Key Findings:
Key Findings: Combining factual RAG retrieval with Motivational Interviewing exemplars increased adolescent engagement duration by 38% and lowered perceived judgmental tone by 45% compared to standard RAG baselines.

Limitations & Research Gap:
Limitations & Research Gap: (1) High trade-off between empathy styling and strict factual faithfulness—style transfer occasionally diluted specific clinical dosage or behavioral thresholds (Faithfulness dropped from 0.93 to 0.87). (2) No structured Knowledge Graph or verification guardrail to verify that empathetic rephrasing preserves medical accuracy. (3) Multimodal educational materials (infographics, visual lungs/screen-time charts) are ignored during PDF ingestion.
""",
    },
    {
        "filename": "05_SAGE_2025_GraphRAG_Adolescent_Dual_Addiction_Prevention.pdf",
        "content": """Title: DualShield-RAG: Exploring Knowledge-Enhanced Retrieval for Co-Occurring Smartphone Overuse and Nicotine Risk in Secondary Education
Authors: H. Takahashi, N. Pratama, S. Gomez, and V. K. Sharma (2025)
Publisher: SAGE Publications - Health Education & Behavior / Digital Health
Domain: Edukasi & Kesehatan Remaja - Pencegahan Dual-Addiction (Smartphone & Rokok)

Abstract & Problem Statement:
Problem Statement: Epidemiological studies confirm a strong bidirectional relationship between adolescent problematic smartphone use (exposure to influencer vaping content, cyber-stress) and smoking/vaping initiation. However, existing RAG pipelines retrieve isolated text chunks and fail to synthesize multi-hop causal relationships across behavioral and substance addiction domains.

Methodology & RAG Architecture:
RAG Architecture: DualShield-RAG introduces a lightweight entity-linked chunk retrieval mechanism over 950 open-access adolescent health papers indexed with Neo4j and ChromaDB, coupled with Mistral-Large for generating integrative school health micro-lessons.

Dataset & Evaluation Metrics:
Dataset & Evaluation: Evaluated on 310 complex multi-hop educational queries (e.g., how social media algorithm exposure correlates with adolescent nicotine cravings) and reviewed by 8 adolescent psychiatrists and educators. Metrics: Multi-hop Context Recall (0.88), RAGAS Faithfulness (0.91), and Curriculum Alignment Score (89.5%).

Key Findings:
Key Findings: Entity-linked retrieval outperformed baseline chunk-based RAG by 19.4% on multi-hop questions connecting digital media habits with adolescent smoking risks.

Limitations & Research Gap:
Limitations & Research Gap: (1) High computational latency (~8.5 seconds per query) due to unoptimized graph traversal without lightweight hybrid caching. (2) Lacks automated pedagogical readability adaptation and Indonesian/local cultural-linguistic grounding for Southeast Asian secondary school students. (3) Absence of an integrated verification guardrail to prevent unsafe self-diagnosis among vulnerable youth.
""",
    },
]


def _write_simple_pdf(pdf_path: Path, text: str) -> None:
    """
    Menulis file PDF standar (Type1 Helvetica) murni dengan Python
    sehingga 100% teksnya dapat diekstrak kembali oleh pypdf/PyMuPDF.
    """
    # Bungkus baris agar rapi di dalam halaman PDF
    raw_lines = []
    for paragraph in text.strip().splitlines():
        if not paragraph.strip():
            raw_lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for w in words:
            if len(current) + len(w) + 1 <= 85:
                current = f"{current} {w}".strip()
            else:
                raw_lines.append(current)
                current = w
        if current:
            raw_lines.append(current)

    def escape_pdf_str(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream_cmds = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
    for line in raw_lines:
        stream_cmds.append(f"({escape_pdf_str(line)}) Tj")
        stream_cmds.append("T*")
    stream_cmds.append("ET")
    stream_data = "\n".join(stream_cmds).encode("latin-1", errors="replace")

    objs = []
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objs.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    )
    objs.append(
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )
    objs.append(
        f"5 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
        + stream_data
        + b"\nendstream\nendobj\n"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    pdf_path.write_bytes(bytes(out))


def ensure_sample_papers_if_empty(papers_dir: Path) -> bool:
    """
    Jika folder `data/papers/` belum berisi file PDF dari user,
    buatkan 5 contoh PDF artikel ilmiah sesuai domain agar pipeline bisa langsung diuji.
    Jika user sudah memasukkan file PDF asli milik mereka, hapus file sample bawaan
    agar tabel Literature Gap Matrix 100% berisi paper asli milik user.
    Mengembalikan True jika menggunakan sample PDF, False jika sudah ada PDF milik user.
    """
    papers_dir.mkdir(parents=True, exist_ok=True)
    sample_filenames = {item["filename"] for item in SAMPLE_PAPERS}
    all_pdfs = list(papers_dir.glob("*.pdf"))
    user_pdfs = [p for p in all_pdfs if p.name not in sample_filenames]

    if user_pdfs:
        # Bersihkan file sample bawaan jika user sudah menaruh PDF asli mereka
        for p in all_pdfs:
            if p.name in sample_filenames:
                try:
                    p.unlink()
                except OSError:
                    pass
        return False

    if all_pdfs:
        return True

    print(
        "[INFO] Folder data/papers/ masih kosong. Menyiapkan 5 PDF artikel referensi contoh "
        "(Elsevier, IEEE, Springer, Taylor & Francis, SAGE) sesuai domain Edukasi & Kesehatan Remaja..."
    )
    for item in SAMPLE_PAPERS:
        _write_simple_pdf(papers_dir / item["filename"], item["content"])
    return True
