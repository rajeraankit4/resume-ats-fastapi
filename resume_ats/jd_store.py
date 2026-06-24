# Extracts Job Descriptions from PDF/DOCX files, filters relevant content,
# classifies domains, generates embeddings, and stores results in PostgreSQL.

import os
import re
import zipfile
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

from .config import DATABASE_URL
from .constants import JD_PATTERNS, NON_JD_PATTERNS, PRIORITY_PATTERNS, RELEVANCE_QUERIES
from .embeddings import classify_domain, get_model
from .text_utils import get_text_from_file, normalize_text

_db_connection_failed = False


def get_db_connection():
    global _db_connection_failed
    if not DATABASE_URL or _db_connection_failed:
        return None, None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print(f"DB connection failed: {e}")
        _db_connection_failed = True
        return None, None


def is_probable_jd(file_name: str, text: str) -> bool:
    combined = normalize_text(f"{file_name} {text}")
    has_jd_signals = any(re.search(pattern, combined) for pattern in JD_PATTERNS)
    has_non_jd_signals = any(re.search(pattern, combined) for pattern in NON_JD_PATTERNS)
    if has_non_jd_signals and not has_jd_signals:
        return False
    return has_jd_signals or not has_non_jd_signals


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_logical_chunks(text: str) -> List[str]:
    parts = re.split(r"\n{2,}|(?=\s*[\u2022\-\*])", text)
    return [normalize_space(p) for p in parts if len(normalize_space(p)) >= 35]


def chunk_is_priority(chunk: str) -> bool:
    lowered = chunk.lower()
    return any(re.search(pat, lowered) for pat in PRIORITY_PATTERNS)


def filter_relevant_chunks(chunks: List[str], embedder: Optional[SentenceTransformer]) -> List[Tuple[str, float]]:
    if not chunks:
        return []

    if embedder is None:
        raise RuntimeError("model not available")

    query_embs = embedder.encode(RELEVANCE_QUERIES, convert_to_tensor=True)
    chunk_embs = embedder.encode(chunks, convert_to_tensor=True)
    cos_scores = util.cos_sim(chunk_embs, query_embs)
    max_scores = cos_scores.max(dim=1).values.cpu().numpy()

    mean_score = float(np.mean(max_scores))
    std_score = float(np.std(max_scores))
    dynamic_threshold = mean_score + (0.45 * std_score)

    scored: List[Tuple[str, float]] = []
    seen_texts: set = set()

    for i, raw_score in enumerate(max_scores):
        chunk = chunks[i]
        if chunk in seen_texts:
            continue

        boost = 0.0
        if chunk_is_priority(chunk):
            boost += 0.08
        if chunk.endswith(":") and len(chunk) < 50:
            boost += 0.08

        final = float(raw_score) + boost
        keep = final >= dynamic_threshold or (chunk_is_priority(chunk) and final >= mean_score)
        if keep:
            scored.append((chunk, round(final, 3)))
            seen_texts.add(chunk)

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    texts = [t for t, _ in scored]

    if embedder is None:
        raise RuntimeError("model not available")

    embs = embedder.encode(texts, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embs, embs).cpu().numpy()

    skipped: set = set()
    final_chunks: List[Tuple[str, float]] = []
    for i in range(len(texts)):
        if i in skipped:
            continue
        final_chunks.append(scored[i])
        for j in range(i + 1, len(texts)):
            if sim_matrix[i][j] > 0.95:
                skipped.add(j)

    return final_chunks


def ensure_table(cursor, conn) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id                 SERIAL PRIMARY KEY,
            file_name          VARCHAR(255) UNIQUE NOT NULL,
            jd_text            TEXT,
            cleaned_text       TEXT,
            embedding          FLOAT[],
            domain             VARCHAR(60),
            domain_confidence  FLOAT,
            domain_secondary   VARCHAR(60),
            created_at         TIMESTAMP DEFAULT NOW()
        )
        """
    )
    conn.commit()
    for col, col_type in [
        ("domain", "VARCHAR(60)"),
        ("domain_confidence", "FLOAT"),
        ("domain_secondary", "VARCHAR(60)"),
        ("created_at", "TIMESTAMP DEFAULT NOW()"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS {col} {col_type}")
            conn.commit()
        except Exception:
            conn.rollback()
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jd_domain ON job_descriptions(domain)")
        conn.commit()
    except Exception:
        conn.rollback()


def save_to_database(
    cursor,
    conn,
    file_name: str,
    jd_text: str,
    cleaned_text: str,
    embedding: List[float],
    domain: str,
    domain_confidence: float,
    domain_secondary: Optional[str],
) -> None:
    try:
        cursor.execute(
            """
            INSERT INTO job_descriptions
                (file_name, jd_text, cleaned_text, embedding,
                 domain, domain_confidence, domain_secondary)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_name) DO UPDATE
                SET jd_text           = EXCLUDED.jd_text,
                    cleaned_text      = EXCLUDED.cleaned_text,
                    embedding         = EXCLUDED.embedding,
                    domain            = EXCLUDED.domain,
                    domain_confidence = EXCLUDED.domain_confidence,
                    domain_secondary  = EXCLUDED.domain_secondary
            """,
            (
                file_name,
                jd_text,
                cleaned_text,
                embedding if embedding else None,
                domain,
                domain_confidence,
                domain_secondary,
            ),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"DB insert failed for {file_name}: {e}")


def fetch_jds_by_domain(domain: str) -> List[Dict]:
    conn, cursor = get_db_connection()
    if not conn:
        return []
    try:
        cursor.execute(
            """
            SELECT file_name, jd_text, cleaned_text, embedding,
                   domain, domain_confidence, domain_secondary
            FROM job_descriptions
            WHERE domain = %s OR domain_secondary = %s
            ORDER BY
                CASE WHEN domain = %s THEN 0 ELSE 1 END,
                domain_confidence DESC
            """,
            (domain, domain, domain),
        )
        rows = cursor.fetchall()
        return [
            {
                "file_name": row[0],
                "jd_text": row[1] or "",
                "cleaned_text": row[2] or row[1] or "",
                "embedding": row[3],
                "domain": row[4],
                "domain_confidence": row[5],
                "domain_secondary": row[6],
            }
            for row in rows
        ]
    except Exception as e:
        print(f"DB fetch error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def fetch_all_jds() -> List[Dict]:
    conn, cursor = get_db_connection()

    if not conn:
        return []

    try:
        cursor.execute(
            """
            SELECT file_name,
                   jd_text,
                   cleaned_text,
                   embedding,
                   domain,
                   domain_confidence,
                   domain_secondary
            FROM job_descriptions
            """
        )

        rows = cursor.fetchall()

        return [
            {
                "file_name": row[0],
                "jd_text": row[1] or "",
                "cleaned_text": row[2] or row[1] or "",
                "embedding": row[3],
                "domain": row[4],
                "domain_confidence": row[5],
                "domain_secondary": row[6],
            }
            for row in rows
        ]

    except Exception as e:
        print(f"DB fetch error: {e}")
        return []

    finally:
        cursor.close()
        conn.close()

def fetch_jds_from_folder(folder_path: str) -> List[Dict]:
    if not os.path.isdir(folder_path):
        return []
    records = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith("~$"):
            continue
        if not file_name.lower().endswith((".pdf", ".docx")):
            continue
        jd_text = get_text_from_file(os.path.join(folder_path, file_name))
        if not jd_text or not is_probable_jd(file_name, jd_text):
            continue
        title_hint = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ")
        records.append(
            {
                "file_name": file_name,
                "jd_text": jd_text,
                "cleaned_text": f"{title_hint}\n{jd_text}",
                "embedding": None,
                "domain": None,
                "domain_confidence": None,
                "domain_secondary": None,
            }
        )
    return records


def extract_and_store_jds(folder_path: str) -> None:
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    conn, cursor = get_db_connection()
    if not conn:
        print("DB connection unavailable")
        return

    ensure_table(cursor, conn)
    embedder = get_model()
    if embedder is None:
        print("model not available")
        return
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        print(f"No PDF or DOCX files found in {folder_path}")
        return

    try:
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            full_text = get_text_from_file(file_path)
            if not full_text or not is_probable_jd(file_name, full_text):
                continue

            title_hint = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ")
            chunks = [title_hint] + split_into_logical_chunks(full_text)
            relevant = filter_relevant_chunks(chunks, embedder)
            if not relevant:
                continue

            cleaned_text = "\n".join(text for text, _ in relevant)
            classification_input = f"{title_hint}\n{cleaned_text}"
            domain, confidence, secondary = classify_domain(classification_input, embedder=embedder)

            if embedder is not None:
                final_embedding: List[float] = embedder.encode(cleaned_text).tolist()
            else:
                final_embedding = []

            save_to_database(
                cursor=cursor,
                conn=conn,
                file_name=file_name,
                jd_text=full_text,
                cleaned_text=cleaned_text,
                embedding=final_embedding,
                domain=domain,
                domain_confidence=confidence,
                domain_secondary=secondary,
            )
    finally:
        cursor.close()
        conn.close()


def extract_and_store_jds_from_zip(zip_path: str) -> None:
    if not os.path.isfile(zip_path):
        print(f"Zip file not found: {zip_path}")
        return

    with tempfile.TemporaryDirectory(prefix="resume-ats-zip-") as temp_root:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                member_name = member.filename
                normalized = os.path.normpath(member_name)
                if os.path.isabs(member_name) or normalized.startswith(".."):
                    raise ValueError(f"Unsafe path inside zip: {member_name}")

                if not member_name.lower().endswith((".pdf", ".docx")):
                    continue

                target_path = os.path.join(temp_root, os.path.basename(member_name))
                with archive.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

        extract_and_store_jds(temp_root)
