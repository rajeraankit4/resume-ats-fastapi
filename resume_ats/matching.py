from typing import Dict, List, Optional, Set

import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import (
    DOMAIN_BUCKET_WEIGHT,
    DOMAIN_FIT_WEIGHT,
    DOMAIN_SKILL_TERMS,
    DOMAINS,
    SEMANTIC_RELEVANCE_GATE,
    WEIGHT_KEYWORD,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL,
)
from .embeddings import build_reason, compute_resume_domain_fit, extract_skills, get_model
from .jd_store import fetch_jds_by_domain, fetch_jds_from_folder
from .text_utils import get_text_from_file, parse_embedding


def compute_semantic_scores(
    resume_text: str,
    jds: List[Dict],
    embedder: Optional[SentenceTransformer],
) -> List[float]:
    if embedder is not None:
        resume_vec = embedder.encode(resume_text, normalize_embeddings=True).reshape(1, -1)
        scores = []
        for jd in jds:
            jd_vec = parse_embedding(jd.get("embedding"))
            if jd_vec is None:
                jd_vec = embedder.encode(jd["cleaned_text"], normalize_embeddings=True).reshape(1, -1)
            scores.append(float(cosine_similarity(resume_vec, jd_vec)[0][0]))
        return scores

    corpus = [resume_text] + [jd["cleaned_text"] or jd["jd_text"] or jd["file_name"] for jd in jds]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
    resume_vec = matrix[0]
    return [float(cosine_similarity(resume_vec, matrix[i + 1])[0][0]) for i in range(len(jds))]


def score_skill_overlap(resume_skills: Set[str], jd_skills: Set[str]) -> float:
    if not jd_skills:
        return 0.0
    overlap = resume_skills & jd_skills
    precision = len(overlap) / len(jd_skills)
    recall = len(overlap) / max(len(resume_skills), 1)
    return min(1.0, 0.7 * precision + 0.3 * recall)


def score_keyword_density(resume_text: str, jd_text: str) -> float:
    stopwords = {
        "with", "that", "this", "have", "will", "from", "they", "been", "your", "team",
        "work", "role", "able", "good", "also", "must", "strong", "well", "over", "into",
        "should", "would", "could", "their", "about", "which", "after", "other", "some", "such",
    }
    jd_words = {w for w in re.findall(r"\b[a-z]{4,}\b", jd_text.lower()) if w not in stopwords}
    if not jd_words:
        return 0.0
    resume_lower = resume_text.lower()
    hits = sum(1 for w in jd_words if re.search(r"\b" + re.escape(w) + r"\b", resume_lower))
    return min(1.0, hits / len(jd_words))


def compute_ats_score(semantic: float, skill_overlap: float, keyword_density: float) -> float:
    raw = WEIGHT_SEMANTIC * semantic + WEIGHT_SKILL * skill_overlap + WEIGHT_KEYWORD * keyword_density
    return max(0.0, min(1.0, raw))


def jd_matches_domain(domain: str, jd: Dict) -> bool:
    stored_domain = jd.get("domain")
    stored_conf = jd.get("domain_confidence") or 0.0
    if stored_domain == domain and stored_conf >= 0.55:
        return True
    if jd.get("domain_secondary") == domain and stored_conf >= 0.35:
        return True

    combined = f"{jd['file_name']}\n{jd['cleaned_text'] or jd['jd_text']}"
    counts = {name: len(extract_skills(combined, DOMAIN_SKILL_TERMS[name])) for name in DOMAIN_SKILL_TERMS}
    best_domain = max(counts, key=counts.get)

    if domain == "Software":
        return counts["Software"] >= 2 or (best_domain == "Software" and counts["Software"] >= 1)
    if domain == "Analyst":
        return counts["Analyst"] >= 2 or (best_domain == "Analyst" and counts["Analyst"] >= 1)
    return counts["Core_NonTech"] >= 2 or (
        best_domain == "Core_NonTech" and counts["Core_NonTech"] >= max(counts["Software"], counts["Analyst"])
    )


def score_resume_against_jds(
    resume_text: str,
    domain: str,
    jds: List[Dict],
    top_n: int = 10,
    embedder: Optional[SentenceTransformer] = None,
) -> List[Dict]:
    if domain not in DOMAINS or not resume_text or not jds:
        return []

    embedder = embedder if embedder is not None else get_model()
    skill_terms = DOMAIN_SKILL_TERMS[domain]
    resume_skills = extract_skills(resume_text, skill_terms)
    semantic_scores = compute_semantic_scores(resume_text, jds, embedder)
    results = []

    for jd, semantic in zip(jds, semantic_scores):
        if not jd_matches_domain(domain, jd):
            continue
        if semantic < SEMANTIC_RELEVANCE_GATE:
            continue

        jd_text = jd["jd_text"]
        cleaned = jd["cleaned_text"] or jd_text
        jd_skills = extract_skills(f"{jd['file_name']}\n{cleaned}\n{jd_text}", skill_terms)
        skill_overlap = score_skill_overlap(resume_skills, jd_skills)
        keyword_density = score_keyword_density(resume_text, jd_text)
        ats = compute_ats_score(semantic, skill_overlap, keyword_density)
        overlap_terms = resume_skills & jd_skills

        results.append(
            {
                "file_name": jd["file_name"],
                "ats_score": round(ats * 100, 1),
                "semantic": round(semantic * 100, 1),
                "skill_overlap": round(skill_overlap * 100, 1),
                "keyword_density": round(keyword_density * 100, 1),
                "matched_skills": sorted(overlap_terms),
                "reason": build_reason(overlap_terms, semantic, skill_overlap, keyword_density),
            }
        )

    results.sort(key=lambda item: item["ats_score"], reverse=True)
    return results[:top_n]


def match_resume(
    resume_path: str,
    domain: str,
    top_n: int = 10,
    jd_folder: str = "./extracted/JDs",
    verbose: bool = True,
) -> List[Dict]:
    if domain not in DOMAINS:
        if verbose:
            print(f"Unknown domain '{domain}'. Choose from: {list(DOMAINS.keys())}")
        return []

    resume_text = get_text_from_file(resume_path)
    if not resume_text:
        if verbose:
            print("ERROR: Could not extract text from resume.")
        return []

    jds = fetch_jds_by_domain(domain)
    if not jds:
        jds = fetch_jds_from_folder(jd_folder)

    return score_resume_against_jds(resume_text, domain, jds, top_n=top_n)


def match_resume_all_domains(resume_path: str, top_n_per_domain: int = 3) -> List[Dict]:
    resume_text = get_text_from_file(resume_path)
    embedder = get_model()
    if not resume_text:
        return []

    summary_rows = []
    for domain, label in DOMAINS.items():
        results = match_resume(resume_path, domain, top_n=top_n_per_domain, verbose=False)
        domain_fit = round(compute_resume_domain_fit(resume_text, domain, embedder) * 100, 1)
        if results:
            top_score = results[0]["ats_score"]
            top_file = results[0]["file_name"]
            top_matches = results[: min(3, len(results))]
            avg_top = round(sum(item["ats_score"] for item in top_matches) / len(top_matches), 1)
            hits = len(results)
        else:
            top_score = 0.0
            avg_top = 0.0
            top_file = "-"
            hits = 0

        overall_score = round((DOMAIN_FIT_WEIGHT * domain_fit) + (DOMAIN_BUCKET_WEIGHT * avg_top), 1)
        summary_rows.append(
            {
                "domain": domain,
                "label": label,
                "domain_fit": domain_fit,
                "top_score": top_score,
                "avg_top": avg_top,
                "overall_score": overall_score,
                "hits": hits,
                "top_file": top_file,
            }
        )

    summary_rows.sort(key=lambda item: (item["overall_score"], item["domain_fit"], item["avg_top"]), reverse=True)
    return summary_rows


def match_resume_against_jds(
    resume_path: str,
    jds: List[Dict],
    domain: str,
    top_n: int = 10,
) -> List[Dict]:
    resume_text = get_text_from_file(resume_path)
    return score_resume_against_jds(resume_text, domain, jds, top_n=top_n)


def match_resume_all_domains_against_jds(
    resume_path: str,
    jds: List[Dict],
    top_n_per_domain: int = 3,
) -> List[Dict]:
    resume_text = get_text_from_file(resume_path)
    embedder = get_model()
    if not resume_text:
        return []

    summary_rows = []
    for domain, label in DOMAINS.items():
        results = score_resume_against_jds(resume_text, domain, jds, top_n=top_n_per_domain)
        domain_fit = round(compute_resume_domain_fit(resume_text, domain, embedder) * 100, 1)
        if results:
            top_score = results[0]["ats_score"]
            top_file = results[0]["file_name"]
            top_matches = results[: min(3, len(results))]
            avg_top = round(sum(item["ats_score"] for item in top_matches) / len(top_matches), 1)
            hits = len(results)
        else:
            top_score = 0.0
            avg_top = 0.0
            top_file = "-"
            hits = 0

        overall_score = round((DOMAIN_FIT_WEIGHT * domain_fit) + (DOMAIN_BUCKET_WEIGHT * avg_top), 1)
        summary_rows.append(
            {
                "domain": domain,
                "label": label,
                "domain_fit": domain_fit,
                "top_score": top_score,
                "avg_top": avg_top,
                "overall_score": overall_score,
                "hits": hits,
                "top_file": top_file,
            }
        )

    summary_rows.sort(key=lambda item: (item["overall_score"], item["domain_fit"], item["avg_top"]), reverse=True)
    return summary_rows

