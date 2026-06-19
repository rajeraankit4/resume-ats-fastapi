# Classifies Job Descriptions into domains (Software, Analyst, Core_NonTech)
# using semantic similarity, skill matching, and rule-based scoring.

import re
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import DOMAIN_INTENT_PATTERNS, DOMAIN_PROTOTYPES, DOMAIN_SKILL_TERMS
from .text_utils import normalize_text

_model = None
_model_load_attempted = False
_cached_domain_embeddings: Optional[np.ndarray] = None


def get_model() -> Optional[SentenceTransformer]:
    global _model, _model_load_attempted
    if _model is not None:
        return _model
    if _model_load_attempted:
        return None
    _model_load_attempted = True
    try:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model
    except Exception as e:
        print(f"model not available: {e}")
        return None


def extract_skills(text: str, term_bank: set[str]) -> set[str]:
    lowered = normalize_text(text)
    found = set()
    for term in term_bank:
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            found.add(term)
    return found


def compute_domain_signal_counts(text: str) -> dict[str, int]:
    return {domain: len(extract_skills(text, DOMAIN_SKILL_TERMS[domain])) for domain in DOMAIN_SKILL_TERMS}


def has_software_role_signals(text: str) -> bool:
    lowered = normalize_text(text)
    role_patterns = [
        r"\bcs\s*/\s*it\b",
        r"\bcomputer science\b",
        r"\binformation technology\b",
        r"\bprogramming knowledge\b",
        r"\bdata structures\b",
        r"\balgorithms?\b",
        r"\bobject-oriented\b",
        r"\boops?\b",
        r"\bc/c\+\+\b",
        r"\bjava\b",
        r"\bpython\b",
    ]
    hits = sum(1 for pattern in role_patterns if re.search(pattern, lowered))
    return hits >= 3


def has_analyst_role_signals(text: str) -> bool:
    lowered = normalize_text(text)
    role_patterns = [
        r"\bdata analyst\b",
        r"\bbusiness analyst\b",
        r"\bresearch analyst\b",
        r"\bdata science\b",
        r"\bmachine learning\b",
        r"\bartificial intelligence\b",
        r"\bpower bi\b",
        r"\btableau\b",
        r"\bexcel\b",
        r"\bsql\b",
        r"\bpython\b",
        r"\bdashboard\b",
        r"\breporting\b",
        r"\bforecasting\b",
    ]
    hits = sum(1 for pattern in role_patterns if re.search(pattern, lowered))
    return hits >= 3


def has_core_nontech_role_signals(text: str) -> bool:
    lowered = normalize_text(text)
    role_patterns = [
        r"\bembedded\b",
        r"\bfirmware\b",
        r"\bhardware\b",
        r"\belectrical\b",
        r"\bmechanical\b",
        r"\bmanufacturing\b",
        r"\boperations\b",
        r"\bsupply chain\b",
        r"\bprocurement\b",
        r"\baccounting\b",
        r"\bfinance\b",
        r"\blogistics\b",
        r"\bhuman resources\b",
        r"\bcustomer service\b",
    ]
    hits = sum(1 for pattern in role_patterns if re.search(pattern, lowered))
    return hits >= 3


def _get_domain_embeddings(embedder: SentenceTransformer) -> np.ndarray:
    global _cached_domain_embeddings
    if _cached_domain_embeddings is None:
        _cached_domain_embeddings = embedder.encode(list(DOMAIN_PROTOTYPES.values()), normalize_embeddings=True)
    return _cached_domain_embeddings


def _classify_semantic(text: str, embedder: SentenceTransformer) -> Tuple[str, float, Optional[str]]:
    domain_keys = list(DOMAIN_PROTOTYPES.keys())
    jd_vec = embedder.encode(text, normalize_embeddings=True).reshape(1, -1)
    proto_vecs = _get_domain_embeddings(embedder)
    sims = cosine_similarity(jd_vec, proto_vecs)[0]
    sorted_idx = np.argsort(sims)[::-1]
    top_idx, second_idx = int(sorted_idx[0]), int(sorted_idx[1])
    top_score = float(sims[top_idx])
    second_score = float(sims[second_idx])
    secondary = domain_keys[second_idx] if second_score >= max(0.15, top_score * 0.80) else None
    return domain_keys[top_idx], top_score, secondary


def classify_domain(text: str, embedder: Optional[SentenceTransformer] = None) -> Tuple[str, float, Optional[str]]:
    if embedder is None:
        raise RuntimeError("model not available")

    semantic_domain, semantic_confidence, semantic_secondary = _classify_semantic(text, embedder)

    counts = compute_domain_signal_counts(text)
    combined_scores = {}
    software_signal_boost = has_software_role_signals(text)
    analyst_signal_boost = has_analyst_role_signals(text)
    core_signal_boost = has_core_nontech_role_signals(text)

    for domain in DOMAIN_SKILL_TERMS:
        term_component = min(1.0, counts[domain] / 6)
        semantic_component = semantic_confidence if domain == semantic_domain else 0.0
        score = (0.55 * semantic_component) + (0.45 * term_component)

        if software_signal_boost and domain == "Software":
            score += 0.18
        if analyst_signal_boost and domain == "Analyst":
            score += 0.16
        if core_signal_boost and domain == "Core_NonTech":
            score += 0.14

        if software_signal_boost and domain in {"Analyst", "Core_NonTech"}:
            score -= 0.06
        if analyst_signal_boost and domain in {"Software", "Core_NonTech"}:
            score -= 0.04
        if core_signal_boost and domain in {"Software", "Analyst"}:
            score -= 0.04

        if domain == "Software" and counts["Software"] >= counts["Core_NonTech"] + 2:
            score += 0.08
        if domain == "Analyst" and counts["Analyst"] >= counts["Core_NonTech"] + 1:
            score += 0.06
        if domain == "Core_NonTech" and counts["Core_NonTech"] >= counts["Software"] + counts["Analyst"]:
            score += 0.06
        combined_scores[domain] = min(1.0, score)

    sorted_domains = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    best_domain, best_score = sorted_domains[0]
    second_domain, second_score = sorted_domains[1]
    secondary = None
    if second_score >= max(0.18, best_score * 0.80):
        secondary = second_domain
    elif semantic_secondary in DOMAIN_SKILL_TERMS and semantic_secondary != best_domain:
        secondary = semantic_secondary
    return best_domain, float(best_score), secondary


def compute_resume_domain_fit(resume_text: str, domain: str, embedder: Optional[SentenceTransformer]) -> float:
    if embedder is None:
        raise RuntimeError("model not available")

    normalized_resume = normalize_text(resume_text)
    domain_skills = extract_skills(resume_text, DOMAIN_SKILL_TERMS[domain])
    target_component = min(1.0, len(domain_skills) / 6)
    core_component = len(extract_skills(resume_text, DOMAIN_SKILL_TERMS["Core_NonTech"]))
    software_component = len(extract_skills(resume_text, DOMAIN_SKILL_TERMS["Software"]))
    analyst_component = len(extract_skills(resume_text, DOMAIN_SKILL_TERMS["Analyst"]))
    intent_hits = sum(1 for pattern in DOMAIN_INTENT_PATTERNS[domain] if re.search(pattern, normalized_resume))
    intent_component = min(1.0, intent_hits / 4)

    resume_vec = embedder.encode(normalized_resume, normalize_embeddings=True).reshape(1, -1)
    domain_vec = embedder.encode(DOMAIN_PROTOTYPES[domain], normalize_embeddings=True).reshape(1, -1)
    semantic_component = float(cosine_similarity(resume_vec, domain_vec)[0][0])

    fit = (0.45 * target_component) + (0.25 * semantic_component) + (0.30 * intent_component)
    if domain == "Core_NonTech" and software_component >= 6 and analyst_component <= 2 and core_component <= 2:
        fit *= 0.20
    if domain == "Software" and software_component >= analyst_component + core_component:
        fit += 0.08
    if domain == "Software" and analyst_component >= 5 and analyst_component >= core_component + 3:
        fit *= 0.88
    if domain == "Analyst" and analyst_component >= core_component:
        fit += 0.04
    if domain == "Analyst" and analyst_component >= 5 and intent_hits >= 2:
        fit += 0.10
    return max(0.0, min(1.0, fit))


def build_reason(overlap: set[str], semantic: float, skill_score: float, keyword_score: float) -> str:
    parts = []
    if overlap:
        parts.append(f"matched skills: {', '.join(sorted(overlap)[:6])}")
    parts.append(
        "strong semantic match"
        if semantic >= 0.65
        else "moderate semantic match"
        if semantic >= 0.45
        else "low semantic similarity"
    )
    if skill_score < 0.20:
        parts.append("few domain skills in resume")
    if keyword_score < 0.20:
        parts.append("low JD keyword coverage")
    return "; ".join(parts)
