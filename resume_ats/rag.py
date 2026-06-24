from typing import Dict, List

from .embeddings import get_model
from .matching import retrieve_top_jds_global


RAG_TOP_K = 5


def retrieve_context_for_rag(
    resume_text: str,
    top_k: int = RAG_TOP_K,
) -> List[Dict]:
    """
    Uses the SAME retrieval pipeline as ATS.

    Returns top-k semantically relevant JDs.
    """

    embedder = get_model()

    if embedder is None:
        raise RuntimeError("model not available")

    return retrieve_top_jds_global(
        resume_text=resume_text,
        top_k=top_k,
        embedder=embedder,
    )


def build_rag_context(
    retrieved_jds: List[Dict],
) -> str:
    """
    Converts retrieved JDs into a compact text context
    that can be passed to an LLM.
    """

    sections = []

    for idx, jd in enumerate(retrieved_jds, start=1):

        jd_text = (
            jd.get("cleaned_text")
            or jd.get("jd_text")
            or ""
        )

        sections.append(
            f"""
JOB DESCRIPTION {idx}

File Name:
{jd.get("file_name", "Unknown")}

Semantic Match Score:
{jd.get("retrieval_score", 0.0):.3f}

Content:
{jd_text}
"""
        )

    return "\n\n".join(sections)


def build_resume_analysis_prompt(
    resume_text: str,
    rag_context: str,
) -> str:
    """
    Creates the final prompt for the LLM.
    """

    return f"""
You are an expert ATS evaluator, recruiter, and career advisor.

Analyze the resume against the retrieved job descriptions.

========================
RESUME
========================

{resume_text}

========================
RETRIEVED JOB DESCRIPTIONS
========================

{rag_context}

========================
TASKS
========================

1. Identify the most suitable job roles.
2. Explain why the resume matches these roles.
3. Identify missing skills and skill gaps.
4. Suggest resume improvements.
5. Recommend learning areas to improve employability.
6. Provide an overall assessment of job readiness.

Return a structured report.
"""


def prepare_rag_prompt(
    resume_text: str,
) -> str:
    """
    Main RAG entry point.

    Resume
        ↓
    Retrieval
        ↓
    Context Building
        ↓
    Prompt
    """

    retrieved_jds = retrieve_context_for_rag(
        resume_text=resume_text,
        top_k=RAG_TOP_K,
    )

    rag_context = build_rag_context(
        retrieved_jds
    )

    return build_resume_analysis_prompt(
        resume_text=resume_text,
        rag_context=rag_context,
    )