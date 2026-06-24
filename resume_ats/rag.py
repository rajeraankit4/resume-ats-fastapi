from typing import Dict, List

from .embeddings import get_model
from .matching import retrieve_top_jds_global
from .groq_service import generate_response

RAG_TOP_K = 3


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
        You are an expert resume reviewer.

        Resume:
        {resume_text}

        Relevant Job Descriptions:
        {rag_context}

        Task:
        Identify only the most important improvements required in the resume based on the job descriptions.

        Return ONLY valid JSON in this format:

        {{
        "suggestions": [
            "",
            "",
            "",
            "",
            ""
        ]
        }}

        Rules:
        - Maximum 5 suggestions.
        - Each suggestion must be one short sentence.
        - Focus on missing skills, resume gaps, and improvement areas.
        - Do not mention strengths.
        - Do not mention suitable roles.
        - Do not explain reasoning.
        - Do not use markdown.
        - Return JSON only.
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

def run_rag_analysis(
    resume_text: str,
) -> str:

    prompt = prepare_rag_prompt(
        resume_text
    )

    return generate_response(
        prompt
    )

def run_rag_analysis_from_retrieval(
    resume_text: str,
    retrieved_jds: List[Dict],
) -> str:

    rag_context = build_rag_context(
        retrieved_jds[:RAG_TOP_K]
    )

    prompt = build_resume_analysis_prompt(
        resume_text,
        rag_context,
    )

    return generate_response(prompt)