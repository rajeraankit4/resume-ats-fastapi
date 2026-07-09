from dotenv import load_dotenv

load_dotenv()
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from resume_ats.rag import run_rag_analysis_from_retrieval

from resume_ats import (
    DOMAINS,
    DATABASE_URL,
    extract_and_store_jds_from_zip,
    fetch_jds_by_domain,
    get_db_connection,
    get_model,
    match_resume_all_domains,
)


app = FastAPI(
    title="Resume ATS Matcher",
    description="Match uploaded resumes against JD documents using domain-aware ATS scoring.",
    version="1.0.0",
)


APP_VERSION = os.getenv("APP_VERSION", "unknown")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _save_upload(upload: UploadFile, folder: str) -> str:
    filename = os.path.basename(upload.filename or "upload.bin")
    target_path = os.path.join(folder, filename)
    with open(target_path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)
    upload.file.seek(0)
    return target_path


@app.get("/health")
def health():
    model = get_model()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": "sentence-transformer" if model is not None else "model not available",
        "domains": list(DOMAINS.keys()),
    }


@app.get("/debug/state")
def debug_state():
    model = get_model()
    conn, cursor = get_db_connection()
    db_connected = conn is not None
    if cursor is not None:
        cursor.close()
    if conn is not None:
        conn.close()

    domain_counts = {}
    for domain in DOMAINS:
        try:
            domain_counts[domain] = len(fetch_jds_by_domain(domain))
        except Exception as exc:
            domain_counts[domain] = f"error: {exc}"

    return {
        "database_configured": bool(DATABASE_URL),
        "database_connected": db_connected,
        "model": "sentence-transformer" if model is not None else "model not available",
        "domains": list(DOMAINS.keys()),
        "domain_counts": domain_counts,
    }


@app.get("/domains")
def list_domains():
    return [
        {"name": name, "description": description}
        for name, description in DOMAINS.items()
    ]


@app.post("/jds/upload-zip")
async def upload_jd_zip(
    zip_file: UploadFile = File(...),
):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="model not available")

    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured. Set it in your environment or .env file before ingesting JDs.",
        )
    if not zip_file.filename or not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="zip_file must be a .zip archive")

    temp_dir = tempfile.mkdtemp(prefix="resume-ats-jd-zip-")
    zip_path = os.path.join(temp_dir, os.path.basename(zip_file.filename))
    try:
        with open(zip_path, "wb") as out_file:
            shutil.copyfileobj(zip_file.file, out_file)
        zip_file.file.seek(0)

        extract_and_store_jds_from_zip(zip_path)
        return {
            "status": "ok",
            "mode": "zip",
            "message": "JD zip processed and stored.",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/match/all-domains")
async def match_all_domains(
    resume: UploadFile = File(...),
):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="model not available")

    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured. The matcher needs the JD database to score resumes.",
        )

    temp_dir = tempfile.mkdtemp(prefix="resume-ats-")
    try:
        resume_path = _save_upload(resume, temp_dir)
        result = match_resume_all_domains(resume_path)

        ats_results = result["summary_rows"]

        filtered = [
            {
                "domain": r["domain"],
                "domain_fit": r["domain_fit"],
                "overall_score": r["overall_score"]
            }
            for r in ats_results
        ]

        rag_response = run_rag_analysis_from_retrieval(
            result["resume_text"],
            result["retrieved_jds"],
        )
        return {
            "status": "ok",
            "results": filtered,
            "ai_analysis": rag_response,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
