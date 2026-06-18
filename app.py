import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from resume_ats import (
    DOMAINS,
    DATABASE_URL,
    extract_and_store_jds,
    get_model,
    match_resume as match_resume_cli,
    match_resume_all_domains,
)


app = FastAPI(
    title="Resume ATS Matcher",
    description="Match uploaded resumes against JD documents using domain-aware ATS scoring.",
    version="1.0.0",
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
        "model": "sentence-transformer" if model is not None else "tf-idf-fallback",
        "domains": list(DOMAINS.keys()),
    }


@app.get("/domains")
def list_domains():
    return [
        {"name": name, "description": description}
        for name, description in DOMAINS.items()
    ]


@app.post("/jds/ingest")
async def ingest_jds(
    folder_path: str = Query("./extracted/JDs"),
):
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured. Set it in your environment or .env file before ingesting JDs.",
        )

    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    extract_and_store_jds(folder_path)
    return {"status": "ok", "mode": "folder", "folder_path": folder_path, "message": "JD files processed and stored."}


@app.post("/match")
async def match_resume(
    resume: UploadFile = File(...),
    domain: str = Query(..., description="One of: Software, Analyst, Core_NonTech"),
    top_n: int = Query(10, ge=1, le=50),
):
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured. The matcher needs the JD database to score resumes.",
        )

    if domain not in DOMAINS:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'. Use one of: {list(DOMAINS.keys())}")

    temp_dir = tempfile.mkdtemp(prefix="resume-ats-")
    try:
        resume_path = _save_upload(resume, temp_dir)
        results = match_resume_cli(resume_path, domain, top_n=top_n, verbose=False)
        return {"status": "ok", "domain": domain, "top_n": top_n, "results": results}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/match/all-domains")
async def match_all_domains(
    resume: UploadFile = File(...),
    top_n_per_domain: int = Query(3, ge=1, le=10),
):
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured. The matcher needs the JD database to score resumes.",
        )

    temp_dir = tempfile.mkdtemp(prefix="resume-ats-")
    try:
        resume_path = _save_upload(resume, temp_dir)
        results = match_resume_all_domains(resume_path, top_n_per_domain=top_n_per_domain)
        return {
            "status": "ok",
            "top_n_per_domain": top_n_per_domain,
            "results": results,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
