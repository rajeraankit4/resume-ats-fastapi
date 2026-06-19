import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from resume_ats import (
    DOMAINS,
    DATABASE_URL,
    extract_and_store_jds_from_zip,
    get_model,
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


@app.post("/jds/upload-zip")
async def upload_jd_zip(
    zip_file: UploadFile = File(...),
):
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
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured. The matcher needs the JD database to score resumes.",
        )

    temp_dir = tempfile.mkdtemp(prefix="resume-ats-")
    try:
        resume_path = _save_upload(resume, temp_dir)
        results = match_resume_all_domains(resume_path)
        return {
            "status": "ok",
            "results": results,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
