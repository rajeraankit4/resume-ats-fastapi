# Resume ATS + RAG Analysis System

## Overview

This project is an AI-powered Resume Analysis System that evaluates resumes against a database of Job Descriptions (JDs).

The system combines traditional ATS-style scoring with Retrieval-Augmented Generation (RAG) to provide both quantitative scores and AI-generated improvement suggestions.

The goal is to help candidates understand:

* Which career domain best matches their profile
* How well their resume aligns with industry job descriptions
* What skills or keywords are missing
* How they can improve their resume

---

# System Architecture

Resume Upload
↓
Text Extraction
↓
Embedding Generation
↓
Global Semantic JD Retrieval
↓
Top Matching JDs
├── ATS Scoring Pipeline
└── RAG Suggestion Pipeline
↓
Final API Response

---

# Job Description Ingestion

Job Descriptions are uploaded as a ZIP file.

During ingestion:

1. Text is extracted from PDF/DOCX files.
2. Content is cleaned and normalized.
3. Embeddings are generated using Sentence Transformers.
4. Domain is identified.
5. JD metadata and embeddings are stored in PostgreSQL.

Stored Information:

* File Name
* JD Text
* Cleaned Text
* Embedding Vector
* Domain
* Domain Confidence

---

# Global Retrieval Strategy

When a resume is uploaded:

The system does **one semantic retrieval operation** across all stored JDs.

Resume
↓
Compare against all JD embeddings
↓
Retrieve Top 50 matching JDs

This retrieval happens only once.

The retrieved JDs are reused by:

* ATS Scoring
* RAG Analysis

This ensures consistency and reduces computation cost.

---

# ATS Scoring Logic

After retrieval, JDs are grouped by domain:

* Software
* Analyst
* Core_NonTech

For each domain, ATS scores are calculated using:

### 1. Semantic Similarity

Measures how closely the resume matches a JD using embedding similarity.

### 2. Skill Overlap

Measures overlap between resume skills and domain-specific JD skills.

### 3. Keyword Coverage

Measures how many important JD keywords appear in the resume.

### ATS Formula

ATS Score =
(Weighted Semantic Similarity)
+
(Weighted Skill Overlap)
+
(Weighted Keyword Coverage)

Only highly relevant JDs are considered during scoring.

---

# Domain Fit Calculation

Apart from ATS scoring, the system calculates a Domain Fit score.

This determines how strongly the resume aligns with each career domain.

Example:

* Analyst → 99.6%
* Software → 74.0%
* Core_NonTech → 7.2%

---

# Overall Score

Final domain ranking is calculated using:

Overall Score =
(Domain Fit Weight × Domain Fit)
+
(ATS Weight × Average ATS Score)

This allows the system to recommend the most suitable career domain.

---

# RAG Pipeline

The project also includes Retrieval-Augmented Generation (RAG).

Instead of sending the entire JD database to the LLM, only the top retrieved JDs are used.

Resume
↓
Top Matching JDs
↓
Context Builder
↓
LLM Prompt
↓
AI Suggestions

The retrieved JDs act as external knowledge for the language model.

---

# AI Suggestions

The retrieved context is sent to a Groq-hosted LLM.

The model analyzes:

* Resume content
* Retrieved job descriptions

and generates concise improvement suggestions such as:

* Missing technical skills
* Resume enhancement recommendations
* Industry-relevant keywords to include

Example:

{
"suggestions": [
"Add AWS experience.",
"Highlight machine learning projects.",
"Include Docker and containerization skills.",
"Quantify project achievements."
]
}

---

# Why This Is a RAG System

The project follows the standard RAG workflow:

Retrieve
+
Augment Prompt
+
Generate

1. Relevant JDs are retrieved from a database.
2. Retrieved JDs are injected into the prompt.
3. The LLM generates suggestions using the retrieved context.

Since external knowledge is retrieved before generation, the system qualifies as a Retrieval-Augmented Generation (RAG) application.

---

# Technologies Used

Backend:

* FastAPI
* Python

Database:

* PostgreSQL

Embeddings:

* Sentence Transformers
* all-MiniLM-L6-v2

LLM:

* Groq

Document Processing:

* PDF Parsing
* DOCX Parsing

---

# API Endpoints

### POST /jds/upload-zip

Uploads and stores Job Descriptions.

### POST /match/all-domains

Analyzes a resume and returns:

* Domain-wise ATS scores
* Domain fit scores
* Overall scores
* AI-generated improvement suggestions

---

# Key Design Decisions

* Single retrieval shared by ATS and RAG
* Domain-aware resume evaluation
* Embedding-based semantic matching
* Lightweight RAG using only top retrieved JDs
* Structured AI output for frontend integration
* Cost-efficient and scalable architecture
