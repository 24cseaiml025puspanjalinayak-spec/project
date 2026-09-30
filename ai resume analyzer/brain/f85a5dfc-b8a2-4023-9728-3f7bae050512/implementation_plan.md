# Implementation Plan: AI-Based Resume Analyzer (Complete Learning Project)

Build a complete, production-grade, educational **AI-Based Resume Analyzer** project in Python. This project is designed to teach both classical Natural Language Processing (NLP) techniques (tokenization, TF-IDF vectorization, cosine similarity, regex extraction, ATS scoring heuristics) and modern Generative AI capabilities (Google Gemini `gemini-3.8-flash` via `google-genai` SDK for semantic feedback, bullet rewriting, and interview prep).

The project will be housed in `C:\Users\Dell\.gemini\antigravity\scratch\ai-resume-analyzer`.

---

## User Review Required

> [!IMPORTANT]
> **Key Architecture Decision: Hybrid Architecture (Offline NLP + Optional Gemini LLM)**
> - The application will function 100% offline out-of-the-box using built-in NLP parsers, TF-IDF cosine similarity, regex extraction, and ATS heuristic algorithms.
> - If the user provides a `GEMINI_API_KEY` (either in an `.env` file or directly in the UI), it automatically activates advanced Generative AI capabilities (bullet rewriting using the STAR method, semantic role-fit critique, and personalized interview questions).
> - UI is built using **Streamlit** for interactive, visual exploration with gauges, badges, and exportable reports, alongside a fast **CLI** for automated workflows.

---

## Open Questions

- Do you have a preferred UI framework? (We recommend **Streamlit** for learning and rapid portfolio building, but could also provide a FastAPI backend if desired).
- Do you already have a Gemini API Key? (If not, the project is designed so all core features work completely offline without one, and you can plug the key in anytime).

---

## Proposed Project Structure & Components

```
ai-resume-analyzer/
│
├── src/
│   ├── __init__.py
│   ├── extractor.py           # Text extraction from PDF, DOCX, TXT & section parser
│   ├── skills_db.py           # Comprehensive categorized skill taxonomy (1,000+ skills)
│   ├── ats_analyzer.py        # ATS scoring (action verbs, quantification, formatting)
│   ├── matcher.py             # TF-IDF + Cosine Similarity & skill gap analysis
│   ├── llm_analyzer.py        # Gemini 3.8 Flash integration (critique, STAR rewrite, Q&A)
│   └── report_generator.py    # Generates structured reports (Markdown, JSON)
│
├── samples/
│   ├── sample_resume_swe.txt            # Sample Software Engineer resume
│   ├── sample_resume_data_scientist.txt # Sample Data Scientist resume
│   ├── sample_jd_swe.txt                # Sample Software Engineer job description
│   └── sample_jd_data_scientist.txt     # Sample Data Scientist job description
│
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py      # Unit tests for text & section extraction
│   ├── test_matcher.py        # Unit tests for TF-IDF matching & skill gaps
│   └── test_ats_analyzer.py   # Unit tests for ATS scoring & action verb checks
│
├── app.py                     # Interactive Streamlit Web Application
├── cli.py                     # Command-line interface for terminal usage
├── requirements.txt           # Project dependencies (Streamlit, pypdf, python-docx, scikit-learn, google-genai, etc.)
├── .env.example               # Template for API keys
├── README.md                  # Project overview, installation, usage, and portfolio tips
└── LEARNING_GUIDE.md          # Comprehensive educational deep-dive into NLP, ATS, and LLMs
```

---

## Detailed Component Specifications

### 1. Document Extraction & Section Segmentation (`src/extractor.py`)
- Extraction support:
  - PDF: `pypdf` with fallback text cleanup.
  - DOCX: `python-docx` extracting paragraphs and table cells.
  - TXT / Raw Text.
- Section Segmentation: Intelligently detects and isolates sections:
  - Contact Information (Email, Phone, LinkedIn, GitHub, Location)
  - Professional Summary / Objective
  - Work Experience / Employment History
  - Education & Degrees
  - Skills & Technologies
  - Projects & Open Source
  - Certifications & Awards

### 2. Taxonomy & Skill Extraction Engine (`src/skills_db.py`)
- Rich taxonomy categorized across domains:
  - Programming Languages (Python, Java, TypeScript, C++, Go, Rust, etc.)
  - Web & Frameworks (React, Angular, Vue, FastAPI, Django, Spring Boot, etc.)
  - Cloud & Infrastructure (AWS, Azure, GCP, Kubernetes, Docker, Terraform, etc.)
  - Data & AI/ML (PyTorch, TensorFlow, Pandas, Scikit-Learn, Spark, SQL, BigQuery, etc.)
  - DevOps & Tools (Git, CI/CD, Linux, Jenkins, Jira, etc.)
  - Soft Skills (Leadership, Communication, Problem Solving, Agile, etc.)
- Normalization engine: Maps variations (e.g. `k8s` -> `Kubernetes`, `react.js` -> `React`, `node` -> `Node.js`).

### 3. ATS Health & Heuristics Analyzer (`src/ats_analyzer.py`)
- **ATS Score (0 - 100)** calculated using industry-standard checks:
  1. *Contact Info Completeness* (Email, Phone, Links present).
  2. *Section Completeness* (Experience, Education, Skills, Summary detected).
  3. *Action Verb Strength* (Detects high-impact action verbs like *orchestrated, architected, spearhead, deployed* vs passive/weak phrasing).
  4. *Quantifiable Impact & Metrics* (Detects percentages `25%`, revenue `\$100k`, performance improvements, team sizes).
  5. *Length & Bullet Point Density* (Word count recommendations, bullet length distribution).
  6. *Buzzword / Fluff Detection* (Highlights overused clichés like "hard worker", "go-getter").

### 4. Job Description Matcher & Gap Analysis (`src/matcher.py`)
- **TF-IDF Vectorization** (Term Frequency - Inverse Document Frequency) with n-gram support (1-gram to 3-gram).
- **Cosine Similarity Score** (0 - 100%) between resume and job description.
- **Skill Gap Breakdown**:
  - *Matched Skills*: Skills present in both resume and JD.
  - *Missing Critical Skills*: Skills required by JD but missing in resume.
  - *Bonus / Additional Skills*: Skills on resume not explicitly requested.

### 5. AI / LLM Enhancement Engine (`src/llm_analyzer.py`)
- Integrates Google Gemini (`gemini-3.8-flash` via official `google-genai` SDK `client.interactions.create`).
- Features:
  - *Executive Summary & Fit Verdict*: Honest, professional evaluation of candidate-job alignment.
  - *STAR Bullet Point Rewriter*: Transforms weak bullet points into high-impact `Situation -> Task -> Action -> Result` achievements with quantifiable metrics.
  - *Tailored Interview Prep*: Predicts 5-7 behavioral and technical interview questions based on the candidate's resume gaps.
  - *Graceful Fallback*: If no API key is provided, returns clear algorithmic feedback and informs user how to optionally unlock LLM power.

### 6. Interactive Streamlit Dashboard (`app.py`)
- Clean, responsive UI with:
  - Sidebar: API Key input (optional), sample selector, analysis settings.
  - Main Area:
    - Metric Cards: Overall Match %, ATS Score, Skill Gap Count.
    - Tab 1: **Match & Skill Gap Analysis** (Visual badges, missing keywords).
    - Tab 2: **ATS Formatting & Content Health** (Action verb counter, quantified metrics check, section checklist).
    - Tab 3: **AI Smart Recommendations** (STAR bullet rewriter, tailored critique).
    - Tab 4: **Interview Simulator** (Questions tailored to resume gaps).
    - Tab 5: **Full Exportable Report** (Downloadable Markdown / Summary).

### 7. Educational Guide (`LEARNING_GUIDE.md`)
- Step-by-step breakdown of:
  - How modern Applicant Tracking Systems (ATS) actually parse and filter resumes.
  - Mathematical breakdown of TF-IDF and Cosine Similarity with worked examples.
  - Regular expressions for entity extraction.
  - Prompt engineering for resume critiques and structured JSON outputs.
  - How to showcase this project on GitHub and LinkedIn as a portfolio piece.

---

## Verification Plan

### Automated Tests
- Create unit tests in `tests/` and run with `pytest`:
  1. `test_extractor.py`: Verify text extraction, section segmentation, and contact detail regex.
  2. `test_skills_db.py`: Verify skill extraction, normalization, and alias matching.
  3. `test_matcher.py`: Verify TF-IDF calculation, cosine similarity, and skill gap identification.
  4. `test_ats_analyzer.py`: Verify ATS scoring rules, action verb detection, and metric quantification counter.
- Command:
  ```powershell
  .venv\Scripts\pytest -v
  ```

### Manual Verification
1. Run CLI tool on sample data:
   ```powershell
   .venv\Scripts\python cli.py --resume samples/sample_resume_swe.txt --jd samples/sample_jd_swe.txt
   ```
2. Verify Streamlit application launches without syntax or runtime errors:
   ```powershell
   .venv\Scripts\streamlit run app.py --server.headless true
   ```
