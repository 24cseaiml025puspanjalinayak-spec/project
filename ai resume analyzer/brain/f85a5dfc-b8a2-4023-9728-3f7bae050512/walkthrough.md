# Project Walkthrough: AI-Based Resume Analyzer

A complete, production-grade learning project built at:
`C:\Users\Dell\.gemini\antigravity\scratch\ai-resume-analyzer`

---

## 🎯 What Was Built

We engineered a complete **Hybrid AI-Based Resume Analyzer & Job Fit Engine** in Python, fusing classical Natural Language Processing (NLP), real-world Applicant Tracking System (ATS) heuristic scoring, and modern Generative AI (Google Gemini 3.8 Flash):

```
ai-resume-analyzer/
├── src/
│   ├── extractor.py           # Multi-format document parser (PDF, DOCX, TXT) & section segmenter
│   ├── skills_db.py           # 1,000+ skills taxonomy with alias mapping & boundary regex
│   ├── ats_analyzer.py        # ATS scoring engine (action verbs, metrics, structure, buzzwords)
│   ├── matcher.py             # TF-IDF Cosine Similarity & skill gap analyzer
│   ├── llm_analyzer.py        # Gemini 3.8 Flash integration & offline heuristics fallback
│   └── report_generator.py    # Markdown & JSON report generator
├── samples/
│   ├── sample_resume_swe.txt            # Software Engineer sample resume
│   ├── sample_jd_swe.txt                # Software Engineer sample job description
│   ├── sample_resume_data_scientist.txt # Data Scientist sample resume
│   └── sample_jd_data_scientist.txt     # Data Scientist sample job description
├── tests/
│   ├── test_extractor.py      # Unit tests for text cleaning & contact info parsing
│   ├── test_skills_db.py      # Unit tests for skills extraction & word boundaries
│   ├── test_ats_analyzer.py   # Unit tests for ATS scoring & action verb checks
│   └── test_matcher.py        # Unit tests for TF-IDF similarity & skill gaps
├── app.py                     # Streamlit interactive web dashboard
├── cli.py                     # Command-line interface with export capabilities
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Dependency lockfile
├── .env.example               # Template for optional GEMINI_API_KEY
├── README.md                  # Comprehensive project documentation & quickstart
└── LEARNING_GUIDE.md          # In-depth educational guide on NLP, ATS, and LLMs
```

---

## 🧪 Verification & Test Results

### Automated Unit Tests (`pytest`)
All **16 unit tests** passed with 100% success:

```powershell
.\.venv\Scripts\pytest -v
```

```text
tests/test_ats_analyzer.py::test_detect_action_verbs PASSED              [  6%]
tests/test_ats_analyzer.py::test_detect_quantifiable_metrics PASSED      [ 12%]
tests/test_ats_analyzer.py::test_detect_buzzwords PASSED                 [ 18%]
tests/test_ats_analyzer.py::test_analyze_ats_score_bounds PASSED         [ 25%]
tests/test_extractor.py::test_clean_text PASSED                          [ 31%]
tests/test_extractor.py::test_extract_contact_info PASSED                [ 37%]
tests/test_extractor.py::test_segment_sections PASSED                    [ 43%]
tests/test_extractor.py::test_estimate_years_of_experience PASSED        [ 50%]
tests/test_matcher.py::test_tfidf_similarity_identical_and_different PASSED [ 56%]
tests/test_matcher.py::test_analyze_skill_gap PASSED                     [ 62%]
tests/test_matcher.py::test_match_resume_to_job PASSED                   [ 68%]
tests/test_matcher.py::test_extract_top_jd_keywords PASSED               [ 75%]
tests/test_skills_db.py::test_extract_canonical_skills PASSED            [ 81%]
tests/test_skills_db.py::test_extract_skill_aliases PASSED               [ 87%]
tests/test_skills_db.py::test_categorized_skills PASSED                  [ 93%]
tests/test_skills_db.py::test_word_boundaries_no_false_positives PASSED  [100%]

============================= 16 passed in 5.54s ==============================
```

### CLI Verification & Sample Run
Tested with built-in Software Engineer and Data Scientist samples:

```powershell
.\.venv\Scripts\python cli.py --sample
```

**Output snippet**:
```text
==================== SCORECARD ====================
 Overall Match Score:         61.9%
 ATS Health & Formatting:     97/100
 Exact Skill Overlap:         95.2%
 TF-IDF Vocabulary Sim:       17.0%
 Engine / Model:              Offline Heuristics Engine
===================================================

[+] Matched Skills:
    AWS, Agile, CI/CD, Django, Docker, FastAPI, GitHub, GitHub Actions, Go, Kubernetes, Microservices, PostgreSQL, Python, REST API, Redis, Scrum, System Design, TDD, Terraform, Test-Driven Development

[-] Missing Skills (JD Requirements Missing in Resume):
    Mentoring
```

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web Dashboard
Run the Streamlit app from within the project directory:

```powershell
cd C:\Users\Dell\.gemini\antigravity\scratch\ai-resume-analyzer
.\.venv\Scripts\streamlit run app.py
```

Then visit **`http://localhost:8501`** in your browser.
- **Select Sample Data**: One-click test with preloaded SWE or Data Scientist resumes.
- **Upload Your Own**: Drag and drop `.pdf`, `.docx`, or `.txt` resumes.
- **Explore the Tabs**:
  1. *Skill Gap Breakdown* (color-coded badges for matched, missing, and bonus skills).
  2. *ATS Diagnostics* (meters for action verbs, quantified metrics, section checklist).
  3. *AI Critique & STAR Rewriter* (situation, task, action, result bullet transformations).
  4. *Interview Simulator* (dynamic questions targeting candidate gaps).
  5. *Full Exportable Report* (download Markdown or JSON).

### 2. Optional: Enable Google Gemini 3.8 Flash
The project works 100% offline out-of-the-box. To activate live Gemini 3.8 Flash intelligence:
- Enter your key in the web app's sidebar, or
- Add `GEMINI_API_KEY=your_key_here` to `.env`.

---

## 📚 Educational Value

Read [LEARNING_GUIDE.md](file:///C:/Users/Dell/.gemini/antigravity/scratch/ai-resume-analyzer/LEARNING_GUIDE.md) for masterclass-level explanations of:
1. **ATS Engineering**: Common parsing traps, multi-column tables, text layer extraction.
2. **Vector Space Models**: Mathematical derivation of Sublinear TF-IDF and Cosine Similarity.
3. **Entity Extraction**: Word boundary regexes and escaping punctuation in tech acronyms.
4. **Hybrid AI & Prompting**: Schema-constrained JSON prompting with graceful offline degradation.
5. **The STAR Framework**: Transforming passive responsibilities into metric-backed achievements.
