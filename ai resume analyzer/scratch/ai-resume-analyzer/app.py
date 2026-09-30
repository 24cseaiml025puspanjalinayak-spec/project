"""
AI Resume Analyzer & Job Fit Engine - Streamlit Web Application
Interactive dashboard combining Classical NLP, ATS Scoring, and Google Gemini AI.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.extractor import extract_text
from src.matcher import match_resume_to_job
from src.ats_analyzer import analyze_ats
from src.llm_analyzer import analyze_with_gemini
from src.report_generator import generate_markdown_report, generate_json_report


# Page Configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Job Fit Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
    }
    .badge-matched {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    .badge-missing {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ffcdd2;
    }
    .badge-bonus {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #bbdefb;
    }
    .star-card {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_file(rel_path: str) -> str:
    """Helper to read sample files."""
    sample_path = Path(__file__).parent / rel_path
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


# Sidebar Configuration
st.sidebar.title("⚙️ Configuration & Inputs")

# Optional Gemini API Key
api_key_input = st.sidebar.text_input(
    "🔑 Google Gemini API Key (Optional)",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Enter your Gemini API key to activate live Gemini 3.8 Flash reasoning. If omitted, full offline NLP heuristics are used."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 1. Select or Upload Resume")

# Sample Pre-loader buttons
sample_choice = st.sidebar.radio(
    "Quick Test with Sample Data:",
    ("None (Custom Upload)", "Software Engineer (SWE)", "Data Scientist (DS)"),
    index=1
)

preloaded_resume = ""
preloaded_jd = ""
if sample_choice == "Software Engineer (SWE)":
    preloaded_resume = load_sample_file("samples/sample_resume_swe.txt")
    preloaded_jd = load_sample_file("samples/sample_jd_swe.txt")
elif sample_choice == "Data Scientist (DS)":
    preloaded_resume = load_sample_file("samples/sample_resume_data_scientist.txt")
    preloaded_jd = load_sample_file("samples/sample_jd_data_scientist.txt")

resume_file = st.sidebar.file_uploader(
    "Upload Resume (PDF, DOCX, TXT):",
    type=["pdf", "docx", "txt"],
    help="Upload your actual resume file in PDF, Word, or plain text format."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 2. Target Job Description")

jd_file = st.sidebar.file_uploader(
    "Upload Job Description (Optional):",
    type=["txt"],
    help="Upload a job description text file or paste below."
)

# Text area for Job Description
if jd_file is not None:
    jd_text_value = jd_file.read().decode("utf-8", errors="replace")
elif preloaded_jd:
    jd_text_value = preloaded_jd
else:
    jd_text_value = ""

jd_input = st.sidebar.text_area(
    "Or Paste Job Description:",
    value=jd_text_value,
    height=200,
    placeholder="Paste the target job description requirements, responsibilities, and qualifications..."
)

# Main Application Logic
st.markdown('<div class="main-header">🎯 AI Resume Analyzer & Job Fit Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classical NLP + ATS Heuristic Scoring + Google Gemini 3.8 Flash Hybrid Architecture</div>', unsafe_allow_html=True)

# Determine Resume Text Source
resume_text = ""
resume_filename = "Uploaded Resume"

if resume_file is not None:
    resume_filename = resume_file.name
    try:
        resume_text = extract_text(resume_file.getvalue(), filename=resume_file.name)
    except Exception as e:
        st.error(f"Error parsing resume file: {e}")
elif preloaded_resume:
    resume_filename = f"Sample: {sample_choice}"
    resume_text = preloaded_resume

if not resume_text or not jd_input.strip():
    st.info("👈 Please upload a resume and target job description in the sidebar (or use the built-in sample) to begin the analysis.")
    st.stop()

# Execution of Analytics Pipeline with Spinner
with st.spinner("Analyzing resume against job description and evaluating ATS health..."):
    match_data = match_resume_to_job(resume_text, jd_input)
    ats_data = analyze_ats(resume_text)
    effective_key = api_key_input.strip() or os.getenv("GEMINI_API_KEY")
    llm_data = analyze_with_gemini(resume_text, jd_input, match_data, ats_data, api_key=effective_key)

# Top Status Notification
if llm_data.get("is_llm_powered"):
    st.success(f"✨ **Live AI Active**: {llm_data.get('notice')}")
else:
    st.warning(f"⚡ **Offline Mode**: {llm_data.get('notice')}")

# Top Metric Cards
comp_score = match_data.get("composite_score", 0)
ats_score = ats_data.get("total_ats_score", 0)
skill_pct = match_data.get("skill_match_percentage", 0)
tfidf_sim = match_data.get("tfidf_similarity", 0)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="🎯 Overall Job Fit Match",
        value=f"{comp_score}%",
        delta=f"{comp_score - 70:.1f}% vs benchmark" if comp_score >= 70 else f"-{70 - comp_score:.1f}% gap"
    )
with col2:
    st.metric(
        label="🛡️ ATS Health Score",
        value=f"{ats_score}/100",
        delta=f"{ats_score - 75:.1f} pts" if ats_score >= 75 else f"{ats_score - 75:.1f} pts"
    )
with col3:
    st.metric(
        label="🧩 Exact Skill Match",
        value=f"{skill_pct}%",
        delta=f"{match_data['skill_gap']['matched_count']} matched skills"
    )
with col4:
    st.metric(
        label="📝 TF-IDF Similarity",
        value=f"{tfidf_sim}%",
        delta="Vocabulary alignment"
    )

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 Skill Gap Breakdown",
    "🛡️ ATS Health Diagnostics",
    "💡 AI Critique & STAR Rewriter",
    "🎯 Interview Simulator",
    "📄 Full Exportable Report"
])

# ================= TAB 1: Skill Gap Breakdown =================
with tab1:
    st.subheader("Skill Gap & Requirements Analysis")
    skill_gap = match_data["skill_gap"]

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(f"#### ✅ Matched Skills ({skill_gap['matched_count']})")
        if skill_gap["matched_skills"]:
            badges_html = " ".join([f'<span class="badge-pill badge-matched">✓ {s}</span>' for s in skill_gap["matched_skills"]])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.write("No direct skills matched.")

        st.markdown(f"#### 💡 Additional / Bonus Skills ({skill_gap['additional_count']})")
        if skill_gap["additional_skills"]:
            badges_html = " ".join([f'<span class="badge-pill badge-bonus">+ {s}</span>' for s in skill_gap["additional_skills"][:15]])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.write("No bonus skills detected.")

    with c2:
        st.markdown(f"#### ❌ Missing Critical Skills ({skill_gap['missing_count']})")
        if skill_gap["missing_skills"]:
            badges_html = " ".join([f'<span class="badge-pill badge-missing">✗ {s}</span>' for s in skill_gap["missing_skills"]])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.success("🎉 Outstanding! Your resume covers all explicit skills required by this job description!")

        if skill_gap["missing_by_category"]:
            st.markdown("##### Missing Skills by Category:")
            for cat, skills in skill_gap["missing_by_category"].items():
                st.write(f"- **{cat}**: {', '.join(skills)}")

    st.markdown("---")
    st.markdown("#### 🔍 Top Job Description Keywords Alignment")
    top_kw = match_data.get("top_keywords", [])
    coverage = match_data.get("keyword_coverage", {})
    if top_kw:
        cols = st.columns(3)
        for i, kw in enumerate(top_kw[:18]):
            is_present = coverage.get(kw, False)
            icon = "✅" if is_present else "❌"
            cols[i % 3].write(f"{icon} **{kw}**")


# ================= TAB 2: ATS Diagnostics =================
with tab2:
    st.subheader("ATS Formatting, Structure & Content Diagnostics")

    sub = ats_data["sub_scores"]
    stats = ats_data["stats"]

    # Score breakdown meters
    st.markdown("#### ATS Scoring Components")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write(f"**Section Completeness**: {sub['section_completeness']['score']} / {sub['section_completeness']['max']} pts")
        st.progress(sub['section_completeness']['score'] / sub['section_completeness']['max'])

        st.write(f"**Contact Information**: {sub['contact_completeness']['score']} / {sub['contact_completeness']['max']} pts")
        st.progress(sub['contact_completeness']['score'] / sub['contact_completeness']['max'])

        st.write(f"**Action Verb Strength**: {sub['action_verbs']['score']} / {sub['action_verbs']['max']} pts")
        st.progress(sub['action_verbs']['score'] / sub['action_verbs']['max'])

    with col_b:
        st.write(f"**Quantified Metrics & Impact**: {sub['quantifiable_metrics']['score']} / {sub['quantifiable_metrics']['max']} pts")
        st.progress(sub['quantifiable_metrics']['score'] / sub['quantifiable_metrics']['max'])

        st.write(f"**Formatting & Length**: {sub['formatting_and_length']['score']} / {sub['formatting_and_length']['max']} pts")
        st.progress(sub['formatting_and_length']['score'] / sub['formatting_and_length']['max'])

        if sub['buzzword_penalty'] > 0:
            st.warning(f"⚠️ Buzzword / Cliché penalty applied: -{sub['buzzword_penalty']} pts")

    st.markdown("---")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 📬 Contact & Structure Detection")
        contact = ats_data["details"]["contact_info"]
        st.write(f"- **Email**: {'✅ ' + contact['email'] if contact.get('email') else '❌ Missing'}")
        st.write(f"- **Phone**: {'✅ ' + contact['phone'] if contact.get('phone') else '❌ Missing'}")
        st.write(f"- **LinkedIn**: {'✅ ' + contact['linkedin'] if contact.get('linkedin') else '❌ Missing'}")
        st.write(f"- **GitHub**: {'✅ ' + contact['github'] if contact.get('github') else '❌ Missing'}")

        st.markdown("#### 📑 Detected Resume Sections")
        sections_found = ats_data["details"]["detected_sections"]
        st.write(", ".join([f"`{sec.title()}`" for sec in sections_found]) if sections_found else "None detected")

    with col_d:
        st.markdown("#### ⚡ Impact & Action Verbs")
        st.write(f"- **Word Count**: {stats['word_count']} words (Recommended: 400 - 900)")
        st.write(f"- **Bullet Points Detected**: {stats['bullet_count']}")
        st.write(f"- **Strong Action Verbs Count**: {stats['strong_verbs_count']}")
        if ats_data["details"]["action_verbs"]["strong_verbs"]:
            st.caption("Verbs found: " + ", ".join(ats_data["details"]["action_verbs"]["strong_verbs"][:10]))

        st.write(f"- **Quantified Metrics Found**: {stats['metrics_count']}")
        if ats_data["details"]["metrics"]["all_metrics"]:
            st.caption("Metrics: " + ", ".join(ats_data["details"]["metrics"]["all_metrics"][:6]))

    st.markdown("---")
    st.markdown("#### 🔧 Actionable ATS Recommendations")
    if ats_data["suggestions"]:
        for sug in ats_data["suggestions"]:
            st.info(f"💡 {sug}")
    else:
        st.success("✨ Your resume conforms cleanly to ATS formatting guidelines!")


# ================= TAB 3: AI Critique & STAR Rewriter =================
with tab3:
    st.subheader("AI Executive Assessment & STAR Method Rewriter")
    st.info(f"**Assessment Engine**: {llm_data.get('model_used')}")

    st.markdown("### 📋 Executive Summary")
    st.markdown(f"> {llm_data.get('executive_summary')}")

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### 🌟 Top Identified Strengths")
        for strength in llm_data.get("key_strengths", []):
            st.write(f"- ✅ {strength}")

    with col_f:
        st.markdown("#### ⚠️ Critical Candidate Gaps")
        for gap in llm_data.get("critical_gaps", []):
            st.write(f"- 🚩 {gap}")

    st.markdown("---")
    st.markdown("### ⭐ STAR Method Bullet Point Transformations")
    st.caption("The STAR format (Situation, Task, Action, Result) ensures accomplishments are active, clear, and quantified.")

    for i, rewrite in enumerate(llm_data.get("star_bullet_rewrites", []), 1):
        with st.container():
            st.markdown(f"""
            <div class="star-card">
                <span style="font-weight: 700; color: #1E88E5;">Example {i}</span>
                <p><strong>Original / Weak Bullet:</strong> <em>"{rewrite.get('original')}"</em></p>
                <p style="color: #c62828;"><strong>Critique:</strong> {rewrite.get('critique')}</p>
                <p style="color: #2e7d32;"><strong>High-Impact STAR Rewrite:</strong><br>
                <strong>"{rewrite.get('star_rewrite')}"</strong></p>
            </div>
            """, unsafe_allow_html=True)


# ================= TAB 4: Interview Simulator =================
with tab4:
    st.subheader("🎯 Targeted Interview Simulator")
    st.write("These questions are dynamically generated to target your specific resume qualifications and identified skill gaps:")

    for i, q in enumerate(llm_data.get("interview_questions", []), 1):
        with st.expander(f"Q{i}: [{q.get('category', 'Technical')}] {q.get('question')}", expanded=(i == 1)):
            st.markdown(f"**Target Competency**: `{q.get('category')}`")
            st.markdown(f"**Recruiter / Hiring Manager Rationale**:")
            st.write(f"_{q.get('rationale')}_")
            st.markdown("**Preparation Strategy**:")
            st.write(
                "Structure your answer using the STAR method: State the specific context, the technical challenge, "
                "the exact tools and decisions you made, and back up the final outcome with measurable metrics."
            )


# ================= TAB 5: Full Exportable Report =================
with tab5:
    st.subheader("📄 Export Executive Resume Evaluation Report")

    markdown_report = generate_markdown_report(resume_filename, "Target Role", match_data, ats_data, llm_data)
    json_report = generate_json_report(resume_filename, "Target Role", match_data, ats_data, llm_data)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label="💾 Download Markdown Report (.md)",
            data=markdown_report,
            file_name="resume_analysis_report.md",
            mime="text/markdown"
        )
    with col_btn2:
        st.download_button(
            label="📦 Download JSON Data (.json)",
            data=json_report,
            file_name="resume_analysis_data.json",
            mime="application/json"
        )

    st.markdown("---")
    st.markdown("#### Report Preview")
    st.markdown(markdown_report)
