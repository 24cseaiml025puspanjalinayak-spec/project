"""
Report Generator Module
Compiles complete analysis results into structured Markdown and JSON reports for export and CLI display.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Dict, Any


def generate_markdown_report(
    resume_name: str,
    jd_title: str,
    match_data: Dict[str, Any],
    ats_data: Dict[str, Any],
    llm_data: Dict[str, Any]
) -> str:
    """Generates an executive-ready Markdown report."""
    comp_score = match_data.get("composite_score", 0)
    ats_score = ats_data.get("total_ats_score", 0)
    skill_pct = match_data.get("skill_match_percentage", 0)
    tfidf_sim = match_data.get("tfidf_similarity", 0)

    contact = ats_data.get("details", {}).get("contact_info", {})
    matched_skills = match_data.get("skill_gap", {}).get("matched_skills", [])
    missing_skills = match_data.get("skill_gap", {}).get("missing_skills", [])
    additional_skills = match_data.get("skill_gap", {}).get("additional_skills", [])
    suggestions = ats_data.get("suggestions", [])
    star_rewrites = llm_data.get("star_bullet_rewrites", [])
    questions = llm_data.get("interview_questions", [])

    lines = [
        f"# AI Resume Analysis & Job Fit Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Model:** {llm_data.get('model_used', 'Classical NLP')}",
        f"**Target Role:** {jd_title or 'Target Job Description'} | **Resume Document:** {resume_name or 'Uploaded Resume'}",
        "",
        "---",
        "",
        "## Executive Scorecard",
        f"| Metric | Score | Benchmark Target |",
        f"| :--- | :--- | :--- |",
        f"| **Overall Match Score** | **{comp_score}%** | 75%+ Recommended |",
        f"| **ATS Health & Formatting** | **{ats_score}/100** | 80+/100 Recommended |",
        f"| **Exact Skill Overlap** | **{skill_pct}%** | 70%+ Recommended |",
        f"| **TF-IDF Vocabulary Similarity** | **{tfidf_sim}%** | 40%+ Semantic Alignment |",
        "",
        f"> **Executive Summary**: {llm_data.get('executive_summary', 'Analysis completed.')}",
        "",
        "---",
        "",
        "## Candidate Profile & Contact Information",
        f"- **Email**: {contact.get('email') or '⚠️ Not detected'}",
        f"- **Phone**: {contact.get('phone') or '⚠️ Not detected'}",
        f"- **LinkedIn**: {contact.get('linkedin') or '⚠️ Not detected'}",
        f"- **GitHub**: {contact.get('github') or '⚠️ Not detected'}",
        f"- **Word Count**: {ats_data.get('stats', {}).get('word_count', 0)} words",
        f"- **Bullet Points**: {ats_data.get('stats', {}).get('bullet_count', 0)} bullets",
        "",
        "---",
        "",
        "## Skill Gap Breakdown",
        f"### ✅ Matched Competencies ({len(matched_skills)})",
        ", ".join([f"`{s}`" for s in matched_skills]) if matched_skills else "_None identified._",
        "",
        f"### ❌ Missing Critical Competencies ({len(missing_skills)})",
        ", ".join([f"`{s}`" for s in missing_skills]) if missing_skills else "_No major skill gaps identified._",
        "",
        f"### 💡 Bonus / Additional Skills ({len(additional_skills)})",
        ", ".join([f"`{s}`" for s in additional_skills[:15]]) if additional_skills else "_None._",
        "",
        "---",
        "",
        "## ATS Formatting & Diagnostic Health",
        f"- **Strong Action Verbs Count**: {ats_data.get('stats', {}).get('strong_verbs_count', 0)}",
        f"- **Quantified Metrics Found**: {ats_data.get('stats', {}).get('metrics_count', 0)}",
        f"- **Buzzword Clichés Detected**: {ats_data.get('stats', {}).get('buzzwords_count', 0)}",
        "",
        "### Actionable Improvements:",
    ]

    for sug in suggestions:
        lines.append(f"- 🔧 {sug}")

    if not suggestions:
        lines.append("- ✨ No critical ATS formatting issues detected.")

    lines.extend([
        "",
        "---",
        "",
        "## STAR Method Bullet Point Rewrites",
        "Transforming weak descriptions into impactful, quantifiable achievements:"
    ])

    for i, rewrite in enumerate(star_rewrites, 1):
        lines.extend([
            f"### Example {i}",
            f"- **Original / Weak Bullet**: *\"{rewrite.get('original', '')}\"*",
            f"- **Issue**: {rewrite.get('critique', '')}",
            f"- **Recommended STAR Rewrite**: **\"{rewrite.get('star_rewrite', '')}\"**",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## Recommended Interview Preparation Questions",
        "Targeting identified resume gaps and technical depths:"
    ])

    for i, q in enumerate(questions, 1):
        lines.extend([
            f"{i}. **[{q.get('category', 'Technical')}]**: {q.get('question', '')}",
            f"   - *Recruiter Intent*: _{q.get('rationale', '')}_",
            ""
        ])

    return "\n".join(lines)


def generate_json_report(
    resume_name: str,
    jd_title: str,
    match_data: Dict[str, Any],
    ats_data: Dict[str, Any],
    llm_data: Dict[str, Any]
) -> str:
    """Returns a full JSON serialization of the analysis results."""
    combined = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "resume_name": resume_name,
            "target_role": jd_title,
            "model": llm_data.get("model_used")
        },
        "scores": {
            "composite_match": match_data.get("composite_score"),
            "ats_health_score": ats_data.get("total_ats_score"),
            "skill_match_percentage": match_data.get("skill_match_percentage"),
            "tfidf_similarity": match_data.get("tfidf_similarity")
        },
        "match_details": match_data,
        "ats_diagnostics": ats_data,
        "ai_recommendations": llm_data
    }
    return json.dumps(combined, indent=2)
