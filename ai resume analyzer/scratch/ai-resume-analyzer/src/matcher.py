"""
Job Description Matcher and Skill Gap Analysis Engine
Computes TF-IDF vector representations, Cosine Similarity, and performs
granular skill gap breakdown between resumes and job descriptions.
"""

from __future__ import annotations
from typing import Dict, List, Set, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.skills_db import extract_skills, get_flat_skills


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculates the cosine similarity percentage between a resume and a job description
    using TF-IDF n-gram vectorization (1-gram and 2-gram).
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(sim_score) * 100, 1)


def extract_top_jd_keywords(jd_text: str, top_n: int = 15) -> List[str]:
    """
    Extracts the highest-ranked TF-IDF keywords from the job description.
    """
    if not jd_text.strip():
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=1000
    )
    tfidf_matrix = vectorizer.fit_transform([jd_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # Sort keywords by score
    sorted_indices = np.argsort(scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:top_n] if len(feature_names[i]) > 2]
    return top_keywords


def analyze_skill_gap(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Extracts skills from both resume and job description, categorizing skills into:
    - Matched Skills (in both)
    - Missing Skills (in JD but absent from resume)
    - Additional Skills (in resume but not mentioned in JD)
    """
    resume_skills_categorized = extract_skills(resume_text)
    jd_skills_categorized = extract_skills(jd_text)

    resume_skills_flat: Set[str] = set(get_flat_skills(resume_text))
    jd_skills_flat: Set[str] = set(get_flat_skills(jd_text))

    matched_skills = sorted(list(resume_skills_flat.intersection(jd_skills_flat)))
    missing_skills = sorted(list(jd_skills_flat.difference(resume_skills_flat)))
    additional_skills = sorted(list(resume_skills_flat.difference(jd_skills_flat)))

    if jd_skills_flat:
        skill_match_pct = round((len(matched_skills) / len(jd_skills_flat)) * 100, 1)
    else:
        skill_match_pct = 100.0 if resume_skills_flat else 0.0

    # Categorized breakdown of missing skills
    missing_by_cat: Dict[str, List[str]] = {}
    for cat, skills in jd_skills_categorized.items():
        missing_in_cat = [s for s in skills if s in missing_skills]
        if missing_in_cat:
            missing_by_cat[cat] = missing_in_cat

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
        "matched_count": len(matched_skills),
        "missing_count": len(missing_skills),
        "additional_count": len(additional_skills),
        "total_jd_skills_count": len(jd_skills_flat),
        "skill_match_percentage": skill_match_pct,
        "missing_by_category": missing_by_cat,
        "resume_categorized": resume_skills_categorized,
        "jd_categorized": jd_skills_categorized
    }


def match_resume_to_job(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Performs full matching evaluation:
    - TF-IDF Cosine Similarity
    - Skill Gap Extraction
    - Top JD Keyword Check
    - Weighted Composite Score
    """
    tfidf_sim = compute_tfidf_similarity(resume_text, jd_text)
    skill_gap = analyze_skill_gap(resume_text, jd_text)
    top_keywords = extract_top_jd_keywords(jd_text, top_n=20)

    # Check which top keywords are found in resume
    resume_lower = resume_text.lower()
    keyword_coverage = {
        kw: (kw.lower() in resume_lower)
        for kw in top_keywords
    }
    covered_count = sum(1 for covered in keyword_coverage.values() if covered)
    keyword_match_pct = round((covered_count / len(top_keywords) * 100), 1) if top_keywords else 0.0

    # Composite Score calculation
    # 40% TF-IDF Cosine Similarity, 40% Exact Skill Match %, 20% Keyword Coverage
    composite_score = round(
        (0.40 * tfidf_sim) +
        (0.40 * skill_gap["skill_match_percentage"]) +
        (0.20 * keyword_match_pct),
        1
    )

    return {
        "composite_score": composite_score,
        "tfidf_similarity": tfidf_sim,
        "skill_match_percentage": skill_gap["skill_match_percentage"],
        "keyword_coverage_percentage": keyword_match_pct,
        "skill_gap": skill_gap,
        "top_keywords": top_keywords,
        "keyword_coverage": keyword_coverage
    }
