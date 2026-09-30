"""
Unit tests for TF-IDF similarity, skill gap calculation, and resume-job matching.
"""

from src.matcher import (
    compute_tfidf_similarity,
    analyze_skill_gap,
    match_resume_to_job,
    extract_top_jd_keywords
)


def test_tfidf_similarity_identical_and_different():
    text1 = "Python FastAPI Docker Kubernetes AWS PostgreSQL engineer"
    text2 = "Looking for Python FastAPI Docker Kubernetes AWS PostgreSQL engineer"
    sim = compute_tfidf_similarity(text1, text2)
    assert sim > 60.0

    unrelated = "French cuisine pastry chef baking bread and croissants"
    sim_unrelated = compute_tfidf_similarity(text1, unrelated)
    assert sim_unrelated < 20.0


def test_analyze_skill_gap():
    resume = "Skills: Python, FastAPI, Docker, PostgreSQL."
    jd = "Requirements: Python, FastAPI, Docker, Kubernetes, AWS."

    gap = analyze_skill_gap(resume, jd)
    assert "Python" in gap["matched_skills"]
    assert "FastAPI" in gap["matched_skills"]
    assert "Docker" in gap["matched_skills"]
    assert "Kubernetes" in gap["missing_skills"]
    assert "AWS" in gap["missing_skills"]
    assert gap["skill_match_percentage"] == 60.0  # 3 out of 5


def test_match_resume_to_job():
    resume = """
    Jane Doe | jane@example.com
    Software Engineer skilled in Python, Django, AWS, Docker, and PostgreSQL.
    Architected cloud systems and built REST APIs.
    """
    jd = """
    We need a Python Engineer with Django, AWS, Docker, and Kubernetes experience.
    Must have experience designing REST APIs and cloud infrastructure.
    """
    result = match_resume_to_job(resume, jd)
    assert 0 <= result["composite_score"] <= 100
    assert result["composite_score"] >= 40
    assert "Python" in result["skill_gap"]["matched_skills"]
    assert "Kubernetes" in result["skill_gap"]["missing_skills"]


def test_extract_top_jd_keywords():
    jd = "Kubernetes Docker Python Microservices AWS Cloud Architecture FastAPI"
    keywords = extract_top_jd_keywords(jd, top_n=5)
    assert len(keywords) > 0
