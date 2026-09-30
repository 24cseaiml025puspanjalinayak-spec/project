"""
Unit tests for ATS analyzer, scoring heuristics, and action verbs.
"""

from src.ats_analyzer import (
    analyze_ats,
    detect_action_verbs,
    detect_quantifiable_metrics,
    detect_buzzwords
)


def test_detect_action_verbs():
    text = "Architected and spearheaded microservices. Helped with bug fixes and was responsible for tests."
    res = detect_action_verbs(text)
    assert "architected" in res["strong_verbs"]
    assert "spearheaded" in res["strong_verbs"]
    assert "helped with" in res["weak_phrases"]
    assert "responsible for" in res["weak_phrases"]


def test_detect_quantifiable_metrics():
    text = "Improved performance by 35%, increased revenue by $2.5M, and scaled 4x across 10,000 users."
    metrics = detect_quantifiable_metrics(text)
    assert any("35%" in m for m in metrics["percentages"])
    assert any("$2.5M" in m or "$2" in m for m in metrics["currencies"])
    assert any("4x" in m for m in metrics["multipliers"])
    assert metrics["metrics_count"] >= 3


def test_detect_buzzwords():
    text = "A hardworking team player and go-getter with high synergy."
    buzzwords = detect_buzzwords(text)
    assert "team player" in buzzwords
    assert "go-getter" in buzzwords
    assert "synergy" in buzzwords


def test_analyze_ats_score_bounds():
    sample_text = """
    Alex Chen
    alex@example.com | (555) 123-4567 | https://linkedin.com/in/alexchen

    PROFESSIONAL SUMMARY
    Senior Engineer with 5 years experience.

    TECHNICAL SKILLS
    Python, AWS, Docker, Kubernetes

    WORK EXPERIENCE
    • Architected distributed systems handling 15M requests, reducing latency by 40%.
    • Spearheaded database migration saving $150K annually.
    • Deployed automated CI/CD pipelines across 12 services.

    EDUCATION
    BS in Computer Science
    """
    res = analyze_ats(sample_text)
    assert 0 <= res["total_ats_score"] <= 100
    assert res["total_ats_score"] >= 60  # Should score high with good structure and metrics
    assert res["details"]["contact_info"]["email"] == "alex@example.com"
