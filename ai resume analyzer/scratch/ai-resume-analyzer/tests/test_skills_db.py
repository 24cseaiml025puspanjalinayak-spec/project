"""
Unit tests for skills database and taxonomy extraction.
"""

from src.skills_db import extract_skills, get_flat_skills, SKILLS_TAXONOMY


def test_extract_canonical_skills():
    text = "Experienced in Python, Docker, Kubernetes, and PostgreSQL with AWS."
    skills = get_flat_skills(text)
    assert "Python" in skills
    assert "Docker" in skills
    assert "Kubernetes" in skills
    assert "PostgreSQL" in skills
    assert "AWS" in skills


def test_extract_skill_aliases():
    text = "Built web apps with reactjs, nodejs, and deployed on k8s with gcp."
    skills = get_flat_skills(text)
    assert "React" in skills
    assert "Node.js" in skills
    assert "Kubernetes" in skills
    assert "Google Cloud" in skills


def test_categorized_skills():
    text = "Strong Python and TypeScript developer using FastAPI and React."
    categorized = extract_skills(text)
    assert "Python" in categorized["Programming Languages"]
    assert "TypeScript" in categorized["Programming Languages"]
    assert "FastAPI" in categorized["Frameworks & Web Tech"]
    assert "React" in categorized["Frameworks & Web Tech"]


def test_word_boundaries_no_false_positives():
    # 'go' shouldn't match 'good' or 'category', 'r' shouldn't match 'car'
    text = "A good manager in our category drove a car."
    skills = get_flat_skills(text)
    assert "Go" not in skills
    assert "R" not in skills
