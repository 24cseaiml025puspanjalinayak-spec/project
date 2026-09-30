"""
Unit tests for text extraction, contact information parser, and section segmenter.
"""

from src.extractor import clean_text, extract_contact_info, segment_sections, estimate_years_of_experience


def test_clean_text():
    raw = "Hello   world \u2022 bullet item \u2013 dash \r\n\r\n\r\n\r\nNew line"
    cleaned = clean_text(raw)
    assert "• bullet item" in cleaned
    assert "-" in cleaned
    assert "\r" not in cleaned


def test_extract_contact_info():
    text = """
    John Doe
    Email: john.doe@example.com
    Phone: (555) 123-4567
    LinkedIn: https://linkedin.com/in/johndoe-dev
    GitHub: https://github.com/johndoe-code
    """
    contact = extract_contact_info(text)
    assert contact["email"] == "john.doe@example.com"
    assert contact["phone"] == "(555) 123-4567"
    assert contact["linkedin"] == "https://linkedin.com/in/johndoe-dev"
    assert contact["github"] == "https://github.com/johndoe-code"


def test_segment_sections():
    resume_text = """
    Jane Smith
    jane@example.com

    PROFESSIONAL SUMMARY
    Passionate software engineer with 5 years experience.

    WORK EXPERIENCE
    Senior Engineer at TechCorp
    - Built cloud systems

    EDUCATION
    BS in Computer Science, Stanford University

    TECHNICAL SKILLS
    Python, AWS, Docker
    """
    sections = segment_sections(resume_text)
    assert "software engineer" in sections["summary"].lower()
    assert "techecorp" in sections["experience"].lower() or "senior engineer" in sections["experience"].lower()
    assert "stanford" in sections["education"].lower()
    assert "python" in sections["skills"].lower()


def test_estimate_years_of_experience():
    text = "Senior Developer with 7+ years of experience in distributed systems."
    years = estimate_years_of_experience(text)
    assert years == 7.0

    text2 = "Worked at Corp A from 2018 - 2023 and Corp B from 2023 - Present"
    years2 = estimate_years_of_experience(text2)
    assert years2 is not None
    assert years2 >= 5.0
