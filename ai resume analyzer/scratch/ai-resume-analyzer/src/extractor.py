"""
Document Text Extraction and Section Segmentation Engine
Extracts raw text from PDF, DOCX, and TXT files, extracts contact details,
and segments resumes into standard structural sections.
"""

from __future__ import annotations
import io
import re
from typing import Dict, List, Optional, Union
from pypdf import PdfReader
from docx import Document


# Standard Section Headers and their canonical identifiers
SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        r"summary", r"professional summary", r"executive summary",
        r"career summary", r"profile", r"personal statement",
        r"objective", r"career objective", r"about me"
    ],
    "experience": [
        r"experience", r"work experience", r"professional experience",
        r"employment history", r"work history", r"relevant experience",
        r"career history", r"internships"
    ],
    "education": [
        r"education", r"academic background", r"academic qualifications",
        r"academic history", r"degrees", r"educational background"
    ],
    "skills": [
        r"skills", r"technical skills", r"core competencies",
        r"areas of expertise", r"technologies", r"tools & technologies",
        r"skills & tools", r"hard skills", r"key skills"
    ],
    "projects": [
        r"projects", r"personal projects", r"key projects",
        r"academic projects", r"portfolio projects", r"open source"
    ],
    "certifications": [
        r"certifications", r"certificates", r"licenses",
        r"credentials", r"professional certifications", r"courses"
    ],
    "awards": [
        r"awards", r"honors", r"achievements", r"accomplishments",
        r"recognition", r"publications"
    ]
}


def clean_text(text: str) -> str:
    """Normalizes whitespace and standardizes bullet points and dashes."""
    if not text:
        return ""
    # Normalize bullet points
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25CF]\s*", "\n• ", text)
    # Normalize dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """Extracts text content from a PDF file path or byte stream."""
    if isinstance(file_input, (bytes, io.BytesIO)):
        stream = file_input if isinstance(file_input, io.BytesIO) else io.BytesIO(file_input)
        reader = PdfReader(stream)
    else:
        reader = PdfReader(file_input)

    extracted_pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text)

    return clean_text("\n\n".join(extracted_pages))


def extract_text_from_docx(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """Extracts text content from a DOCX file path or byte stream."""
    if isinstance(file_input, (bytes, io.BytesIO)):
        stream = file_input if isinstance(file_input, io.BytesIO) else io.BytesIO(file_input)
        doc = Document(stream)
    else:
        doc = Document(file_input)

    text_parts: List[str] = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text.strip())

    # Also capture table contents if present
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text_parts.append(row_text)

    return clean_text("\n\n".join(text_parts))


def extract_text(file_input: Union[str, bytes, io.BytesIO], filename: str = "") -> str:
    """
    Unified entry point to extract text from PDF, DOCX, or TXT file or buffer.
    """
    fname_lower = filename.lower()
    if fname_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_input)
    elif fname_lower.endswith(".docx"):
        return extract_text_from_docx(file_input)
    elif fname_lower.endswith(".txt") or not filename:
        if isinstance(file_input, bytes):
            return clean_text(file_input.decode("utf-8", errors="replace"))
        elif isinstance(file_input, io.BytesIO):
            return clean_text(file_input.getvalue().decode("utf-8", errors="replace"))
        elif isinstance(file_input, str):
            # Check if file_input is a path or raw text
            if "\n" not in file_input and (file_input.endswith(".txt") or len(file_input) < 300):
                try:
                    with open(file_input, "r", encoding="utf-8", errors="replace") as f:
                        return clean_text(f.read())
                except Exception:
                    return clean_text(file_input)
            return clean_text(file_input)

    raise ValueError(f"Unsupported file format for filename: '{filename}'. Expected .pdf, .docx, or .txt")


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extracts email, phone, LinkedIn, GitHub, and portfolio URLs from text.
    """
    # Email regex
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    email = email_match.group(0) if email_match else None

    # Phone regex (international & US formats)
    phone_pattern = r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
    phone_match = re.search(phone_pattern, text)
    phone = phone_match.group(0).strip() if phone_match else None

    # LinkedIn regex
    linkedin_match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_-]+)/?",
        text,
        re.IGNORECASE
    )
    linkedin = f"https://linkedin.com/in/{linkedin_match.group(1)}" if linkedin_match else None

    # GitHub regex
    github_match = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)/?",
        text,
        re.IGNORECASE
    )
    github = f"https://github.com/{github_match.group(1)}" if github_match else None

    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
    }


def segment_sections(text: str) -> Dict[str, str]:
    """
    Segments resume text into canonical sections:
    summary, experience, education, skills, projects, certifications, awards, other.
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {key: [] for key in SECTION_PATTERNS}
    sections["other"] = []

    current_section = "other"

    # Precompile section header checkers
    header_regexes: Dict[str, List[re.Pattern]] = {
        sec: [re.compile(rf"^[\s#*•\-–]*{pat}[\s:•\-–]*$", re.IGNORECASE) for pat in pats]
        for sec, pats in SECTION_PATTERNS.items()
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_section:
                sections[current_section].append(line)
            continue

        # Check if line looks like a header (short line, <= 40 chars)
        matched_section = None
        if len(stripped) <= 40:
            for sec, regexes in header_regexes.items():
                if any(rx.match(stripped) for rx in regexes):
                    matched_section = sec
                    break

        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(line)

    # Convert line lists to clean joined strings
    return {sec: clean_text("\n".join(lines)) for sec, lines in sections.items()}


def estimate_years_of_experience(text: str) -> Optional[float]:
    """
    Estimates years of experience using date ranges and explicit statements.
    Examples matched: '2018 - 2023', '2019 to Present', '5+ years experience'.
    """
    # 1. Look for explicit statements like '5+ years of experience'
    explicit_match = re.search(
        r"\b(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience\b",
        text,
        re.IGNORECASE
    )
    if explicit_match:
        try:
            return float(explicit_match.group(1))
        except ValueError:
            pass

    # 2. Look for year ranges like 2018 - 2023, 2020 - Present
    year_ranges = re.findall(
        r"\b(20\d{2}|19\d{2})\s*(?:-|–|—|to)\s*(20\d{2}|present|current)\b",
        text,
        re.IGNORECASE
    )
    if year_ranges:
        current_year = 2026
        total_span = 0
        spans: List[tuple[int, int]] = []
        for start_str, end_str in year_ranges:
            start_yr = int(start_str)
            end_yr = current_year if end_str.lower() in ("present", "current") else int(end_str)
            if 1980 <= start_yr <= end_yr <= current_year + 1:
                spans.append((start_yr, end_yr))

        if spans:
            # Merge overlapping intervals
            spans.sort(key=lambda x: x[0])
            merged = [spans[0]]
            for current in spans[1:]:
                prev_start, prev_end = merged[-1]
                if current[0] <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, current[1]))
                else:
                    merged.append(current)

            total_years = sum(end - start for start, end in merged)
            return float(total_years) if total_years > 0 else None

    return None
