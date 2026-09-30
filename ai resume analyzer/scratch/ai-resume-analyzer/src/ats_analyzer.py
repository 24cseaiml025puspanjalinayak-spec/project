"""
ATS (Applicant Tracking System) Health & Formatting Analyzer
Evaluates resumes on action verb strength, quantified metrics, structural completeness,
length heuristics, and buzzword density to calculate an actionable ATS Score (0-100).
"""

from __future__ import annotations
import re
from typing import Dict, List, Any
from src.extractor import extract_contact_info, segment_sections

# High-impact strong action verbs categorized by function
STRONG_ACTION_VERBS = [
    # Leadership & Orchestration
    "spearheaded", "orchestrated", "architected", "championed", "directed",
    "established", "founded", "led", "mentored", "supervised", "pioneered",
    # Engineering & Building
    "engineered", "developed", "designed", "implemented", "constructed",
    "programmed", "coded", "configured", "built", "authored", "deployed",
    # Optimization & Performance
    "optimized", "accelerated", "streamlined", "automated", "maximized",
    "enhanced", "transformed", "revamped", "restructured", "scaled",
    # Analysis & Research
    "analyzed", "evaluated", "diagnosed", "audited", "formulated",
    "researched", "investigated", "benchmarked", "synthesized", "modeled",
    # Impact & Results
    "boosted", "generated", "reduced", "delivered", "negotiated",
    "surpassed", "expanded", "secured", "doubled", "tripled"
]

# Weak or passive phrases that degrade resume impact
WEAK_VERB_PHRASES = [
    "responsible for", "helped with", "worked on", "assisted in", "tasked with",
    "duties included", "participated in", "served as", "was involved in", "handled"
]

# Overused buzzwords and vague clichés
BUZZWORDS = [
    "hard worker", "team player", "go-getter", "self-motivated", "detail-oriented",
    "results-driven", "synergy", "think outside the box", "dynamic", "fast-learner",
    "passionate", "strategic thinker", "proactive"
]


def detect_action_verbs(text: str) -> Dict[str, Any]:
    """
    Finds strong action verbs and flags weak passive phrasing.
    """
    text_lower = text.lower()
    found_strong = [verb for verb in STRONG_ACTION_VERBS if re.search(rf"\b{re.escape(verb)}\b", text_lower)]
    found_weak = [phrase for phrase in WEAK_VERB_PHRASES if re.search(rf"\b{re.escape(phrase)}\b", text_lower)]

    return {
        "strong_verbs_count": len(found_strong),
        "strong_verbs": sorted(found_strong),
        "weak_phrases_count": len(found_weak),
        "weak_phrases": sorted(found_weak)
    }


def detect_quantifiable_metrics(text: str) -> Dict[str, Any]:
    """
    Detects metrics like percentages (25%), dollar amounts ($150k),
    multipliers (3x), and statistical figures that prove tangible impact.
    """
    percentages = re.findall(r"\b\d+(?:\.\d+)?%", text)
    currencies = re.findall(r"[\$€£₹]\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*[kKmMbB]|(?:\s+million|\s+thousand|\s+billion))?", text)
    multipliers = re.findall(r"\b\d+(?:\.\d+)?x\b", text, re.IGNORECASE)
    large_numbers = re.findall(r"\b\d{2,}(?:,\d{3})+\+?\b|\b\d+\s*\+\s*(?:users|clients|customers|requests|transactions)\b", text, re.IGNORECASE)

    all_metrics = list(set(percentages + currencies + multipliers + large_numbers))

    return {
        "metrics_count": len(all_metrics),
        "percentages": list(set(percentages)),
        "currencies": list(set(currencies)),
        "multipliers": list(set(multipliers)),
        "all_metrics": all_metrics
    }


def detect_buzzwords(text: str) -> List[str]:
    """Identifies overused cliches that lack concrete evidence."""
    text_lower = text.lower()
    return [word for word in BUZZWORDS if re.search(rf"\b{re.escape(word)}\b", text_lower)]


def analyze_ats(text: str) -> Dict[str, Any]:
    """
    Performs comprehensive ATS compatibility analysis and computes a score from 0 to 100.
    """
    contact_info = extract_contact_info(text)
    sections = segment_sections(text)
    action_verb_data = detect_action_verbs(text)
    metrics_data = detect_quantifiable_metrics(text)
    buzzwords = detect_buzzwords(text)

    # Word count and bullet heuristics
    words = text.split()
    word_count = len(words)
    bullet_count = len(re.findall(r"(?:^|\n)\s*[•\-\*]\s+", text))

    # --- Scoring Breakdown ---
    # 1. Section Completeness (Max 20 pts)
    section_score = 0
    missing_sections = []
    if sections.get("experience"):
        section_score += 8
    else:
        missing_sections.append("Work Experience")

    if sections.get("skills"):
        section_score += 5
    else:
        missing_sections.append("Skills")

    if sections.get("education"):
        section_score += 4
    else:
        missing_sections.append("Education")

    if sections.get("summary") or sections.get("projects"):
        section_score += 3
    else:
        missing_sections.append("Summary or Projects")

    # 2. Contact Information Completeness (Max 15 pts)
    contact_score = 0
    missing_contact = []
    if contact_info.get("email"):
        contact_score += 6
    else:
        missing_contact.append("Email Address")

    if contact_info.get("phone"):
        contact_score += 5
    else:
        missing_contact.append("Phone Number")

    if contact_info.get("linkedin") or contact_info.get("github"):
        contact_score += 4
    else:
        missing_contact.append("LinkedIn or GitHub profile link")

    # 3. Action Verb Strength (Max 25 pts)
    strong_count = action_verb_data["strong_verbs_count"]
    weak_count = action_verb_data["weak_phrases_count"]
    if strong_count >= 10:
        action_verb_score = 25
    elif strong_count >= 6:
        action_verb_score = 20
    elif strong_count >= 3:
        action_verb_score = 14
    elif strong_count >= 1:
        action_verb_score = 8
    else:
        action_verb_score = 0
    # Penalty for weak phrasing
    action_verb_score = max(0, action_verb_score - (weak_count * 2))

    # 4. Quantifiable Metrics & Results (Max 25 pts)
    metric_count = metrics_data["metrics_count"]
    if metric_count >= 8:
        metric_score = 25
    elif metric_count >= 5:
        metric_score = 20
    elif metric_count >= 3:
        metric_score = 15
    elif metric_count >= 1:
        metric_score = 8
    else:
        metric_score = 0

    # 5. Length & Formatting (Max 15 pts)
    formatting_score = 0
    formatting_notes = []
    # Word count evaluation (Ideal: 400 - 900 words)
    if 400 <= word_count <= 950:
        formatting_score += 8
    elif 300 <= word_count < 400 or 950 < word_count <= 1300:
        formatting_score += 5
        formatting_notes.append("Resume length is slightly outside optimal range (400-900 words).")
    else:
        formatting_score += 2
        formatting_notes.append("Resume word count is either too brief (<300 words) or excessive (>1300 words).")

    # Bullet point density
    if bullet_count >= 6:
        formatting_score += 7
    elif bullet_count >= 3:
        formatting_score += 4
        formatting_notes.append("Consider converting dense paragraphs into structured bullet points.")
    else:
        formatting_score += 1
        formatting_notes.append("Very few bullet points detected. ATS scanners favor concise bulleted achievements.")

    # Penalty for buzzwords
    buzzword_penalty = min(5, len(buzzwords) * 2)

    total_score = max(0, min(100, (
        section_score +
        contact_score +
        action_verb_score +
        metric_score +
        formatting_score -
        buzzword_penalty
    )))

    # Actionable suggestions
    suggestions: List[str] = []
    if missing_sections:
        suggestions.append(f"Add missing core sections: {', '.join(missing_sections)}.")
    if missing_contact:
        suggestions.append(f"Include essential contact info: {', '.join(missing_contact)}.")
    if strong_count < 6:
        suggestions.append("Incorporate more strong action verbs (e.g. 'architected', 'spearheaded', 'automated').")
    if weak_count > 0:
        suggestions.append(f"Replace passive phrases ({', '.join(action_verb_data['weak_phrases'][:3])}) with active verbs.")
    if metric_count < 4:
        suggestions.append("Add quantifiable achievements (e.g. percentages, revenue saved, latency reduced).")
    if buzzwords:
        suggestions.append(f"Replace vague buzzwords ({', '.join(buzzwords[:3])}) with concrete demonstrations of skill.")
    suggestions.extend(formatting_notes)

    return {
        "total_ats_score": round(total_score, 1),
        "sub_scores": {
            "section_completeness": {"score": section_score, "max": 20},
            "contact_completeness": {"score": contact_score, "max": 15},
            "action_verbs": {"score": action_verb_score, "max": 25},
            "quantifiable_metrics": {"score": metric_score, "max": 25},
            "formatting_and_length": {"score": formatting_score, "max": 15},
            "buzzword_penalty": buzzword_penalty
        },
        "stats": {
            "word_count": word_count,
            "bullet_count": bullet_count,
            "strong_verbs_count": strong_count,
            "metrics_count": metric_count,
            "buzzwords_count": len(buzzwords)
        },
        "details": {
            "contact_info": contact_info,
            "action_verbs": action_verb_data,
            "metrics": metrics_data,
            "buzzwords": buzzwords,
            "detected_sections": [k for k, v in sections.items() if v and k != "other"]
        },
        "suggestions": suggestions
    }
