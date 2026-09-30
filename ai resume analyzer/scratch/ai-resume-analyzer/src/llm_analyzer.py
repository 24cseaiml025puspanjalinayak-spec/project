"""
LLM Semantic Evaluator & Resume Coach
Uses Google Gemini (gemini-3.8-flash via google-genai SDK) for deep semantic critique,
STAR-format bullet point rewriting, and targeted interview preparation.
Features a robust offline heuristic fallback when no API key is provided.
"""

from __future__ import annotations
import os
import json
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()


def _get_gemini_client(api_key: Optional[str] = None):
    """Initializes Google GenAI client if API key is present."""
    effective_key = api_key or os.getenv("GEMINI_API_KEY")
    if not effective_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=effective_key)
    except Exception:
        return None


def generate_offline_feedback(
    resume_text: str,
    jd_text: str,
    match_data: Dict[str, Any],
    ats_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes rich, actionable recommendations using rule-based heuristics
    when running offline or without an active Gemini API key.
    """
    comp_score = match_data.get("composite_score", 0)
    missing_skills = match_data.get("skill_gap", {}).get("missing_skills", [])
    matched_skills = match_data.get("skill_gap", {}).get("matched_skills", [])
    strong_verbs = ats_data.get("details", {}).get("action_verbs", {}).get("strong_verbs", [])
    metrics = ats_data.get("details", {}).get("metrics", {}).get("all_metrics", [])

    # Synthesize fit summary
    if comp_score >= 75:
        summary = (
            f"Strong candidate alignment ({comp_score}% composite match). The resume reflects substantial "
            f"overlap with key competencies ({', '.join(matched_skills[:4]) if matched_skills else 'core requirements'}). "
            f"Focus on highlighting leadership and business outcomes to stand out."
        )
    elif comp_score >= 50:
        summary = (
            f"Moderate candidate alignment ({comp_score}% composite match). While foundational qualifications match "
            f"({', '.join(matched_skills[:3]) if matched_skills else 'several requirements'}), address critical gaps "
            f"in ({', '.join(missing_skills[:3]) if missing_skills else 'domain tooling'}) to improve interview callback rates."
        )
    else:
        summary = (
            f"Low candidate-job alignment ({comp_score}% composite match). Significant gaps detected against requirements. "
            f"Tailor the resume specifically for this position by emphasizing relevant projects and acquiring core skills like "
            f"{', '.join(missing_skills[:4]) if missing_skills else 'the required tech stack'}."
        )

    # Heuristic STAR bullet rewrites
    star_rewrites = [
        {
            "original": "Worked on backend microservices and helped with API response times.",
            "critique": "Lacks specific technologies, proactive action verbs, and quantifiable performance impact.",
            "star_rewrite": (
                "Architected and deployed 4 high-throughput Python/FastAPI microservices; "
                "optimized Redis caching and database query indexes to slash p99 API latency by 42% (580ms to 330ms)."
            )
        },
        {
            "original": "Responsible for managing cloud infrastructure and CI/CD pipelines.",
            "critique": "Uses passive phrasing ('responsible for') and omits scale and reliability metrics.",
            "star_rewrite": (
                "Orchestrated Kubernetes clusters and automated multi-stage GitHub Actions CI/CD pipelines, "
                "reducing deployment cycle times from 45 minutes to under 8 minutes with zero downtime across 12 services."
            )
        },
        {
            "original": "Assisted team in developing machine learning models for customer predictions.",
            "critique": "Uses weak verb 'assisted' and omits model architecture, dataset volume, and business value.",
            "star_rewrite": (
                "Engineered end-to-end customer churn prediction pipeline using Scikit-Learn and XGBoost on 2.4M records, "
                "achieving 89% ROC-AUC and empowering retention teams to preserve an estimated $340K in annual ARR."
            )
        }
    ]

    # Dynamic interview questions based on missing skills & gaps
    interview_questions = []
    if missing_skills:
        for skill in missing_skills[:4]:
            interview_questions.append({
                "category": f"Technical Competency ({skill})",
                "question": f"The job description highlights {skill}, which was not prominent on your resume. Can you describe your familiarity with {skill}, or a comparable technology you've mastered quickly?",
                "rationale": f"Targets skill gap identified in JD requirements."
            })

    interview_questions.extend([
        {
            "category": "System Architecture & Scale",
            "question": "Describe the most challenging technical bottleneck you diagnosed and resolved in production. What metrics did you monitor and what was the outcome?",
            "rationale": "Evaluates problem-solving depth and metric-driven engineering."
        },
        {
            "category": "Behavioral & Prioritization",
            "question": "Tell me about a time when business priorities shifted right before a major release. How did you realign technical deliverables and communicate with stakeholders?",
            "rationale": "Assesses adaptability and cross-functional leadership under pressure."
        }
    ])

    return {
        "is_llm_powered": False,
        "model_used": "Offline Heuristics Engine",
        "executive_summary": summary,
        "key_strengths": [
            f"Demonstrated proficiency in {skill}" for skill in matched_skills[:4]
        ] if matched_skills else ["Foundational background demonstrated in resume."],
        "critical_gaps": [
            f"Missing required qualification: {skill}" for skill in missing_skills[:5]
        ] if missing_skills else ["No major explicit technical skill gaps identified."],
        "star_bullet_rewrites": star_rewrites,
        "interview_questions": interview_questions,
        "notice": "Offline mode active. Provide a GEMINI_API_KEY in the sidebar or environment to unlock Gemini 3.8 Flash live AI insights."
    }


def analyze_with_gemini(
    resume_text: str,
    jd_text: str,
    match_data: Dict[str, Any],
    ats_data: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates resume against job description using Google Gemini (gemini-3.8-flash).
    Falls back gracefully to offline heuristic feedback if API key is absent or on network errors.
    """
    client = _get_gemini_client(api_key)
    if not client:
        return generate_offline_feedback(resume_text, jd_text, match_data, ats_data)

    matched = match_data.get("skill_gap", {}).get("matched_skills", [])
    missing = match_data.get("skill_gap", {}).get("missing_skills", [])
    comp_score = match_data.get("composite_score", 0)
    ats_score = ats_data.get("total_ats_score", 0)

    prompt = f"""
You are an expert Executive Recruiter and ATS Optimization Specialist.
Analyze the following candidate's resume against the target job description.

Candidate Resume Excerpt:
\"\"\"
{resume_text[:3500]}
\"\"\"

Target Job Description:
\"\"\"
{jd_text[:2500]}
\"\"\"

Known Metrics:
- Pre-computed Match Score: {comp_score}%
- Pre-computed ATS Score: {ats_score}/100
- Matched Skills Identified: {', '.join(matched) if matched else 'None'}
- Missing Critical Skills: {', '.join(missing) if missing else 'None'}

Please provide a structured JSON response with the following keys:
{{
  "executive_summary": "A 2-3 sentence executive assessment of fit, highlighting match level and top differentiator.",
  "key_strengths": ["3-4 specific strengths where candidate stands out for this exact role"],
  "critical_gaps": ["2-4 specific missing experiences, technologies, or risks the hiring manager will notice"],
  "star_bullet_rewrites": [
    {{
      "original": "A weak or typical bullet point found or inferred from the candidate's resume",
      "critique": "Specific explanation of why this bullet is weak (e.g. passive verb, lack of metrics)",
      "star_rewrite": "High-impact rewritten bullet using Situation-Task-Action-Result format with realistic metrics and strong verbs"
    }},
    {{
      "original": "Second weak bullet point",
      "critique": "Critique of second bullet",
      "star_rewrite": "High-impact rewritten bullet"
    }},
    {{
      "original": "Third weak bullet point",
      "critique": "Critique of third bullet",
      "star_rewrite": "High-impact rewritten bullet"
    }}
  ],
  "interview_questions": [
    {{
      "category": "Technical Gap / Architecture / Behavioral",
      "question": "Targeted interview question testing the candidate's specific background or identified gap",
      "rationale": "Why the recruiter should ask this question"
    }}
  ]
}}

Return ONLY valid JSON. Do not wrap in markdown quotes if possible, or use standard ```json ... ``` tags.
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )
        raw_text = interaction.output_text or ""

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            data = json.loads(json_match.group(0))
            data["is_llm_powered"] = True
            data["model_used"] = "Google Gemini 3.8 Flash"
            data["notice"] = "Live AI analysis powered by Google Gemini 3.8 Flash."
            return data
        else:
            # Fallback if parsing failed
            feedback = generate_offline_feedback(resume_text, jd_text, match_data, ats_data)
            feedback["notice"] = "Gemini responded with non-JSON format; showing parsed offline recommendations."
            return feedback

    except Exception as e:
        feedback = generate_offline_feedback(resume_text, jd_text, match_data, ats_data)
        feedback["notice"] = f"Gemini API request note ({str(e)[:80]}...); showing offline heuristics."
        return feedback
