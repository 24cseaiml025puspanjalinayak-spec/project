"""
Command-Line Interface (CLI) for AI Resume Analyzer
Usage:
    python cli.py --resume samples/sample_resume_swe.txt --jd samples/sample_jd_swe.txt
    python cli.py --sample
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path so src imports work smoothly
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.extractor import extract_text
from src.matcher import match_resume_to_job
from src.ats_analyzer import analyze_ats
from src.llm_analyzer import analyze_with_gemini
from src.report_generator import generate_markdown_report, generate_json_report


def run_cli():
    parser = argparse.ArgumentParser(
        description="AI-Based Resume Analyzer - Match resumes against job descriptions with ATS and AI feedback."
    )
    parser.add_argument("--resume", type=str, help="Path to resume file (.pdf, .docx, .txt)")
    parser.add_argument("--jd", type=str, help="Path to job description text file or string")
    parser.add_argument("--sample", action="store_true", help="Run with built-in Software Engineer sample")
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API Key (optional)")
    parser.add_argument("--export", type=str, default=None, help="File path to save the generated Markdown report")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()

    if args.sample or (not args.resume and not args.jd):
        print("\n[*] Running with built-in Software Engineer sample dataset...\n")
        resume_path = project_root / "samples" / "sample_resume_swe.txt"
        jd_path = project_root / "samples" / "sample_jd_swe.txt"
    else:
        if not args.resume:
            print("Error: --resume argument is required (or use --sample)")
            sys.exit(1)
        if not args.jd:
            print("Error: --jd argument is required (or use --sample)")
            sys.exit(1)
        resume_path = Path(args.resume)
        jd_path = Path(args.jd)

    # Load Resume
    if not resume_path.exists():
        print(f"Error: Resume file '{resume_path}' does not exist.")
        sys.exit(1)
    with open(resume_path, "rb") as f:
        resume_text = extract_text(f.read(), filename=resume_path.name)

    # Load Job Description
    if jd_path.exists():
        with open(jd_path, "r", encoding="utf-8", errors="replace") as f:
            jd_text = f.read()
    else:
        # Treat as raw text if not a file
        jd_text = str(args.jd)

    print(f"[File] Analyzing Resume: {resume_path.name}")
    print(f"[Target] Matching against Job Description: {jd_path.name if jd_path.exists() else 'Custom JD'}")
    print("-" * 65)

    # 1. Matching & Skill Gap Analysis
    match_data = match_resume_to_job(resume_text, jd_text)

    # 2. ATS Compatibility
    ats_data = analyze_ats(resume_text)

    # 3. AI / LLM Critique
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    llm_data = analyze_with_gemini(resume_text, jd_text, match_data, ats_data, api_key=api_key)

    if args.json:
        print(generate_json_report(resume_path.name, "Target Job", match_data, ats_data, llm_data))
        return

    # Formatted Console Output
    print(f"\n==================== SCORECARD ====================")
    print(f" Overall Match Score:         {match_data['composite_score']}%")
    print(f" ATS Health & Formatting:     {ats_data['total_ats_score']}/100")
    print(f" Exact Skill Overlap:         {match_data['skill_match_percentage']}%")
    print(f" TF-IDF Vocabulary Sim:       {match_data['tfidf_similarity']}%")
    print(f" Engine / Model:              {llm_data['model_used']}")
    print(f"===================================================\n")

    skill_gap = match_data["skill_gap"]
    print("[+] Matched Skills:")
    print("    " + (", ".join(skill_gap["matched_skills"]) if skill_gap["matched_skills"] else "None"))

    print("\n[-] Missing Skills (JD Requirements Missing in Resume):")
    print("    " + (", ".join(skill_gap["missing_skills"]) if skill_gap["missing_skills"] else "None (Full match!)"))

    print(f"\n[*] ATS Health Highlights:")
    print(f"    - Word Count: {ats_data['stats']['word_count']} | Bullets: {ats_data['stats']['bullet_count']}")
    print(f"    - Strong Action Verbs: {ats_data['stats']['strong_verbs_count']} detected")
    print(f"    - Quantifiable Metrics: {ats_data['stats']['metrics_count']} found")

    if ats_data["suggestions"]:
        print("\n[!] ATS Optimization Suggestions:")
        for sug in ats_data["suggestions"]:
            print(f"    * {sug}")

    print(f"\n[+] Executive Summary:")
    print(f"    {llm_data['executive_summary']}")

    print("\n[★] STAR Method Bullet Rewrite Sample:")
    rewrites = llm_data.get("star_bullet_rewrites", [])
    if rewrites:
        sample_rw = rewrites[0]
        print(f"    Original: \"{sample_rw.get('original', '')}\"")
        print(f"    Rewritten: \"{sample_rw.get('star_rewrite', '')}\"")

    if args.export:
        report_md = generate_markdown_report(resume_path.name, "Target Job", match_data, ats_data, llm_data)
        with open(args.export, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\n[Saved] Full report successfully exported to: {args.export}")

    print("\n>>> Analysis Complete! To explore interactively with the UI, run: streamlit run app.py\n")


if __name__ == "__main__":
    run_cli()
