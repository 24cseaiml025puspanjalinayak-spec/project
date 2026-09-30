"""
Skills Taxonomy and Extraction Engine
Categorized skill databases, aliases, normalization, and regex-based boundary extraction.
"""

from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple

# Comprehensive taxonomy categorized by domain
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
        "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "Dart", "Elixir",
        "Haskell", "Bash", "Shell", "PowerShell", "Perl", "Lua", "Julia"
    ],
    "Frameworks & Web Tech": [
        "React", "Next.js", "Angular", "Vue.js", "Nuxt.js", "Node.js", "Express.js",
        "Django", "Flask", "FastAPI", "Spring Boot", "ASP.NET", "Ruby on Rails",
        "Laravel", "HTML5", "CSS3", "Tailwind CSS", "Bootstrap", "GraphQL",
        "REST API", "gRPC", "WebSockets", "Redux", "Svelte", "jQuery"
    ],
    "Cloud & DevOps": [
        "AWS", "Amazon Web Services", "Microsoft Azure", "Google Cloud", "GCP",
        "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitLab CI",
        "GitHub Actions", "Prometheus", "Grafana", "Helm", "Nginx", "Linux",
        "CI/CD", "Serverless", "CloudFormation", "ArgoCD", "Puppet", "Chef"
    ],
    "Data & AI/ML": [
        "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP",
        "Computer Vision", "LLMs", "Generative AI", "PyTorch", "TensorFlow",
        "Keras", "Scikit-Learn", "Pandas", "NumPy", "SciPy", "Apache Spark",
        "PySpark", "BigQuery", "Snowflake", "Databricks", "Apache Kafka",
        "Apache Airflow", "dbt", "Hadoop", "Tableau", "Power BI", "Data Mining",
        "Feature Engineering", "Vector Databases", "LangChain", "LlamaIndex"
    ],
    "Databases & Storage": [
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "DynamoDB", "Cassandra", "Neo4j", "Oracle", "Microsoft SQL Server",
        "SQLite", "Firebase", "Supabase", "ChromaDB", "Pinecone", "Milvus"
    ],
    "Architecture & Methodologies": [
        "Agile", "Scrum", "Kanban", "Microservices", "System Design",
        "Object-Oriented Programming", "OOP", "Test-Driven Development", "TDD",
        "Behavior-Driven Development", "BDD", "Unit Testing", "Integration Testing",
        "Design Patterns", "Event-Driven Architecture", "Domain-Driven Design",
        "Git", "GitHub", "GitLab", "Jira", "Clean Architecture"
    ],
    "Soft Skills": [
        "Leadership", "Mentoring", "Cross-Functional Collaboration",
        "Problem Solving", "Critical Thinking", "Communication", "Time Management",
        "Adaptability", "Stakeholder Management", "Project Management",
        "Conflict Resolution", "Public Speaking", "Analytical Skills"
    ]
}

# Skill synonyms and alias normalization
SKILL_ALIASES: Dict[str, str] = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "k8s": "Kubernetes",
    "reactjs": "React",
    "react.js": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "node": "Node.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "postgres": "PostgreSQL",
    "mongo": "MongoDB",
    "amazon web services": "AWS",
    "aws": "AWS",
    "google cloud platform": "Google Cloud",
    "gcp": "Google Cloud",
    "microsoft azure": "Microsoft Azure",
    "azure": "Microsoft Azure",
    "natural language processing": "NLP",
    "nlp": "NLP",
    "large language models": "LLMs",
    "llm": "LLMs",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "scikit learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "ci cd": "CI/CD",
    "ci/cd": "CI/CD",
    "continuous integration": "CI/CD",
    "github action": "GitHub Actions",
    "elastic search": "Elasticsearch",
    "ms sql": "Microsoft SQL Server",
    "oop": "Object-Oriented Programming",
    "tdd": "Test-Driven Development",
    "bdd": "Behavior-Driven Development"
}


def _build_regex_pattern(skill: str) -> re.Pattern:
    """
    Builds a case-insensitive regex pattern with safe word boundaries,
    correctly handling punctuation characters like C++, C#, .NET, CI/CD, Next.js.
    """
    escaped = re.escape(skill)
    # If the skill starts or ends with alphanumeric characters, apply word boundary \b
    prefix = r"\b" if re.match(r"^\w", skill) else r"(?:^|[\s,;/()\[\]])"
    suffix = r"\b" if re.match(r".*\w$", skill) else r"(?:$|[\s,;/()\[\]])"
    return re.compile(f"{prefix}{escaped}{suffix}", re.IGNORECASE)


# Precompile extraction patterns
_COMPILED_PATTERNS: List[Tuple[str, str, re.Pattern]] = []

for category, skills in SKILLS_TAXONOMY.items():
    for skill in skills:
        _COMPILED_PATTERNS.append((category, skill, _build_regex_pattern(skill)))

_ALIAS_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (canonical, _build_regex_pattern(alias))
    for alias, canonical in SKILL_ALIASES.items()
]


def extract_skills(text: str) -> Dict[str, List[str]]:
    """
    Extracts all known skills found in raw text, organized by category.
    Returns a dictionary mapping category names to lists of detected skills.
    """
    if not text:
        return {cat: [] for cat in SKILLS_TAXONOMY}

    detected_by_cat: Dict[str, Set[str]] = {cat: set() for cat in SKILLS_TAXONOMY}
    all_detected_canonical: Set[str] = set()

    # Check direct canonical taxonomy skills
    for category, skill, pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            detected_by_cat[category].add(skill)
            all_detected_canonical.add(skill.lower())

    # Check alias patterns and assign to appropriate category
    for canonical, pattern in _ALIAS_PATTERNS:
        if canonical.lower() not in all_detected_canonical and pattern.search(text):
            # Find which category this canonical skill belongs to
            for cat, skills in SKILLS_TAXONOMY.items():
                if any(s.lower() == canonical.lower() for s in skills):
                    detected_by_cat[cat].add(canonical)
                    all_detected_canonical.add(canonical.lower())
                    break

    # Convert sets to sorted lists
    return {cat: sorted(list(skills)) for cat, skills in detected_by_cat.items()}


def get_flat_skills(text: str) -> List[str]:
    """
    Extracts and returns a flat, sorted, deduplicated list of skills found in text.
    """
    categorized = extract_skills(text)
    flat_set: Set[str] = set()
    for skills in categorized.values():
        flat_set.update(skills)
    return sorted(list(flat_set))


def get_all_known_skills() -> Set[str]:
    """Returns a flat set of all known skills across all categories."""
    known: Set[str] = set()
    for skills in SKILLS_TAXONOMY.values():
        known.update(skills)
    return known
