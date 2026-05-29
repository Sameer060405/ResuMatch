"""
skill_extractor.py
------------------
Rule-based skill extraction using a predefined skill dictionary.
For each skill in the dictionary, we check whether it appears in
the input text (using whole-word matching for single-word skills
and substring matching for multi-word phrases).
"""

import re

# ------------------------------------------------------------------
# Predefined skill dictionary — add or remove skills as needed
# ------------------------------------------------------------------
SKILLS_DICT = {
    # Programming Languages
    "python", "r", "java", "c++", "scala", "julia", "javascript", "sql",

    # Machine Learning & AI
    # "natural language processing" removed — synonym of "nlp", keeping only one
    "machine learning", "deep learning", "neural networks", "nlp",
    "computer vision", "reinforcement learning",
    "transfer learning", "feature engineering", "model deployment",
    "classification", "clustering", "regression",

    # Libraries & Frameworks
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "numpy", "pandas", "matplotlib", "seaborn",
    "opencv", "nltk", "spacy", "hugging face",
    "xgboost", "lightgbm",

    # Databases & Big Data
    "mysql", "postgresql", "mongodb",
    "hadoop", "spark", "hive", "kafka", "elasticsearch",

    # Cloud Platforms
    "aws", "azure", "gcp", "google cloud",

    # MLOps & DevOps Tools
    "docker", "kubernetes", "git", "github",
    "mlflow", "airflow", "jenkins",

    # Visualization & BI
    "tableau", "power bi", "excel", "looker",

    # Statistics & Mathematics
    "statistics", "probability", "linear algebra",
    "calculus", "a/b testing", "hypothesis testing",

    # Soft Skills
    "communication", "problem solving", "teamwork",
    "leadership", "storytelling",
}


def extract_skills(text):
    """
    Scan `text` for every skill in SKILLS_DICT.

    Strategy:
      - Multi-word skills (e.g. "machine learning"): simple substring match
      - Single-word skills (e.g. "python"):           whole-word regex match
        so that "r" does not match inside "regression".

    Returns a sorted list of matched skill strings.
    """
    text_lower = text.lower()
    found = set()

    for skill in SKILLS_DICT:
        if " " in skill:
            # Multi-word: check plain substring
            if skill in text_lower:
                found.add(skill)
        else:
            # Single-word: require word boundaries
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.add(skill)

    return sorted(found)
