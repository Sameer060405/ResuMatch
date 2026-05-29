"""
test_run.py — quick sanity check without needing a PDF.
Run with:  python test_run.py
"""

import pandas as pd
from preprocessing   import preprocess
from skill_extractor import extract_skills
from matcher         import rank_jobs

# Simulated resume text (paste anything here to test)
sample_resume = """
I am a data science enthusiast with strong Python programming skills.
I have worked with pandas, numpy, and matplotlib for data analysis.
I built machine learning models using scikit-learn and have experience
with SQL for querying databases. I also know statistics and have done
A/B testing. I am familiar with Git for version control.
"""

print("=" * 55)
print("STEP 1 — Preprocessed Resume Text")
print("=" * 55)
print(preprocess(sample_resume))

print("\n" + "=" * 55)
print("STEP 2 — Extracted Skills from Resume")
print("=" * 55)
skills = extract_skills(sample_resume)
print(", ".join(s.title() for s in skills))

print("\n" + "=" * 55)
print("STEP 3 — Job Match Results")
print("=" * 55)
jobs_df   = pd.read_csv("jobs.csv")
results, _ = rank_jobs(sample_resume, jobs_df)

for i, row in results.iterrows():
    print(f"\n#{i+1} {row['job_title']}")
    print(f"   Match Score  : {row['match_score']}%")
    print(f"   Missing Skills: {row['missing_skills']}")
