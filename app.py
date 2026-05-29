"""
app.py
------
Streamlit front-end for the Resume Screening & Job Recommendation System.

Run with:
    streamlit run app.py

Flow:
    1. User uploads a resume PDF
    2. Text is extracted from the PDF
    3. Skills are extracted using the rule-based extractor
    4. Resume is compared against all jobs via TF-IDF + Cosine Similarity
    5. Jobs are ranked by match score
    6. Missing skills and learning recommendations are displayed
"""

import streamlit as st
import pandas    as pd

from preprocessing   import extract_text_from_pdf
from skill_extractor import extract_skills
from matcher         import rank_jobs

# ── Page Setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume Screening & Job Recommendation System")
st.markdown(
    "Upload your resume PDF and instantly see how well it matches "
    "various job roles — along with skills you should learn next."
)
st.markdown("---")


# ── Load Job Dataset ──────────────────────────────────────────────
@st.cache_data
def load_jobs():
    """Load and cache the job descriptions CSV."""
    return pd.read_csv("jobs.csv")

jobs_df = load_jobs()


# ── Sidebar Info ──────────────────────────────────────────────────
with st.sidebar:
    st.header("About This App")
    st.markdown(
        "**Tech Stack:** Python · NLTK · Scikit-learn · Streamlit\n\n"
        "**Method:** TF-IDF Vectorization + Cosine Similarity\n\n"
        f"**Jobs in dataset:** {len(jobs_df)}"
    )
    st.markdown("---")
    st.markdown("Upload a text-based PDF resume on the right to get started.")


# ── File Upload ───────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])

if uploaded_file is None:
    st.info("Waiting for a PDF resume to be uploaded …")
    st.stop()

# ── Processing ────────────────────────────────────────────────────
with st.spinner("Extracting and analysing your resume …"):
    # Step 1 – Extract raw text from the PDF
    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text.strip():
        st.error(
            "Could not extract text from this PDF. "
            "Please upload a text-based (not scanned) PDF."
        )
        st.stop()

    # Step 2 – Extract skills mentioned in the resume
    resume_skills = extract_skills(resume_text)

    # Step 3 – Rank all jobs by match score
    results_df, skills_set = rank_jobs(resume_text, jobs_df)

st.success("Analysis complete!")
st.markdown("---")

# ── Section 1: Extracted Skills ───────────────────────────────────
st.subheader("🛠️ Skills Detected in Your Resume")

if resume_skills:
    # Display skills as coloured badges
    badges = "  ".join([f"`{s.title()}`" for s in resume_skills])
    st.markdown(badges)
else:
    st.warning(
        "No recognised skills were found. "
        "Make sure your resume explicitly lists technologies and tools."
    )

st.markdown("---")

# ── Section 2: Ranked Job Matches ────────────────────────────────
st.subheader("🏆 Top Job Matches")

for rank, row in results_df.iterrows():
    score = row['match_score']

    # Choose a colour label based on score band
    if score >= 70:
        band = "🟢 Strong Match"
    elif score >= 40:
        band = "🟡 Moderate Match"
    else:
        band = "🔴 Low Match"

    header = f"#{rank + 1}  {row['job_title']}  —  {score}%  ({band})"

    with st.expander(header, expanded=(rank < 3)):
        # Visual progress bar
        st.progress(int(score))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Required Skills:**")
            st.write(row['required_skills'])

        with col2:
            st.markdown("**Missing Skills:**")
            if row['missing_skills'] == "None":
                st.success("You meet all the skill requirements!")
            else:
                st.warning(row['missing_skills'])

st.markdown("---")

# ── Section 3: Skill Recommendations ─────────────────────────────
st.subheader("📚 Recommended Skills to Learn")
st.markdown(
    "Based on the skills missing from your **top 5 job matches**, "
    "consider adding these to your skill set:"
)

# Aggregate missing skills across the top 5 results
top5 = results_df.head(5)
all_missing = set()
for _, row in top5.iterrows():
    if row['missing_skills'] != "None":
        for skill in row['missing_skills'].split(", "):
            all_missing.add(skill.strip())

if all_missing:
    # Show in two columns for readability
    missing_list = sorted(all_missing)
    col_a, col_b = st.columns(2)
    midpoint     = len(missing_list) // 2 + len(missing_list) % 2

    with col_a:
        for skill in missing_list[:midpoint]:
            st.markdown(f"- **{skill.title()}**")
    with col_b:
        for skill in missing_list[midpoint:]:
            st.markdown(f"- **{skill.title()}**")
else:
    st.success(
        "Impressive! Your resume already covers the key skills "
        "for your top matches."
    )
