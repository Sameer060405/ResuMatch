"""
matcher.py
----------
Core matching logic:
  1. Preprocess resume and all job descriptions
  2. Build a TF-IDF matrix over the combined corpus
  3. Compute Cosine Similarity between the resume vector and each job vector
  4. Identify missing skills per job
  5. Return a ranked DataFrame (highest match first)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise      import cosine_similarity

from preprocessing    import preprocess
from skill_extractor  import extract_skills


def compute_match_scores(resume_text, job_descriptions):
    """
    Build TF-IDF vectors for the resume and every job description,
    then return the cosine similarity of the resume against each job.

    Parameters
    ----------
    resume_text      : raw resume string
    job_descriptions : list of raw job description strings

    Returns
    -------
    scores : numpy array of shape (n_jobs,) with values in [0, 1]
    """
    # Preprocess all texts before vectorizing
    preprocessed_resume = preprocess(resume_text)
    preprocessed_jobs   = [preprocess(desc) for desc in job_descriptions]

    # Combine into one corpus so TF-IDF vocabulary is built from all documents
    corpus = [preprocessed_resume] + preprocessed_jobs

    vectorizer  = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Row 0 = resume, rows 1..n = jobs
    resume_vector = tfidf_matrix[0]
    job_vectors   = tfidf_matrix[1:]

    # cosine_similarity returns a (1, n_jobs) array; flatten to 1-D
    scores = cosine_similarity(resume_vector, job_vectors)[0]
    return scores


def rank_jobs(resume_text, jobs_df):
    """
    Compare the resume against every job in jobs_df and return
    a ranked result table.

    Parameters
    ----------
    resume_text : raw resume string
    jobs_df     : DataFrame with columns [job_title, job_description, required_skills]

    Returns
    -------
    result_df    : DataFrame sorted by match_score descending, columns:
                   [job_title, match_score, required_skills, missing_skills]
    resume_skills: set of skill strings found in the resume
    """
    resume_skills    = set(extract_skills(resume_text))
    job_descriptions = jobs_df['job_description'].tolist()

    raw_scores = compute_match_scores(resume_text, job_descriptions)

    # --- Normalize TF-IDF Scores ---
    # Raw cosine similarity on short documents is always small (0.05 – 0.35).
    # Normalizing spreads scores to [0, 1] so TF-IDF reflects *relative* closeness:
    # "which job's description is most similar to this resume, compared to all others?"
    # This makes TF-IDF a meaningful 50% contributor instead of a low-weight noise term.
    score_min = raw_scores.min()
    score_max = raw_scores.max()
    if score_max > score_min:
        normalized_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        normalized_scores = raw_scores  # all jobs equally similar — no normalization needed

    results = []
    for i, (_, row) in enumerate(jobs_df.iterrows()):
        # Combine description + required_skills column for skill extraction
        job_text   = row['job_description'] + " " + row['required_skills']
        job_skills = set(extract_skills(job_text))

        # Skills the job requires that the resume does not mention
        missing = sorted(job_skills - resume_skills)

        # --- Skill Overlap Ratio ---
        # How many of the job's required skills does the resume already have?
        # Formula: matched_skills / total_job_skills
        if job_skills:
            skill_overlap = len(resume_skills & job_skills) / len(job_skills)
        else:
            skill_overlap = 0.0

        # --- Final Score ---
        # Skill overlap is the CEILING — the maximum score possible for this job.
        # Normalized TF-IDF determines how much of that ceiling you achieve.
        #
        #   normalized_tfidf = 1.0 (best content match) → you get 100% of skill_overlap
        #   normalized_tfidf = 0.0 (worst content match) → you get 50% of skill_overlap
        #
        # This keeps TF-IDF as a meaningful NLP signal without letting a strong
        # text match inflate the score when skill coverage is actually low.
        hybrid_score = skill_overlap * (0.5 + 0.5 * normalized_scores[i])

        results.append({
            'job_title'      : row['job_title'],
            'match_score'    : round(hybrid_score * 100, 2),  # percentage
            'required_skills': row['required_skills'],
            'missing_skills' : ", ".join(missing) if missing else "None",
        })

    result_df = pd.DataFrame(results)
    result_df = (result_df
                 .sort_values('match_score', ascending=False)
                 .reset_index(drop=True))

    return result_df, resume_skills
