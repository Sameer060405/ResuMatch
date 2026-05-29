# Resume Screening & Job Recommendation System

An NLP-powered tool that reads a resume PDF, extracts skills, compares the
resume against multiple job descriptions using TF-IDF and Cosine Similarity,
and recommends which skills to learn next.

---

## Problem Statement

Recruiters review hundreds of resumes per job opening. On the other side,
candidates apply for many roles without knowing how well their profile
actually matches each position. This project automates that matching process:

- Given a resume and a set of job descriptions, **how well does the candidate
  fit each role?**
- Which skills are **missing** from the resume for a specific role?
- What should the candidate **learn next** to improve their employability?

---

## Project Structure

```
project/
├── app.py              # Streamlit UI — user uploads PDF, results displayed here
├── preprocessing.py    # PDF text extraction + NLP cleaning pipeline
├── skill_extractor.py  # Rule-based skill extraction from text
├── matcher.py          # TF-IDF vectorization + Cosine Similarity ranking
├── jobs.csv            # 16 sample job descriptions across 5 roles
├── requirements.txt    # Python package dependencies
└── README.md           # This file
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| NLTK | Tokenization, stopword removal, lemmatization |
| Scikit-learn | TF-IDF Vectorizer and Cosine Similarity |
| PyPDF2 | Extract text from uploaded PDF resumes |
| Pandas | Load and manipulate the jobs CSV dataset |
| NumPy | Underlying array operations (used by scikit-learn) |
| Streamlit | Interactive web UI — no HTML/CSS required |

---

## Dataset

The file `jobs.csv` contains **16 manually crafted job descriptions** across
five common data-related roles:

| Role | Count |
|---|---|
| Data Scientist | 3 |
| Machine Learning Engineer | 3 |
| AI Engineer | 3 |
| Business Analyst | 3 |
| Data Analyst | 3 |
| (Bonus variants) | 1 |

Each row has three columns:
- `job_title` — human-readable job name
- `job_description` — a paragraph describing duties and required experience
- `required_skills` — comma-separated list of key skills

---

## NLP Pipeline Explained

Every text document (resume and job descriptions) passes through the same
four-step preprocessing pipeline before any comparison.

### Step 1 — Lowercase

```
"Experience with TensorFlow and PyTorch" → "experience with tensorflow and pytorch"
```

Ensures that "Python" and "python" are treated as the same word.

### Step 2 — Remove Special Characters

```
"C++ and A/B testing" → "c  and a b testing"
```

Regex removes anything that is not a-z or a space, reducing noise.

### Step 3 — Tokenization

```
"experience with sklearn" → ["experience", "with", "sklearn"]
```

Splits the string into a list of individual words (tokens) using NLTK's
`word_tokenize`.

### Step 4 — Stopword Removal

```
["experience", "with", "sklearn"] → ["experience", "sklearn"]
```

Removes common English words (the, is, and, with …) that carry no
meaningful signal for similarity comparison.

### Step 5 — Lemmatization

```
["models", "building", "running"] → ["model", "build", "run"]
```

Reduces words to their base (lemma) form so that "models" and "model"
are treated as the same feature. Uses NLTK's `WordNetLemmatizer`.

---

## TF-IDF Explained

**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**.
It converts a collection of text documents into a numerical matrix where
each cell represents how important a word is to a specific document
relative to the whole corpus.

### Term Frequency (TF)

How often does a word appear in this document?

```
TF(word, doc) = (count of word in doc) / (total words in doc)
```

### Inverse Document Frequency (IDF)

How rare is this word across all documents? Common words get
penalised; rare words that appear in only one document get boosted.

```
IDF(word) = log( (1 + total docs) / (1 + docs containing word) ) + 1
```

### TF-IDF Score

```
TF-IDF(word, doc) = TF × IDF
```

A high TF-IDF score means the word is frequent in this document
but rare across other documents — making it a strong signal.

**In this project:**
We combine the resume + all job descriptions into one corpus, fit a
single `TfidfVectorizer`, and transform every document into a vector.
This ensures all documents share the same vocabulary (feature space).

---

## Cosine Similarity Explained

Two TF-IDF vectors live in high-dimensional space. **Cosine Similarity**
measures the angle between them — not their magnitude, only their direction.

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Result of **1.0** → vectors point in the same direction → perfect match
- Result of **0.0** → vectors are perpendicular → no shared terms

**Why cosine and not Euclidean distance?**
Euclidean distance is sensitive to document length. A short resume and a
long job description would appear dissimilar just because of length, even
if they share all the same key terms. Cosine Similarity normalises for
length, making it ideal for text comparison.

**Match Percentage** is simply:

```
match_score = cosine_similarity(resume_vector, job_vector) × 100
```

---

## Skill Extraction Logic

Skill extraction uses a **rule-based dictionary approach**:

1. A `SKILLS_DICT` set in `skill_extractor.py` contains ~70 known skills
   (programming languages, ML libraries, cloud platforms, soft skills, etc.)
2. For each skill in the dictionary:
   - **Multi-word skills** (e.g., `"machine learning"`, `"a/b testing"`):
     plain substring search on the lowercased text.
   - **Single-word skills** (e.g., `"python"`, `"r"`, `"sql"`):
     whole-word regex match (`\bpython\b`) so that `"r"` does not
     accidentally match inside `"regression"`.

**Limitations of rule-based extraction:**
- Will miss skills not in the dictionary.
- Cannot understand synonyms (e.g., "sklearn" vs. "scikit-learn").
- Cannot infer implicit skills from project descriptions.

**Why not use Named Entity Recognition (NER)?**
NER models require training data labelled for skills, which adds significant
complexity. For a concise, interview-ready project, rule-based extraction
is transparent, fast, and easy to explain.

---

## How Match Score Is Computed — End to End

```
Resume PDF
    │
    ▼
extract_text_from_pdf()         ← PyPDF2 reads every page
    │
    ▼
preprocess()                    ← clean → tokenize → remove stopwords → lemmatize
    │
    ▼
TfidfVectorizer.fit_transform() ← builds vocabulary from resume + all jobs
    │
    ▼
cosine_similarity(              ← dot product of unit vectors
    resume_vector,
    job_vectors
)
    │
    ▼
Match Score (%) = similarity × 100
    │
    ▼
Missing Skills = job_skills − resume_skills   ← set difference
    │
    ▼
Rank + Display in Streamlit UI
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

### 3. Use the app

1. Open the browser tab Streamlit opens (usually `http://localhost:8501`).
2. Click **Browse files** and upload a text-based resume PDF.
3. Wait a few seconds for the analysis to complete.
4. Review your matched jobs, missing skills, and learning recommendations.

---

## Possible Future Improvements

| Improvement | Benefit |
|---|---|
| Replace rule-based skill extraction with a SpaCy NER model | Catch skills not in the dictionary |
| Use BERT / Sentence-Transformers for embeddings | Understand semantic meaning, not just word overlap |
| Add a real-world job dataset from LinkedIn/Indeed | More diverse and realistic matches |
| Implement synonym handling (sklearn = scikit-learn) | Reduce false "missing skill" results |
| Add resume scoring feedback (ATS-style) | Explain why the score is what it is |
| Store results in a database for multiple users | Multi-user support |
| Add file support beyond PDF (DOCX, TXT) | Broader usability |
| Deploy on Streamlit Cloud or Hugging Face Spaces | Share publicly with a URL |

---


