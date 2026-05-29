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

## Resume Bullet Points

Use these on your resume under a Projects section:

```
Resume Screening & Job Recommendation System | Python, NLTK, Scikit-learn, Streamlit

• Built an end-to-end NLP pipeline that extracts text from resume PDFs using
  PyPDF2 and preprocesses it with NLTK (tokenization, stopword removal,
  lemmatization) to enable accurate text analysis.

• Implemented TF-IDF vectorization (Scikit-learn) and Cosine Similarity to
  quantitatively match resumes against 16 job descriptions across 5 job roles,
  ranking results by match percentage.

• Developed a rule-based skill extraction engine using a curated dictionary of
  70+ industry skills with regex whole-word matching to accurately identify
  technical skills in unstructured text.

• Identified skills gaps between candidate profiles and job requirements using
  set-difference logic, surfacing prioritised learning recommendations for
  the top 5 job matches.

• Designed an interactive Streamlit web application with progress bars, expandable
  job cards, and a two-column recommendation layout — requiring zero front-end
  (HTML/CSS/JS) code.
```

---

## Interview Questions & Detailed Answers

---

### Q1. What is TF-IDF and why did you use it instead of simple word counts?

**Answer:**

TF-IDF stands for Term Frequency–Inverse Document Frequency. It is a
numerical statistic that reflects how important a word is to a specific
document in a corpus.

- **TF (Term Frequency):** How often does a word appear in a document?
  Normalised by dividing by the total number of words in that document.
- **IDF (Inverse Document Frequency):** The log of the total number of
  documents divided by the number of documents that contain the word.
  Words that appear in many documents (like "the", "experience") get a
  low IDF weight. Words unique to a few documents (like "pytorch") get
  a high weight.
- **TF-IDF = TF × IDF.**

I used TF-IDF instead of raw word counts because raw counts favour longer
documents and common words unfairly. TF-IDF normalises both problems:
length is handled by TF, and common words are down-weighted by IDF. This
gives more meaningful feature vectors for comparing a resume to job descriptions.

---

### Q2. What is Cosine Similarity and why is it better than Euclidean distance for text?

**Answer:**

Cosine Similarity measures the cosine of the angle between two vectors:

```
similarity = (A · B) / (||A|| × ||B||)
```

The result ranges from 0 (no overlap) to 1 (identical direction).

For text, we care about the **direction** of the vector (which words are
present and how prominent they are), not the **magnitude** (how long
the document is). A short resume and a long job description can be
very similar in content but very different in length.

Euclidean distance would penalise the length difference and return a
large distance even when both documents cover the same topics. Cosine
Similarity normalises for length, making it the standard choice for
document similarity tasks.

---

### Q3. Walk me through the NLP preprocessing pipeline you built.

**Answer:**

The pipeline has five steps, implemented in `preprocessing.py`:

1. **Lowercasing** — `text.lower()` — ensures case-insensitive comparison.
   "Python" and "python" become the same token.

2. **Remove special characters** — regex `re.sub(r'[^a-z\s]', ' ', text)` —
   strips punctuation, numbers, and symbols. This reduces vocabulary noise.

3. **Tokenization** — NLTK's `word_tokenize()` splits the string into
   individual word tokens.

4. **Stopword removal** — NLTK's `stopwords.words('english')` provides a
   list of ~180 common words (the, is, and, with…). Removing them ensures
   the TF-IDF matrix focuses on meaningful content words.

5. **Lemmatization** — NLTK's `WordNetLemmatizer` reduces words to their
   dictionary base form. "models" → "model", "running" → "run". This
   prevents the same concept from occupying multiple features in the
   TF-IDF vocabulary.

The result is a clean string of meaningful tokens ready for vectorization.

---

### Q4. How did you extract skills from the resume? Why rule-based and not ML-based?

**Answer:**

I used a rule-based dictionary approach. A `SKILLS_DICT` set contains
~70 known technical and soft skills. For each skill in the dictionary:

- **Multi-word skills** (e.g., "machine learning"): plain substring match
  on the lowercased text.
- **Single-word skills** (e.g., "r", "python"): whole-word regex match
  using `\b` word boundaries to prevent "r" from matching inside
  "regression".

I chose rule-based extraction over ML-based NER (Named Entity Recognition)
because:
- It is **transparent** — anyone can read the dictionary and understand it.
- It is **fast** — no model loading overhead.
- It is **easy to extend** — just add a skill to the dictionary.
- An NER model requires labelled training data for the "skill" entity type,
  which significantly increases project complexity without adding much value
  at this scale.

The trade-off is that it will miss skills not in the dictionary and cannot
handle synonyms automatically.

---

### Q5. What is lemmatization and how does it differ from stemming?

**Answer:**

Both techniques reduce a word to a base form, but they work differently:

- **Stemming** chops off word endings using simple rules:
  `"running" → "run"`, `"studies" → "studi"` (not always a real word).
  It is fast but crude and can produce non-dictionary tokens.

- **Lemmatization** uses a vocabulary and morphological analysis to
  return the actual dictionary base form (lemma):
  `"running" → "run"`, `"studies" → "study"`, `"better" → "good"`.
  It is slower but produces valid words that are easier to interpret.

In this project, I used NLTK's `WordNetLemmatizer` because the resulting
tokens remain readable and the vocabulary produced for TF-IDF is cleaner.

---

### Q6. How exactly is the match score computed?

**Answer:**

The full pipeline:

1. Both the resume and all job descriptions are preprocessed (cleaned,
   tokenized, stopwords removed, lemmatized).
2. All preprocessed texts are combined into a single corpus.
3. Scikit-learn's `TfidfVectorizer` is fitted on this corpus, building a
   shared vocabulary and computing TF-IDF weights for every document.
4. The resume becomes row 0 of the TF-IDF matrix; each job description
   occupies one subsequent row.
5. `cosine_similarity(resume_vector, job_vectors)` computes the similarity
   between the resume vector and every job vector.
6. Each similarity score (0–1) is multiplied by 100 to produce a
   percentage match score.
7. Results are sorted in descending order.

The key design choice is fitting the vectorizer on the combined corpus
(not just job descriptions) so the vocabulary includes terms from the
resume as well.

---

### Q7. What are stopwords, and what would happen if you did not remove them?

**Answer:**

Stopwords are common English words that appear in almost every document:
"the", "is", "and", "with", "of", "in", etc. NLTK provides a list of
about 180 English stopwords.

If we did not remove them:
- They would dominate the TF-IDF vocabulary because they appear in every
  document with high frequency.
- However, their IDF weight would be very low (close to zero) because they
  appear in every document, so in TF-IDF they are naturally down-weighted.
- Removing them still reduces vocabulary size, speeds up vectorization,
  and makes the feature space cleaner and more interpretable.

In practice, both approaches give similar results, but explicit removal is
considered good practice and is more efficient.

---

### Q8. How did you find the missing skills for each job?

**Answer:**

Python's set arithmetic makes this very clean:

```python
resume_skills = set(extract_skills(resume_text))          # e.g. {python, sql, pandas}
job_skills    = set(extract_skills(job_description_text)) # e.g. {python, sql, pytorch, docker}

missing = job_skills - resume_skills  # set difference → {pytorch, docker}
```

The set difference operator (`-`) returns elements that are in the job's
skill set but not in the resume's skill set. These are exactly the skills
the candidate lacks for that specific role.

The results are then sorted alphabetically and stored in the results DataFrame.

---

### Q9. What are the limitations of your approach and how would you improve it?

**Answer:**

**Limitations:**

1. **Vocabulary mismatch** — TF-IDF is keyword-based. If the resume says
   "sklearn" and the job says "scikit-learn", they are treated as different
   words even though they mean the same thing.

2. **No semantic understanding** — TF-IDF cannot understand that "deep
   learning" and "neural networks" are closely related concepts.

3. **Rule-based skill extraction misses unlisted skills** — Any skill not
   in `SKILLS_DICT` will be silently ignored.

4. **Small dataset** — 16 job descriptions is enough for a demo but not
   realistic for production use.

5. **No resume structure parsing** — The whole resume is treated as one blob
   of text; sections like Experience, Education, Projects are not distinguished.

**Improvements:**

1. Use **Sentence-BERT** or **OpenAI embeddings** to capture semantic
   similarity beyond keyword overlap.
2. Train a **SpaCy NER** model on a skills dataset (e.g., ESCO) for
   better skill extraction.
3. Integrate a **synonym dictionary** so "sklearn" maps to "scikit-learn".
4. Use a **real job postings dataset** from LinkedIn or Kaggle.
5. Parse resume sections separately to weight experience more than education.

---

### Q10. Why did you use Streamlit instead of Flask or Django?

**Answer:**

For a data science or ML project, Streamlit is the fastest path from
Python code to an interactive web app:

- **No front-end knowledge required** — no HTML, CSS, or JavaScript.
- **Widgets like `st.file_uploader`, `st.progress`, `st.expander`** are
  one-liners that would take dozens of lines in Flask + HTML templates.
- **`st.cache_data`** provides built-in caching for data loading.
- **Rapid prototyping** — a Streamlit app can be built in hours.

Flask or Django would be the right choice if the project needed a REST
API, user authentication, a database backend, or full control over the
front-end. For a data science portfolio project demonstrating NLP skills,
Streamlit is the industry-standard choice.
