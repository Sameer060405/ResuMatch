"""
preprocessing.py
----------------
Handles all NLP preprocessing steps:
  1. PDF text extraction
  2. Text cleaning (lowercase, remove special chars)
  3. Tokenization, stopword removal, lemmatization
"""

import re
import nltk
import PyPDF2

# Download required NLTK data on first run
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer


def extract_text_from_pdf(pdf_file):
    """
    Read every page of a PDF and concatenate the text.
    pdf_file can be a file path string or a file-like object (Streamlit upload).
    """
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()


def clean_text(text):
    """
    Basic cleaning:
      - Lowercase everything
      - Remove numbers and special characters (keep only a-z and spaces)
      - Collapse multiple whitespace into one space
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # remove non-alpha chars
    text = re.sub(r'\s+',       ' ', text).strip()
    return text


def preprocess(text):
    """
    Full NLP pipeline applied to a single string:
      1. Clean text
      2. Tokenize into individual words
      3. Remove English stopwords (the, is, and …)
      4. Lemmatize each token (running → run, models → model)

    Returns a single space-joined string of processed tokens,
    ready for TF-IDF vectorization.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))

    cleaned = clean_text(text)
    tokens  = word_tokenize(cleaned)

    # Keep tokens that are not stopwords and are longer than 2 characters
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(tokens)
