import os
import re
import math
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PAGES_DIR = os.path.join("..", "task1", "pages")
TOKENS_FILE = os.path.join("..", "task2", "tokens.txt")
LEMMAS_FILE = os.path.join("..", "task2", "lemmas.txt")
TFIDF_TERMS_DIR = "tfidf_terms"
TFIDF_LEMMAS_DIR = "tfidf_lemmas"

STOP_WORDS = set(stopwords.words("english"))
VALID_TOKEN_RE = re.compile(r'^[a-zA-Z]{2,}$')


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def get_doc_tokens(text):
    """Return list of all valid tokens in a document (with duplicates for counting)."""
    words = word_tokenize(text.lower())
    return [w for w in words if VALID_TOKEN_RE.match(w) and w not in STOP_WORDS]


def main():
    lemmatizer = WordNetLemmatizer()

    # Load valid tokens and lemmas from Task 2
    with open(TOKENS_FILE, "r", encoding="utf-8") as f:
        valid_tokens = set(line.strip() for line in f if line.strip())

    # Load lemma groups from Task 2
    lemma_groups = {}
    with open(LEMMAS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lemma = parts[0]
                tokens = parts[1:]
                lemma_groups[lemma] = tokens

    page_files = sorted(
        [f for f in os.listdir(PAGES_DIR) if f.endswith(".txt")],
        key=lambda x: int(x.replace(".txt", ""))
    )
    N = len(page_files)

    print(f"Processing {N} documents...")

    # First pass: collect document frequencies
    doc_term_freq = {}  # doc_id -> Counter of terms
    doc_lemma_freq = {}  # doc_id -> Counter of lemmas
    term_doc_count = defaultdict(int)  # term -> number of docs containing it
    lemma_doc_count = defaultdict(int)  # lemma -> number of docs containing it

    for filename in page_files:
        doc_id = int(filename.replace(".txt", ""))
        filepath = os.path.join(PAGES_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()

        text = extract_text_from_html(html)
        tokens = get_doc_tokens(text)

        # Term frequencies
        term_counter = Counter(t for t in tokens if t in valid_tokens)
        doc_term_freq[doc_id] = term_counter

        # Lemma frequencies
        lemma_counter = Counter()
        for t in tokens:
            lemma = lemmatizer.lemmatize(t)
            if lemma in lemma_groups:
                lemma_counter[lemma] += 1
        doc_lemma_freq[doc_id] = lemma_counter

        # Document frequency counts
        for term in set(term_counter.keys()):
            term_doc_count[term] += 1
        for lemma in set(lemma_counter.keys()):
            lemma_doc_count[lemma] += 1

    # Second pass: compute TF-IDF and write files
    for filename in page_files:
        doc_id = int(filename.replace(".txt", ""))

        # TF-IDF for terms
        term_counter = doc_term_freq[doc_id]
        total_terms = sum(term_counter.values()) if term_counter else 1

        term_tfidf = []
        for term, count in term_counter.items():
            tf = count / total_terms
            idf = math.log(N / term_doc_count[term]) if term_doc_count[term] > 0 else 0
            tfidf = tf * idf
            term_tfidf.append((term, idf, tfidf))

        term_tfidf.sort(key=lambda x: x[0])
        outpath = os.path.join(TFIDF_TERMS_DIR, f"{doc_id}.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            for term, idf, tfidf in term_tfidf:
                f.write(f"{term} {idf:.6f} {tfidf:.6f}\n")

        # TF-IDF for lemmas
        lemma_counter = doc_lemma_freq[doc_id]
        total_lemma_tokens = sum(lemma_counter.values()) if lemma_counter else 1

        lemma_tfidf = []
        for lemma, count in lemma_counter.items():
            tf = count / total_lemma_tokens
            idf = math.log(N / lemma_doc_count[lemma]) if lemma_doc_count[lemma] > 0 else 0
            tfidf = tf * idf
            lemma_tfidf.append((lemma, idf, tfidf))

        lemma_tfidf.sort(key=lambda x: x[0])
        outpath = os.path.join(TFIDF_LEMMAS_DIR, f"{doc_id}.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            for lemma, idf, tfidf in lemma_tfidf:
                f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

    print(f"TF-IDF term files saved to {TFIDF_TERMS_DIR}/")
    print(f"TF-IDF lemma files saved to {TFIDF_LEMMAS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
