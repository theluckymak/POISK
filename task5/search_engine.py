import os
import re
import math
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

PAGES_DIR = os.path.join("..", "task1", "pages")
INDEX_FILE = os.path.join("..", "task1", "index.txt")

STOP_WORDS = set(stopwords.words("english"))
VALID_TOKEN_RE = re.compile(r'^[a-zA-Z]{2,}$')
lemmatizer = WordNetLemmatizer()

# Global data loaded at startup
doc_vectors = {}      # doc_id -> sparse vector dict {term_idx: tfidf}
vocabulary = []       # list of all terms
term_to_idx = {}      # term -> index in vocabulary
doc_urls = {}         # doc_id -> URL
doc_titles = {}       # doc_id -> page title
idf_values = {}       # term -> idf
N = 0                 # total number of documents


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else "Untitled"
    for tag in soup(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return title, text


def tokenize(text):
    words = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(w) for w in words
            if VALID_TOKEN_RE.match(w) and w not in STOP_WORDS]


def build_vectors():
    global doc_vectors, vocabulary, term_to_idx, doc_urls, doc_titles, idf_values, N

    # Load URL index
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                doc_urls[int(parts[0])] = parts[1]

    page_files = sorted(
        [f for f in os.listdir(PAGES_DIR) if f.endswith(".txt")],
        key=lambda x: int(x.replace(".txt", ""))
    )
    N = len(page_files)

    # First pass: collect all terms and document frequencies
    doc_tokens = {}
    doc_freq = defaultdict(int)

    for filename in page_files:
        doc_id = int(filename.replace(".txt", ""))
        filepath = os.path.join(PAGES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        title, text = extract_text_from_html(html)
        doc_titles[doc_id] = title
        tokens = tokenize(text)
        doc_tokens[doc_id] = tokens
        unique_terms = set(tokens)
        for term in unique_terms:
            doc_freq[term] += 1

    # Build vocabulary
    vocabulary = sorted(doc_freq.keys())
    term_to_idx = {t: i for i, t in enumerate(vocabulary)}

    # Compute IDF
    for term in vocabulary:
        idf_values[term] = math.log(N / doc_freq[term])

    # Build TF-IDF vectors for each document
    for doc_id, tokens in doc_tokens.items():
        counter = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = {}
        for term, count in counter.items():
            if term in term_to_idx:
                tf = count / total
                tfidf = tf * idf_values[term]
                if tfidf > 0:
                    vector[term_to_idx[term]] = tfidf
        doc_vectors[doc_id] = vector

    print(f"Built vectors for {len(doc_vectors)} documents, vocabulary size: {len(vocabulary)}")


def query_to_vector(query_text):
    tokens = tokenize(query_text)
    counter = Counter(tokens)
    total = len(tokens) if tokens else 1
    vector = {}
    for term, count in counter.items():
        if term in term_to_idx:
            tf = count / total
            tfidf = tf * idf_values.get(term, 0)
            if tfidf > 0:
                vector[term_to_idx[term]] = tfidf
    return vector


def cosine_similarity(vec_a, vec_b):
    # Sparse dot product
    dot = 0.0
    for idx, val in vec_a.items():
        if idx in vec_b:
            dot += val * vec_b[idx]

    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) if vec_a else 0
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) if vec_b else 0

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(query_text, top_k=10):
    query_vec = query_to_vector(query_text)
    if not query_vec:
        return []

    scores = []
    for doc_id, doc_vec in doc_vectors.items():
        sim = cosine_similarity(query_vec, doc_vec)
        if sim > 0:
            scores.append((doc_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in scores[:top_k]:
        results.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "title": doc_titles.get(doc_id, f"Document {doc_id}"),
            "url": doc_urls.get(doc_id, "#"),
        })
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = search(query, top_k=10)
    return render_template("index.html", query=query, results=results)


if __name__ == "__main__":
    print("Building search index...")
    build_vectors()
    print("Starting web server on http://localhost:5000")
    app.run(debug=False, port=5000)
