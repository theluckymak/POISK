"""
Vector Search Engine using TF-IDF and Cosine Similarity.

Builds TF-IDF vectors from crawled Wikipedia HTML pages,
then ranks documents by cosine similarity to a free-text query.

Usage:
    python vector_search.py                  # interactive mode
    python vector_search.py "your query"     # single query mode
"""

import os
import re
import sys
import json
import math
import time
from collections import Counter

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────

PAGES_DIR = "pages"
INDEX_FILE = "index.txt"
TOP_K = 10  # default number of results to show


# ─── Text extraction ────────────────────────────────────────────────────────

def extract_text(html_content: str) -> str:
    """Extract visible text from HTML, stripping scripts/styles/nav."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript",
                     "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_title(html_content: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        return title.replace(" - Wikipedia", "").strip()
    return "Unknown"


# ─── Document loading ───────────────────────────────────────────────────────

def load_documents(pages_dir: str) -> tuple[list[int], list[str], list[str]]:
    """
    Load all crawled pages, extract text and titles.
    Returns (doc_ids, texts, titles).
    """
    doc_ids = []
    texts = []
    titles = []

    files = sorted(
        [f for f in os.listdir(pages_dir) if f.endswith(".txt")],
        key=lambda f: int(f.replace(".txt", ""))
    )

    for filename in files:
        doc_id = int(filename.replace(".txt", ""))
        filepath = os.path.join(pages_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()

        text = extract_text(html)
        title = extract_title(html)

        doc_ids.append(doc_id)
        texts.append(text)
        titles.append(title)

    return doc_ids, texts, titles


# ─── URL mapping ────────────────────────────────────────────────────────────

def load_url_map(index_file: str) -> dict[int, str]:
    """Load doc_id -> URL mapping from index.txt."""
    url_map = {}
    if not os.path.exists(index_file):
        return url_map
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                parts = line.split("\t", 1)
                url_map[int(parts[0])] = parts[1]
    return url_map


# ─── Vector Search Engine ───────────────────────────────────────────────────

class VectorSearchEngine:
    """
    TF-IDF based vector search engine with cosine similarity ranking.
    """

    def __init__(self, pages_dir: str = PAGES_DIR, index_file: str = INDEX_FILE):
        print("Loading and indexing documents...")
        start = time.time()

        self.doc_ids, texts, self.titles = load_documents(pages_dir)
        self.url_map = load_url_map(index_file)
        self.num_docs = len(self.doc_ids)

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            token_pattern=r"[a-zA-Z]{2,}",
            sublinear_tf=True,       # use 1 + log(tf) instead of raw tf
            norm="l2",               # L2-normalize each document vector
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        elapsed = time.time() - start
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"Indexed {self.num_docs} documents, {vocab_size} terms "
              f"in {elapsed:.1f}s\n")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Search for documents matching the query.
        Returns list of {doc_id, title, url, score} dicts, ranked by score.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                break
            doc_id = self.doc_ids[idx]
            results.append({
                "doc_id": doc_id,
                "title": self.titles[idx],
                "url": self.url_map.get(doc_id, ""),
                "score": float(score),
            })

        return results

    def format_results(self, results: list[dict]) -> str:
        """Format search results for display."""
        if not results:
            return "  No relevant documents found."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"  {i:2d}. [{r['doc_id']:3d}] {r['title']}"
                f"  (score: {r['score']:.4f})"
            )
            if r["url"]:
                lines.append(f"       {r['url']}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return statistics about the index."""
        vocab = self.vectorizer.get_feature_names_out()
        return {
            "num_documents": self.num_docs,
            "vocabulary_size": len(vocab),
            "matrix_shape": self.tfidf_matrix.shape,
            "sparsity": 1.0 - (self.tfidf_matrix.nnz /
                                (self.tfidf_matrix.shape[0] *
                                 self.tfidf_matrix.shape[1])),
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    engine = VectorSearchEngine()

    stats = engine.get_stats()
    print("Vector Search Engine (TF-IDF + Cosine Similarity)")
    print(f"  Documents:   {stats['num_documents']}")
    print(f"  Vocabulary:  {stats['vocabulary_size']} terms")
    print(f"  Matrix:      {stats['matrix_shape'][0]} x {stats['matrix_shape'][1]}")
    print(f"  Sparsity:    {stats['sparsity']:.2%}")
    print()
    print("Enter a query in natural language. Type 'quit' or 'exit' to stop.")
    print()

    # Single query mode via CLI argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}\n")
        results = engine.search(query)
        print(f"Top {len(results)} results:\n")
        print(engine.format_results(results))
        return

    # Interactive mode
    while True:
        try:
            query = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Bye!")
            break

        results = engine.search(query)
        print(f"\nTop {len(results)} results:\n")
        print(engine.format_results(results))
        print()


if __name__ == "__main__":
    main()
