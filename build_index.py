"""
Build an inverted index from crawled HTML pages.

Reads HTML files from pages/ directory, extracts text using BeautifulSoup,
tokenizes, and produces an inverted index saved as JSON.
"""

import os
import re
import json
from bs4 import BeautifulSoup

PAGES_DIR = "pages"
INDEX_OUTPUT = "inverted_index.json"


def extract_text(html_content: str) -> str:
    """Extract visible text from HTML, stripping scripts/styles."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphabetic tokens (≥2 chars)."""
    return [w for w in re.findall(r"[a-zA-Z]{2,}", text.lower())]


def build_index(pages_dir: str) -> dict:
    """
    Build inverted index: term -> sorted list of document IDs.
    Returns (index, doc_titles) tuple.
    """
    inverted_index: dict[str, set[int]] = {}
    doc_titles: dict[int, str] = {}

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

        # Extract page title from HTML
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else filename
        title = title.replace(" - Wikipedia", "").strip()
        doc_titles[doc_id] = title

        tokens = tokenize(text)
        unique_tokens = set(tokens)

        for token in unique_tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)

        print(f"  [{doc_id:3d}] {title} — {len(unique_tokens)} unique terms")

    # Convert sets to sorted lists for JSON serialization
    index_serializable = {
        term: sorted(list(doc_ids))
        for term, doc_ids in sorted(inverted_index.items())
    }

    return index_serializable, doc_titles


def main():
    print("Building inverted index from crawled pages...\n")

    index, doc_titles = build_index(PAGES_DIR)

    output = {
        "documents": {str(k): v for k, v in sorted(doc_titles.items())},
        "total_documents": len(doc_titles),
        "total_terms": len(index),
        "index": index,
    }

    with open(INDEX_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Documents indexed: {len(doc_titles)}")
    print(f"  Unique terms:      {len(index)}")
    print(f"  Index saved to:    {INDEX_OUTPUT}")


if __name__ == "__main__":
    main()
