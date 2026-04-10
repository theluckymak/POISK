import os
import re
import json
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PAGES_DIR = os.path.join("..", "task1", "pages")
INDEX_FILE = "inverted_index.json"

STOP_WORDS = set(stopwords.words("english"))
VALID_TOKEN_RE = re.compile(r'^[a-zA-Z]{2,}$')


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def build_index():
    lemmatizer = WordNetLemmatizer()
    inverted_index = {}

    page_files = sorted(
        [f for f in os.listdir(PAGES_DIR) if f.endswith(".txt")],
        key=lambda x: int(x.replace(".txt", ""))
    )

    all_doc_ids = set()
    print(f"Building inverted index from {len(page_files)} pages...")

    for filename in page_files:
        doc_id = int(filename.replace(".txt", ""))
        all_doc_ids.add(doc_id)
        filepath = os.path.join(PAGES_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()

        text = extract_text_from_html(html)
        words = word_tokenize(text.lower())

        for w in words:
            if not VALID_TOKEN_RE.match(w):
                continue
            if w in STOP_WORDS:
                continue
            lemma = lemmatizer.lemmatize(w)
            if lemma not in inverted_index:
                inverted_index[lemma] = set()
            inverted_index[lemma].add(doc_id)

    # Convert sets to sorted lists for JSON
    index_serializable = {k: sorted(list(v)) for k, v in inverted_index.items()}

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_serializable, f, ensure_ascii=False, indent=1)

    print(f"Index contains {len(inverted_index)} terms across {len(all_doc_ids)} documents.")
    print(f"Saved to {INDEX_FILE}")

    return inverted_index, all_doc_ids


if __name__ == "__main__":
    build_index()
