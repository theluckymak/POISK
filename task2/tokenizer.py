import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PAGES_DIR = os.path.join("..", "task1", "pages")
TOKENS_FILE = "tokens.txt"
LEMMAS_FILE = "lemmas.txt"

STOP_WORDS = set(stopwords.words("english"))

# Regex: only pure alphabetic words, 2+ chars
VALID_TOKEN_RE = re.compile(r'^[a-zA-Z]{2,}$')


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "link", "meta", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def tokenize_and_clean(text):
    words = word_tokenize(text.lower())
    tokens = set()
    for w in words:
        if not VALID_TOKEN_RE.match(w):
            continue
        if w in STOP_WORDS:
            continue
        tokens.add(w)
    return tokens


def main():
    lemmatizer = WordNetLemmatizer()

    all_tokens = set()
    page_files = sorted(
        [f for f in os.listdir(PAGES_DIR) if f.endswith(".txt")],
        key=lambda x: int(x.replace(".txt", ""))
    )

    print(f"Processing {len(page_files)} pages...")

    for filename in page_files:
        filepath = os.path.join(PAGES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text_from_html(html)
        tokens = tokenize_and_clean(text)
        all_tokens.update(tokens)

    print(f"Total unique tokens: {len(all_tokens)}")

    # Group tokens by lemma
    lemma_to_tokens = defaultdict(set)
    for token in all_tokens:
        lemma = lemmatizer.lemmatize(token)
        lemma_to_tokens[lemma].add(token)

    # Write tokens.txt
    sorted_tokens = sorted(all_tokens)
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        for token in sorted_tokens:
            f.write(token + "\n")

    # Write lemmas.txt
    with open(LEMMAS_FILE, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_to_tokens.keys()):
            tokens_list = sorted(lemma_to_tokens[lemma])
            f.write(lemma + " " + " ".join(tokens_list) + "\n")

    print(f"Unique lemmas: {len(lemma_to_tokens)}")
    print(f"Saved: {TOKENS_FILE}, {LEMMAS_FILE}")


if __name__ == "__main__":
    main()
