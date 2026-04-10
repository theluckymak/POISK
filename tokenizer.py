import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

PAGES_DIR = "pages"
TOKENS_FILE = "tokens.txt"
LEMMAS_FILE = "lemmas.txt"

# English stop words (includes conjunctions, prepositions, articles, etc.)
STOP_WORDS = set(stopwords.words("english"))

# Regex: only pure alphabetic words, at least 2 chars
WORD_RE = re.compile(r"^[a-zA-Z]{2,}$")


def is_noise(word: str) -> bool:
    """Reject words with mixed letters+digits, markup fragments, etc."""
    if not WORD_RE.match(word):
        return True
    return False


def get_wordnet_pos(treebank_tag: str):
    """Map POS tag to WordNet POS for better lemmatization."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def extract_text(html: str) -> str:
    """Extract visible text from HTML, stripping scripts/styles."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def tokenize_and_filter(text: str) -> list[str]:
    """Tokenize text into clean, lowercase words."""
    raw_tokens = text.lower().split()
    filtered = []
    for t in raw_tokens:
        # Strip punctuation from edges
        t = t.strip(".,;:!?\"'()[]{}<>«»—–-_/\\|@#$%^&*+=~`")
        if not t:
            continue
        if is_noise(t):
            continue
        if t in STOP_WORDS:
            continue
        filtered.append(t)
    return filtered


def main():
    lemmatizer = WordNetLemmatizer()

    all_tokens = []

    # Read and process each page
    files = sorted(os.listdir(PAGES_DIR), key=lambda f: int(f.replace(".txt", "")) if f.replace(".txt", "").isdigit() else 0)
    for fname in files:
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(PAGES_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            html = f.read()
        text = extract_text(html)
        tokens = tokenize_and_filter(text)
        all_tokens.extend(tokens)
        print(f"  {fname}: {len(tokens)} tokens extracted")

    # Deduplicate while preserving order
    seen = set()
    unique_tokens = []
    for t in all_tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    print(f"\nTotal unique tokens: {len(unique_tokens)}")

    # Write tokens file
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        for t in unique_tokens:
            f.write(t + "\n")
    print(f"Saved {TOKENS_FILE}")

    # Lemmatize: POS-tag all tokens, then lemmatize with correct POS
    print("Lemmatizing...")
    tagged = pos_tag(unique_tokens)
    lemma_groups = defaultdict(set)
    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        lemma_groups[lemma].add(word)

    # Write lemmas file
    with open(LEMMAS_FILE, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_groups.keys()):
            tokens_str = " ".join(sorted(lemma_groups[lemma]))
            f.write(f"{lemma} {tokens_str}\n")
    print(f"Saved {LEMMAS_FILE} ({len(lemma_groups)} lemmas)")


if __name__ == "__main__":
    main()
