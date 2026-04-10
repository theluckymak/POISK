import json
import re
import sys
import os

INDEX_FILE = "inverted_index.json"


def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


def get_all_docs(index):
    all_docs = set()
    for docs in index.values():
        all_docs.update(docs)
    return all_docs


def tokenize_query(query):
    """Tokenize a boolean query into a list of tokens."""
    tokens = []
    i = 0
    query = query.strip()
    while i < len(query):
        if query[i] == '(':
            tokens.append('(')
            i += 1
        elif query[i] == ')':
            tokens.append(')')
            i += 1
        elif query[i].isspace():
            i += 1
        else:
            j = i
            while j < len(query) and query[j] not in '() \t':
                j += 1
            word = query[i:j]
            tokens.append(word)
            i = j
    return tokens


def parse_expression(tokens, pos, index, all_docs):
    """Parse OR expressions (lowest precedence)."""
    left, pos = parse_and(tokens, pos, index, all_docs)
    while pos < len(tokens) and tokens[pos].upper() == 'OR':
        pos += 1  # skip OR
        right, pos = parse_and(tokens, pos, index, all_docs)
        left = left | right
    return left, pos


def parse_and(tokens, pos, index, all_docs):
    """Parse AND expressions."""
    left, pos = parse_not(tokens, pos, index, all_docs)
    while pos < len(tokens) and tokens[pos].upper() == 'AND':
        pos += 1  # skip AND
        right, pos = parse_not(tokens, pos, index, all_docs)
        left = left & right
    return left, pos


def parse_not(tokens, pos, index, all_docs):
    """Parse NOT expressions."""
    if pos < len(tokens) and tokens[pos].upper() == 'NOT':
        pos += 1  # skip NOT
        operand, pos = parse_primary(tokens, pos, index, all_docs)
        return all_docs - operand, pos
    return parse_primary(tokens, pos, index, all_docs)


def parse_primary(tokens, pos, index, all_docs):
    """Parse primary: parenthesized expression or a single term."""
    if pos < len(tokens) and tokens[pos] == '(':
        pos += 1  # skip (
        result, pos = parse_expression(tokens, pos, index, all_docs)
        if pos < len(tokens) and tokens[pos] == ')':
            pos += 1  # skip )
        return result, pos
    elif pos < len(tokens):
        term = tokens[pos].lower()
        pos += 1
        return index.get(term, set()), pos
    return set(), pos


def boolean_search(query, index, all_docs):
    """Execute a boolean search query."""
    tokens = tokenize_query(query)
    if not tokens:
        return set()
    result, _ = parse_expression(tokens, 0, index, all_docs)
    return result


def main():
    if not os.path.exists(INDEX_FILE):
        print(f"Index file '{INDEX_FILE}' not found. Run build_index.py first.")
        sys.exit(1)

    print("Loading inverted index...")
    index = load_index()
    all_docs = get_all_docs(index)
    print(f"Loaded {len(index)} terms, {len(all_docs)} documents.\n")

    print("Boolean Search Engine")
    print("Operators: AND, OR, NOT, parentheses ()")
    print("Example: (python AND programming) OR (java AND NOT web)")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() == 'exit':
            break

        results = boolean_search(query, index, all_docs)
        if results:
            sorted_results = sorted(results)
            print(f"Found {len(results)} documents: {sorted_results}")
        else:
            print("No documents found.")
        print()


if __name__ == "__main__":
    main()
