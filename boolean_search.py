"""
Boolean Search Engine using an inverted index.

Supports AND, OR, NOT operators and parenthesized sub-expressions.
Query is entered as a string at runtime (not hardcoded).

Grammar (recursive descent):
    query      → or_expr
    or_expr    → and_expr ( 'OR' and_expr )*
    and_expr   → not_expr ( 'AND' not_expr )*
    not_expr   → 'NOT' not_expr | primary
    primary    → '(' or_expr ')' | TERM
"""

import json
import re
import sys

INDEX_FILE = "inverted_index.json"


# ─── Tokenizer ───────────────────────────────────────────────────────────────

class Token:
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    TERM = "TERM"
    EOF = "EOF"

    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


def tokenize_query(query: str) -> list[Token]:
    """Split a Boolean query string into tokens."""
    tokens = []
    i = 0
    while i < len(query):
        ch = query[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "(":
            tokens.append(Token(Token.LPAREN, "("))
            i += 1
            continue

        if ch == ")":
            tokens.append(Token(Token.RPAREN, ")"))
            i += 1
            continue

        # Read a word
        m = re.match(r"[A-Za-z0-9_]+", query[i:])
        if m:
            word = m.group(0)
            upper = word.upper()
            if upper == "AND":
                tokens.append(Token(Token.AND, "AND"))
            elif upper == "OR":
                tokens.append(Token(Token.OR, "OR"))
            elif upper == "NOT":
                tokens.append(Token(Token.NOT, "NOT"))
            else:
                tokens.append(Token(Token.TERM, word.lower()))
            i += len(word)
            continue

        raise ValueError(f"Unexpected character at position {i}: {ch!r}")

    tokens.append(Token(Token.EOF, ""))
    return tokens


# ─── Parser (Recursive Descent) ─────────────────────────────────────────────

class Parser:
    """
    Parses tokenized Boolean query into a result set of document IDs.
    Uses the inverted index to resolve terms and applies set operations.
    """

    def __init__(self, tokens: list[Token], index: dict, all_docs: set[int]):
        self.tokens = tokens
        self.pos = 0
        self.index = index
        self.all_docs = all_docs

    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected_type: str) -> Token:
        tok = self.current()
        if tok.type != expected_type:
            raise ValueError(
                f"Expected {expected_type} but got {tok.type} ({tok.value!r}) "
                f"at position {self.pos}"
            )
        self.pos += 1
        return tok

    def parse(self) -> set[int]:
        result = self.or_expr()
        if self.current().type != Token.EOF:
            raise ValueError(
                f"Unexpected token after query end: {self.current()}"
            )
        return result

    def or_expr(self) -> set[int]:
        """or_expr → and_expr ( 'OR' and_expr )*"""
        result = self.and_expr()
        while self.current().type == Token.OR:
            self.consume(Token.OR)
            right = self.and_expr()
            result = result | right
        return result

    def and_expr(self) -> set[int]:
        """and_expr → not_expr ( 'AND' not_expr )*"""
        result = self.not_expr()
        while self.current().type == Token.AND:
            self.consume(Token.AND)
            right = self.not_expr()
            result = result & right
        return result

    def not_expr(self) -> set[int]:
        """not_expr → 'NOT' not_expr | primary"""
        if self.current().type == Token.NOT:
            self.consume(Token.NOT)
            operand = self.not_expr()
            return self.all_docs - operand
        return self.primary()

    def primary(self) -> set[int]:
        """primary → '(' or_expr ')' | TERM"""
        tok = self.current()

        if tok.type == Token.LPAREN:
            self.consume(Token.LPAREN)
            result = self.or_expr()
            self.consume(Token.RPAREN)
            return result

        if tok.type == Token.TERM:
            self.consume(Token.TERM)
            return set(self.index.get(tok.value, []))

        raise ValueError(
            f"Unexpected token in expression: {tok}"
        )


# ─── Search Engine ───────────────────────────────────────────────────────────

class BooleanSearchEngine:
    def __init__(self, index_path: str):
        print(f"Loading index from {index_path}...")
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.index: dict[str, list[int]] = data["index"]
        self.documents: dict[str, str] = data["documents"]
        self.all_docs: set[int] = {int(k) for k in self.documents}
        self.total_terms = data["total_terms"]
        self.total_docs = data["total_documents"]

        print(f"Loaded {self.total_docs} documents, {self.total_terms} terms.\n")

    def search(self, query: str) -> set[int]:
        tokens = tokenize_query(query)
        parser = Parser(tokens, self.index, self.all_docs)
        return parser.parse()

    def format_results(self, doc_ids: set[int]) -> str:
        if not doc_ids:
            return "  No documents found."
        lines = []
        for doc_id in sorted(doc_ids):
            title = self.documents.get(str(doc_id), f"Document {doc_id}")
            lines.append(f"  [{doc_id:3d}] {title}")
        return "\n".join(lines)


def main():
    engine = BooleanSearchEngine(INDEX_FILE)

    print("Boolean Search Engine")
    print("Operators: AND, OR, NOT  |  Parentheses: ( )")
    print("Type 'quit' or 'exit' to stop.\n")

    # If a query was passed as CLI argument, run it and exit
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        try:
            results = engine.search(query)
            print(f"Found {len(results)} document(s):\n")
            print(engine.format_results(results))
        except ValueError as e:
            print(f"Error: {e}")
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

        try:
            results = engine.search(query)
            print(f"Found {len(results)} document(s):\n")
            print(engine.format_results(results))
            print()
        except ValueError as e:
            print(f"Parse error: {e}\n")


if __name__ == "__main__":
    main()
