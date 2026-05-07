"""
@file:          cli/keyword_search_cli.py
@description:   Keyword search CLI for finding movies
@date:          29 April 2026
@last edited:   30 April 2026
@author:        Kalpan Shah
@version:       1.0.0
"""
import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(), 'cli'))  # Add cli to path for imports

# pylint: disable=wrong-import-position
from query_movie_data import search_movies, \
    build_index, get_term_frequency, get_inverse_doc_freq, \
    get_tfidf_score, get_bm25_inverse_doc_freq, get_bm25_term_frequency, \
    bm25_search_movies, BM25_K1, BM25_B
# pylint: enable=wrong-import-position

def command_line_interface() -> None:
    """
        Define the command line interface for the keyword search CLI
    """
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    subparsers.add_parser("build", help="Build the inverted index")

    # Keyword search for movie names
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="search query")

    # BM25 Keyword Search for movie names
    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25_search_parser.add_argument("query", type=str, help="search query")
    bm25_search_parser.add_argument("limit", type=int, nargs='?', default=5, help="maximum number of results to return")

    # Get the term frequency for a given document and term
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="doc ID to retrieve term frequency for")
    tf_parser.add_argument("term", type=str, help="term to retrieve frequency for")

    # Get the inverse doc frequency for a given term
    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency for a term")
    idf_parser.add_argument("term", type=str,
                            help="term to retrieve inverse document frequency for")

    # Get the TF-IDF score for a given document and term
    tfidf_parser = subparsers.add_parser("tfidf",
                                         help="Get the TF-IDF score for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="doc ID to retrieve TF-IDF score for")
    tfidf_parser.add_argument("term", type=str, help="term to retrieve TF-IDF score for")

    # Get the BM25 IDF score for a given term
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")
    return parser

def execute_command(arg_parser) -> None:
    """
        Execute the command based on the parsed arguments
    """
    args = arg_parser.parse_args()
    match args.command:
        case "build":
            build_index()

        case "search":
            if not args.query:
                raise ValueError("Search query is required for 'search' command")
            print(f"Searching for: {args.query}")
            for i, movie in enumerate(search_movies(args.query)):
                print(f"{i+1}. {movie['title']}")

        case "bm25search":
            if not args.query:
                raise ValueError("Search query is required for 'bm25search' command")
            print(f"Searching for: {args.query}")
            for i, (doc_id, score, title) in enumerate(bm25_search_movies(args.query, args.limit)):
                print(f"{i+1}. ({doc_id}) {title} - Score: {score:.2f}")

        case "tf":
            if not args.doc_id or not args.term:
                raise ValueError("Both doc_id and term are required for 'tf' command")
            _tf: int = get_term_frequency(args.doc_id, args.term)
            print(_tf)

        case "idf":
            if not args.term:
                raise ValueError("Term is required for 'idf' command")
            _idf: float = get_inverse_doc_freq(args.term)
            print(f"Inverse document frequency of '{args.term}': {_idf:.2f}")

        case "tfidf":
            if not args.doc_id or not args.term:
                raise ValueError("Both doc_id and term are required for 'tfidf' command")
            _tf_idf: float = get_tfidf_score(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {_tf_idf:.2f}")

        case "bm25idf":
            if not args.term:
                raise ValueError("Term is required for 'bm25idf' command")

            _bm25_idf: float = get_bm25_inverse_doc_freq(args.term)
            print(f"BM25 IDF score of '{args.term}': {_bm25_idf:.2f}")

        case "bm25tf":
            if not args.doc_id or not args.term:
                raise ValueError("Both doc_id and term are required for 'bm25tf' command")

            _bm25_tf: float = get_bm25_term_frequency(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {_bm25_tf:.2f}")

        case _:
            arg_parser.print_help()


def main() -> None:
    """
        Entry point of the CLI App
    """
    _parser = command_line_interface()
    execute_command(_parser)


if __name__ == "__main__":
    main()
