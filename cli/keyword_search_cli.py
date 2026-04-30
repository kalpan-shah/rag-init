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

from query_movie_data import search_movies # pylint: disable=wrong-import-position

def main() -> None:
    """
        Entry point of the CLI App
    """
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            if args.query:
                print(f"Searching for: {args.query}")
                for i, movie in enumerate(search_movies(args.query)):
                    print(f"{i+1}. {movie['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
