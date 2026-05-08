"""
@file:          cli/semantic_search_cli.py
@description:   Semantic search CLI for finding movies
@date:          08 May 2026
@last edited:   08 May 2026
@author:        Kalpan Shah
@version:       1.0.0
"""
import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(), 'cli'))  # Add cli to path for imports

def main() -> None:
    """
        Define the command line interface for the semantic search CLI
    """
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands: ")

    subparsers.add_parser("verify", help="Verify that the semantic search model is loaded correctly")
    

    args = parser.parse_args()


    match args.command:
        case "verify":
            from lib.semantic_search import verify_model
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()