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

sys.path.append(os.getcwd())  # Add cli to path for imports

from cli.lib.semantic_search import verify_model, embed_text, verify_embeddings, \
    embed_query_text

def main() -> None:
    """
        Define the command line interface for the semantic search CLI
    """
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands: ")

    subparsers.add_parser("verify", help="Verify that the semantic search model is loaded correctly")
    
    embedding_gen_parser = subparsers.add_parser("embed_text", help="Generate an embedding for the input text")
    embedding_gen_parser.add_argument("text", help="Input text to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify that the embeddings are generated and loaded correctly")

    query_parser = subparsers.add_parser("embed_query", help="Convers user search query in embedding vectors")
    query_parser.add_argument("query", help="Input query to generate embeddings for")

    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()
        
        case "embed_query":
            embed_query_text(args.query)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()