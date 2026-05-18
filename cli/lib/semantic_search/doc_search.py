import numpy as np

from cli.lib.semantic_search.semantic_search import SemanticSearch
from cli.lib.data_handler import movie_data



def verify_model():
    _searcher = SemanticSearch()
    print(f"Model loaded: {_searcher.model}")
    print(f"Max sequence length: {_searcher.max_sequence_length}")

def verify_embeddings():
    _searcher = SemanticSearch()
    _searcher.load_or_create_embeddings(movie_data)
    print(f"Number of docs:   {len(_searcher.documents)}")
    print(f"Embeddings shape: {_searcher.embeddings.shape[0]} vectors in {_searcher.embeddings.shape[1]} dimensions")

def embed_text(text: str) -> None:
    _searcher = SemanticSearch()
    print(f"Text: {text}")
    embedding = _searcher.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    print(type(embedding))

def embed_query_text(query: str) -> None:
    _searcher = SemanticSearch()
    print(f"Query: {query}")

    embedding = _searcher.generate_embedding(query)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def search(query: str, limit: int) -> None:
    _searcher = SemanticSearch()
    _searcher.load_or_create_embeddings(movie_data)
    _results = _searcher.search(query, limit)
    for idx, _res in enumerate(_results):
        print("{}. {} (Score: {}) \n{}\n".format(idx, _res['title'], _res['score'], _res['description']))
