""" 
@file:          cli/query_movie_data.py
@description:   Helper functions for querying movie data
@date:          30 April 2026
@last edited:   01 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
from typing import List
from data_preprocessing import preprocess_text, get_tokens
from data_handler import movie_data 
from inverted_index import InvertedIndex

index = InvertedIndex()

def build_index() -> None:
    """
        Build the inverted index from the movie data
    """
    index.build() 


def search_movies(query: str) -> List[dict]:
    """
        V1 - Search for movies if the movie name contains the query string
        V2 - Search for movies using the inverted index DS

        Args:
            query (str): The search query string

        Returns:
            List[dict]: A list of movie dictionaries that match the search query
    """
    try:
        index.load()
        _query = preprocess_text(query)
        _query_tokens = get_tokens(_query)

        doc_ids = set()

        for token in _query_tokens:
            doc_ids.update(index.get_documents(token))
        
        doc_ids = sorted(doc_ids)[:5] # limit to top 5 results
        
        results = [movie for movie in movie_data if movie['id'] in doc_ids]

        return results
        
    except Exception as e:
        print(f"Error loading index: {e}")
        return []
