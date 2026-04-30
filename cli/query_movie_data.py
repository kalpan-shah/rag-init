""" 
@file:          cli/query_movie_data.py
@description:   Helper functions for querying movie data
@date:          30 April 2026
@last edited:   30 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
from typing import List 
from data_preprocessing import preprocess_text, get_tokens
from data_handler import movie_data

def search_movies(query: str) -> List[dict]:
    """
        V1 - Search for movies if the movie name contains the query string 
            Text Processing:
                a. Case insensitive search
                b. punctuation removal
                c. tokenization
    """
    results = []
    _query = preprocess_text(query)
    _query_tokens = get_tokens(_query)

    for movie in movie_data:
        # Check if Valid movie dict
        if not isinstance(movie, dict) or 'title' not in movie:
            continue

        # text preprocessing
        _movie_title = preprocess_text(movie['title'])
        _movie_title_tokens = get_tokens(_movie_title)
        _n_movie_title = " ".join(_movie_title_tokens)

        _token_match = any(token in _n_movie_title for token  in _query_tokens)
        if _token_match:
            results.append(movie)

    return results
