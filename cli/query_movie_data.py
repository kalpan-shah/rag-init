""" 
@file:          cli/keyword_search_cli.py
@description:   Keyword search CLI for finding movies
@date:          30 April 2026
@last edited:   30 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
from typing import List
import os
import json
 
# Assuming running from the project root, adjust path accordingly later if needed
movie_data_json = os.path.join(os.getcwd(),'data', 'movies.json')

def load_data(file_path: str) -> dict:
    """
        Helper function to just load the json and return the data dict
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

movie_data: List[dict] = load_data(movie_data_json)['movies']

def search_movies(query: str) -> List[dict]:
    """
        V1 - Search for movies if the movie name contains the query string 
    """
    results = []
    for movie in movie_data:
        if not isinstance(movie, dict) and 'title' not in movie:
            continue
        
        if query.lower() in movie['title'].lower():
            results.append(movie)
    return results
