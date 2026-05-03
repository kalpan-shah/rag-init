"""
@file:          cli/inverted_index.py
@description:   Inverted index implementation for efficient keyword search
@date:          29 April 2026 
@last edited:   01 April 2026
@author:        Kalpan Shah
@version:       1.0.0 
"""
import os
import math
from typing import Dict, List, Set
from collections import Counter
import pickle
from data_preprocessing import get_tokens, preprocess_text
from data_handler import movie_data, index_file_path, docmap_file_path, tf_file_path

class InvertedIndex:
    """
        Inverted index data structure for efficient keyword search.
    """
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, object] = {} 
        self.term_frequencies: Dict[str, Counter] = {}  
    
    def __add_document(self, doc_id: int, text: str) -> None:
        """
            Add a document to the inverted index.
            Args:
                doc_id (int): Unique identifier for the document
                text (str): Text content of the document
        """ 
        # preprocess text
        trimmed_text = preprocess_text(text)
        
        # a. get tokens
        tokens = get_tokens(trimmed_text)
        # b. update index
        for token in tokens:
            # create a new set for the token if it doesn't exist
            if token not in self.index:
                self.index[token] = set()
            # update the set of document IDs for the token
            self.index[token].add(doc_id) 
        # c. update docmap
        self.docmap[doc_id] = trimmed_text # text
        # d. update term frequencies
        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term: str) -> List[int]:
        """
            Retrieve document IDs associated with a given term - token.
            Args:
                term (str): The search term
            Returns:
                List[int]: A list of document IDs containing the term
        """
        _doc_ids = self.index.get(preprocess_text(term), set())
        print(f"Retrived {len(_doc_ids)} document IDs for term: {term}")
        return sorted(_doc_ids) if _doc_ids else []
    
    def get_tf(self, doc_id: int, term: str) -> int:
        """
            Retrieve the term frequency of a term in a specific document.
            Args:
                doc_id (int): The document ID
                term (str): The search term
            Returns:
                int: The term frequency of the term in the document
        """
        _tokens = get_tokens(preprocess_text(term))
        if len(_tokens) > 1:
            raise ValueError("Term should be a single token.")
        return self.term_frequencies.get(doc_id, Counter()).get(_tokens[0], 0)

    def get_idf(self, term: str) -> float:
        def calculate_idf(total_docs: int, doc_freq: int) -> float:
            print(f"Calculating IDF for term '{term}': total_docs={total_docs}, doc_freq={doc_freq}")
            _idf = math.log((total_docs + 1) / (doc_freq + 1))
            # return round(_idf, 2)
            return _idf
        
        total_docs = len(self.docmap)
        doc_freq = 0
        for doc_id in self.get_documents(term): 
            if self.get_tf(doc_id, term) > 0:
                doc_freq += 1
        return calculate_idf(total_docs, doc_freq)    

    def build(self) -> None:
        """
            Build the inverted index from the movies data.
        """
        for movie in movie_data:
            _in_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], _in_text)
        
        # Sort the document IDs for each token in the index for consistent retrieval
        for token in self.index:
            self.index[token] = set(sorted(self.index[token]))

        # save the index and docmap to disk
        self.save()
        
    def save(self) -> None:
        """
            Save the inverted index and document mapping to disk using pickle.
        """
        with open(index_file_path, 'wb') as i_f:
            pickle.dump(self.index, i_f)
        
        with open(docmap_file_path, 'wb') as d_f:
            pickle.dump(self.docmap, d_f)

        with open(tf_file_path, 'wb') as tf_f:
            pickle.dump(self.term_frequencies, tf_f)

    def load(self) -> None:
        """
            Load the inverted index and document mapping from disk using pickle.
        """

        if not os.path.exists(index_file_path) or not os.path.exists(docmap_file_path):
            raise FileNotFoundError("Inverted index files not found. Please build the index first.")

        with open(index_file_path, 'rb') as f:
            self.index = pickle.load(f)
        
        with open(docmap_file_path, 'rb') as f:
            self.docmap = pickle.load(f)

        with open(tf_file_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
