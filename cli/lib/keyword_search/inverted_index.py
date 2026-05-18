"""
@file:          cli/lib/keyword_search/inverted_index.py
@description:   Inverted index implementation for efficient keyword search
@date:          29 April 2026
@last edited:   11 May 2026
@author:        Kalpan Shah
@version:       1.0.0
"""
import os
import math
from typing import Dict, List, Set
from collections import Counter
import pickle
from cli.lib.keyword_search.data_preprocessing import get_tokens, preprocess_text
from cli.lib.data_handler import movie_data
from cli.lib.keyword_search.cache_paths import index_file_path, docmap_file_path, tf_file_path, \
    doclen_file_path


class InvertedIndex:
    """
        Inverted index data structure for efficient keyword search.
    """
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, object] = {}
        self.term_frequencies: Dict[str, Counter] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.tf_cache: Dict[tuple, int] = {}

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
        # e. update document length
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        """
            Calculate the average document length across all documents in the index.
            Returns:
                float: The average document length
        """
        total_docs = len(self.docmap)
        if total_docs == 0:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        avg_length = total_length / total_docs
        return avg_length

    def get_documents(self, term: str) -> List[int]:
        """
            Retrieve document IDs associated with a given term - token.
            Args:
                term (str): The search term
            Returns:
                List[int]: A list of document IDs containing the term
        """
        _doc_ids = self.index.get(preprocess_text(term), set())
        # print(f"Retrived {len(_doc_ids)} document IDs for term: {term}")
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
        if (doc_id, term) in self.tf_cache:
            return self.tf_cache[(doc_id, term)]
        _tokens = get_tokens(preprocess_text(term))
        if len(_tokens) > 1:
            raise ValueError("Term should be a single token.")
        _tf = self.term_frequencies.get(doc_id, Counter()).get(_tokens[0], 0)
        self.tf_cache[(doc_id, term)] = _tf
        return _tf

    def get_bm25_tf(self, doc_id: int, term: str, K1: float, B: float) -> float:
        """
            Retrieve the BM25 term frequency for a term in a specific document.
            Args:
                doc_id (int): The document ID
                term (str): The search term
                K1 (float): The BM25 parameter
                B (float): The BM25 Length normalization strength parameter

            Returns:
                float: The BM25 improved term frequency of the term in the document
        """
        def calculate_bm25_tf(tf: int, k1: float, b: float, doc_length: int) -> float:
            length_normalizer = 1 - b + b * (doc_length / self.avg_doc_length)
            return (tf * (k1 + 1)) / (tf + k1 * length_normalizer)
        basic_tf = self.get_tf(doc_id, term)

        # BM25 term frequency calculation
        return calculate_bm25_tf(basic_tf, K1, B, self.doc_lengths[doc_id])

    def get_idf(self, term: str) -> float:
        """
            Retrieve the inverse document frequency for a given term.
            Args:
                term (str): The search term
            Returns:
                float: The inverse document frequency for the term
        """
        def calculate_idf(total_docs: int, doc_freq: int) -> float:
            # print(f"Calculating IDF for term '{term}': "
            #       f"total_docs={total_docs}, doc_freq={doc_freq}")
            _idf = math.log((total_docs + 1) / (doc_freq + 1))
            # return round(_idf, 2)
            return _idf

        total_docs = len(self.docmap)
        doc_freq = 0
        for doc_id in self.get_documents(term):
            if self.get_tf(doc_id, term) > 0:
                doc_freq += 1
        return calculate_idf(total_docs, doc_freq)

    def get_bm25_idf(self, term: str) -> float:
        """
            Retrieve the BM25 inverse document frequency for a given term.
            Args:
                term (str): The search term
            Returns:
                float: The BM25 inverse document frequency for the term
        """
        def calculate_bm25_idf(total_docs: int, doc_freq: int) -> float:
            # print(f"Calculating BM25 IDF for term '{term}': "
            #   f"total_docs={total_docs}, doc_freq={doc_freq}")
            _idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            return _idf

        total_docs = len(self.docmap)
        doc_freq = 0
        for doc_id in self.get_documents(term):
            if self.get_tf(doc_id, term) > 0:
                doc_freq += 1
        return calculate_bm25_idf(total_docs, doc_freq)

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

        with open(doclen_file_path, 'wb') as dl_f:
            pickle.dump(self.doc_lengths, dl_f)

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

        with open(doclen_file_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)

        # calc the average document length after loading
        self.avg_doc_length = self.__get_avg_doc_length()
