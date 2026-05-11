"""
@file:          cli/lib/keyword_search/cache_paths.py
@description:   Cache paths for the inverted index files
@date:          29 April 2026
@last edited:   11 May 2026
@author:        Kalpan Shah
@version:       1.0.0
"""

import os

# disk path for the inverted index file
cache_dir = os.path.join(os.getcwd(), 'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

index_file_path = os.path.join(cache_dir, 'index.pkl')
docmap_file_path = os.path.join(cache_dir, 'docmap.pkl')
tf_file_path = os.path.join(cache_dir, 'term_frequencies.pkl')
doclen_file_path = os.path.join(cache_dir, 'doc_lengths.pkl')
