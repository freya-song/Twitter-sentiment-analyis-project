#!/usr/bin/env python
"""
This script performs preprocessing for a sentiment
analysis task with a CNN + Embedding model

"""

import os
import re
import string
import zipfile
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

class PreProcessor:
    """
    This is the main class for the PreProcessor
    
    """
    # Turns out 50 might be the best length
    def __init__(self, max_length_tweet = 256, max_length_dictionary = 400003):
        """
        Constructor of the class. Init with the imput args
        
        """
        if max_length_tweet < 0:
            raise ValueError("max_length_tweet can not be negative!")
        if max_length_dictionary < 0:
            raise ValueError("max_length_dictionary can not be negative!")
        self.MAX_LENGTH_TWEET = max_length_tweet
        self.MAX_LENGTH_DICTIONARY = max_length_dictionary
        self.COMBINED_PAT = r'|'.join((r'@[A-Za-z0-9]+', r'https?://[A-Za-z0-9./]+'))
        self.EMB_DICT = self.load_embedding_dict()
        self.STOPWORDS = self.load_stopword()
    def load_embedding_dict(self):
        """
        This is a helper function to load the dictionary
        in the most memory efficient way
        
        """
        dic = {}
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(curr_dir, "Lib/word_list")
        if ".zip/" in file_path:
            archive_path = os.path.abspath(file_path)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = zipfile.ZipFile(archive_path, "r")
            embeddings = archive.read(path_inside).decode("utf8").split("\n")
            idx = 0
            for embedding in embeddings:
                word = embedding.strip()
                if idx >= self.MAX_LENGTH_DICTIONARY:
                    break
                dic[word] = idx
                idx += 1
        else:
            with open(file_path, 'r', encoding = 'utf-8') as embeddings:
                idx = 0
                for embedding in embeddings:
                    word = embedding.strip()
                    if idx >= self.MAX_LENGTH_DICTIONARY:
                        break
                    dic[word] = idx
                    idx += 1
        return dic
    def load_stopword(self):
        """
        This is a helper function to load the stopword
        in the most memory efficient way
        
        """
        dic = {}
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(curr_dir, "Lib/twitter-stopwords")
        if ".zip/" in file_path:
            archive_path = os.path.abspath(file_path)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = zipfile.ZipFile(archive_path, "r")
            stopwrods = archive.read(path_inside).decode("utf8").strip().split(",")
        else:
            words = open(file_path, 'r', encoding = 'utf-8')
            stopwrods = words.readline().strip().split(",")
        return stopwrods
    def clean_text(self, text):
        """
        Clean the raw text to remove URLs and remove
        any other non-English chars, inplace

        """
        # HTML decoding and @ mention
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        stripped = re.sub(self.COMBINED_PAT, '', souped)
        # Byte order marks
        try:
            clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            clean = stripped
        # Keep letters only
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        # To lower case
        text = letters_only.lower()
        return text
    def tokenize_text(self, text):
        """
        Convert a string into an array of tokens.
        
        """
        # Tokenize and eliminate size under MAX_LENGTH_TWEET
        tokens = TweetTokenizer().tokenize(text)
        # Filter stopwords
        #tokens = [word for word in tokens if word not in self.STOPWORDS]
        # Eliminate size under MAX_LENGTH_TWEET
        tokens = tokens[:self.MAX_LENGTH_TWEET]
        return tokens
    def replace_token_with_index(self, tokens):
        """
        replace each token in a list of tokens by their corresponding
        index in GloVe dictionary and producing a list of indexes
        -1 if word doesn't exists in the dictionary
        
        """
        idxs = list(map(lambda x: self.EMB_DICT[x] if x in self.EMB_DICT else 1, tokens))
        return idxs
    def pad_sequence(self, idxs):
        """
        Padding a list of indices with 0 until a maximum length
        
        """
        if len(idxs) < self.MAX_LENGTH_TWEET:
            pad = idxs + [0] * (self.MAX_LENGTH_TWEET - len(idxs))
            return pad
        return idxs[:self.MAX_LENGTH_TWEET]
    def pre_process(self, text):
        """
        Run the pre-processing together

        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        idxs = self.replace_token_with_index(tokens)
        pad = self.pad_sequence(idxs)
        return pad
