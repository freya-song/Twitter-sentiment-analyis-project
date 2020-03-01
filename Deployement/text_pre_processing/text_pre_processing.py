#!/usr/bin/env python
"""This script performs preprocessing for a sentiment
analysis task with a CNN + Embedding model"""

import os
import re
import string
import zipfile
from nltk.tokenize import TweetTokenizer

class PreProcessor:
    """This is the main class for the PreProcessor"""
    def __init__(self, max_length_tweet, max_length_dictionary = 400003):
        """Constructor of the class. Init with the imput args"""
        if max_length_tweet < 0:
            raise ValueError("max_length_tweet can not be negative!")
        if max_length_dictionary < 0:
            raise ValueError("max_length_dictionary can not be negative!")
        self.max_length_tweet = max_length_tweet
        self.max_length_dictionary = max_length_dictionary
        self.emb_dict = self.load_dict()
    def load_dict(self):
        """This is a helper function to load the dictionary
        in the most memory efficient way"""
        dic = {}
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(curr_dir, "glove/word_list.txt")
        if ".zip/" in file_path:
            archive_path = os.path.abspath(file_path)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = zipfile.ZipFile(archive_path, "r")
            embeddings = archive.read(path_inside).decode("utf8").split("\n")
            for embedding in embeddings:
                try:
                    word, idx = embedding.strip().split()
                except:
                    break
                if int(idx) > self.max_length_dictionary:
                    break
                dic[word] = int(idx)
        else:
            with open(file_path, 'r', encoding = 'utf-8') as embeddings:
                for embedding in embeddings:
                    word, idx = embedding.strip().split()
                    if int(idx) > self.max_length_dictionary:
                        break
                    dic[word] = int(idx)
        return dic
    def clean_text(self, text):
        """Clean the raw text to remove URLs and remove
        any other non-English chars, inplace"""
        #Cleaning the raw text to remove URLs
        text = re.sub(
            r"http\S+",
            "",
            text
        ).strip()
        #Removing any other non-English chars
        printable = set(string.printable)
        text = ''.join(
            filter(
                lambda x: x in printable,
                text
            )
        )
        return text
    def tokenize_text(self, text):
        """Convert a string into an array of tokens."""
        tokens = TweetTokenizer().tokenize(text)[:self.max_length_tweet]
        return tokens
    def replace_token_with_index(self, tokens):
        """replace each token in a list of tokens by their corresponding
        index in GloVe dictionary and producing a list of indexes
        -1 if word doesn't exists in the dictionary"""
        idxs = list(map(lambda x: self.emb_dict[x.lower()] if x.lower() in self.emb_dict else 1, tokens))
        return idxs
    def pad_sequence(self, idxs):
        """Padding a list of indices with 0 until a maximum length"""
        if len(idxs) < self.max_length_tweet:
            pad = idxs + [0] * (self.max_length_tweet - len(idxs))
            return pad
        return idxs[:self.max_length_tweet]
    def pre_process(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        idxs = self.replace_token_with_index(tokens)
        pad = self.pad_sequence(idxs)
        return pad
