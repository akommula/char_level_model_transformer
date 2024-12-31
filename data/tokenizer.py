import os
import numpy as np

from collections import Counter

# Map character ascii values to integers
class BuildTokenizer:
    def __init__(self, path):
        self.num_unique_chars = 2
        self.char2idx = {}
        
        print("Starting Train Ingestion...")
        self.train = self.tokenize(os.path.join(path, 'train.txt')) 
        #self.train = np.ones(1024, dtype=np.int64)
        print("Train dataset size:", len(self.train))
        
        print("Starting Valid Ingestion...")
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        #self.valid = np.ones(1024, dtype=np.int64)
        print("Valid dataset size:", len(self.train))
        
        print("Starting Test Ingestion...")
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        #self.test = np.ones(1024, dtype=np.int64)
        print("Test dataset size:", len(self.train))
        
    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                chars = line.split() + ['<eos>']
                for char in chars:
                    self.add_character(char)
                
                tokens += len(chars)
                
        with open(path, 'r') as f:
            ids = np.zeros(tokens, dtype=np.int64)
            idx = 0
            
            for line in f:
                chars = line.split() + ['<eos>']

                ids[idx:idx + len(chars)] = [self.char2idx[char] for char in chars]
                idx += len(chars)
                
        return ids

    def add_character(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.num_unique_chars
            self.num_unique_chars += 1
        