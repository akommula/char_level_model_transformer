import os
import numpy as np

from collections import Counter

# Map character ascii values to integers
class BuildTokenizer:
    def __init__(self, path):
        self.num_unique_chars = 1
        self.char2idx = {}
        
        print("Starting Train Ingestion...")
        if not os.path.exists(os.path.join(path, 'train_np.npy')):
            self.add_character('32') # Treat space character as a special token
            self.train = self.tokenize(path, 'train')
        else:
            self.train = np.load(os.path.join(path, 'train_np.npy'))
        print("Train dataset size:", len(self.train))
        
        print("Starting Valid Ingestion...")
        if not os.path.exists(os.path.join(path, 'valid_np.npy')):
            self.valid = self.tokenize(path, 'valid')
        else:
            self.valid = np.load(os.path.join(path, 'valid_np.npy'))
        print("Valid dataset size:", len(self.valid))
        
        print("Starting Test Ingestion...")
        if not os.path.exists(os.path.join(path, 'test_np.npy')):
            self.test = self.tokenize(path, 'test')
        else:
            self.test =  np.load(os.path.join(path, 'test_np.npy'))            
        print("Test dataset size:", len(self.test))
        
        # Prevent off by 1 indexing errors
        self.num_unique_chars = len(np.unique(np.concatenate([self.train, self.valid, self.test]))) + 1
        
    def tokenize(self, base, split):
        path = os.path.join(base, f"{split}.txt")
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
        
        np.save(os.path.join(base, f"{split}_np.npy"), ids) 
        return ids

    def add_character(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.num_unique_chars
            self.num_unique_chars += 1
        
