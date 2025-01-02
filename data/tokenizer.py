import os
import numpy as np

from collections import Counter

# Map character ascii values to integers
class BuildTokenizer:
    def __init__(self, path, tokenize_from_scratch = False):
        self.num_unique_chars = 1
        self.char2idx = {}
        self.idx2char = {}
        self.tokenize_from_scratch = tokenize_from_scratch
        
        print("Starting Train Ingestion...")
        if tokenize_from_scratch or not os.path.exists(os.path.join(path, 'train_np.npy')):
            self.add_character('32') # Treat space character as a special token
            self.train = self.init_tokenize(path, 'train')
        else:
            self.train = np.load(os.path.join(path, 'train_np.npy'))
        print("Train dataset size:", len(self.train))
        
        print("Starting Valid Ingestion...")
        if tokenize_from_scratch or not os.path.exists(os.path.join(path, 'valid_np.npy')):
            self.valid = self.init_tokenize(path, 'valid')
        else:
            self.valid = np.load(os.path.join(path, 'valid_np.npy'))
        print("Valid dataset size:", len(self.valid))
        
        print("Starting Test Ingestion...")
        if tokenize_from_scratch or not os.path.exists(os.path.join(path, 'test_np.npy')):
            self.test = self.init_tokenize(path, 'test')
        else:
            self.test =  np.load(os.path.join(path, 'test_np.npy'))            
        print("Test dataset size:", len(self.test))
        
        # Prevent off by 1 indexing errors
        self.num_unique_chars = len(np.unique(np.concatenate([self.train, self.valid, self.test]))) + 1
        
    def init_tokenize(self, base, split = None):
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
        
        if split is not None:
            np.save(os.path.join(base, f"{split}_np.npy"), ids) 
        
        return ids

    def encode(self, text): 
        num_tokens = 0
        for line in text.split('\n'):
            num_tokens += (len(line) + 1)

        ids = np.zeros(num_tokens, dtype=np.int64)
        idx = 0
        
        for line in text.split('\n'):
            chars = [str(ord(c)) for c in line] + ['<eos>']
                        
            ids[idx:idx + len(chars)] = [self.char2idx[char] for char in chars]
            idx += len(chars)

        return (ids)

    def decode(self, tokens):
        result = ""
        for t in tokens:
            char = self.idx2char[t]
            result += ('\n' if char == '<eos>' else chr(int(char)))

        return result

    def add_character(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.num_unique_chars
            self.idx2char[self.num_unique_chars] = char
            self.num_unique_chars += 1
        
