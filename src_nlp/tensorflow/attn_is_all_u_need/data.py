from config import args
from collections import Counter

import numpy as np


class DataLoader:
    def __init__(self, source_path, target_path):
        self.source_words = self.read_data(source_path)
        self.target_words = self.read_data(target_path)

        self.source_word2idx = self.build_index(self.source_words)
        self.target_word2idx = self.build_index(self.target_words, is_target=True)

    
    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


    def build_index(self, data, is_target=False):
        chars = [char for line in data.split('\n') for char in line]
        chars = [char for char, freq in Counter(chars).items() if freq > args.min_freq]
        if is_target:
            symbols = ['<pad>','<start>','<end>','<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}
        else:
            symbols = ['<pad>','<unk>'] if not args.tied_embedding else ['<pad>','<start>','<end>','<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}


    def pad(self, data, word2idx, max_len, is_target=False):
        res = []
        for line in data.split('\n'):
            temp_line = [word2idx.get(char, word2idx['<unk>']) for char in line]
            if len(temp_line) >= max_len:
                if is_target:
                    temp_line = temp_line[:(max_len-1)] + [word2idx['<end>']]
                else:
                    temp_line = temp_line[:max_len]
            if len(temp_line) < max_len:
                if is_target:
                    temp_line += ([word2idx['<end>']] + [word2idx['<pad>']]*(max_len-len(temp_line)-1)) 
                else:
                    temp_line += [word2idx['<pad>']] * (max_len - len(temp_line))
            res.append(temp_line)
        return np.array(res)


    def load(self):
        source_idx = self.pad(self.source_words, self.source_word2idx, args.source_max_len)
        target_idx = self.pad(self.target_words, self.target_word2idx, args.target_max_len, is_target=True)
        return source_idx, target_idx
