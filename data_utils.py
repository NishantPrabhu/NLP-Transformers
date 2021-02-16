
""" 
Data handlers.
"""
import re
import os
import math
import torch 
import string
import random
import numpy as np
from transformers import AutoTokenizer


class DataLoader:

    def __init__(self, data, labels, batch_size, shuffle=True, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.word2idx = self.tokenizer.vocab
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.pad_token = self.word2idx['[PAD]']
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.ptr = 0  
        if shuffle:
            seq = np.random.permutation(np.arange(len(self.data)))
            self.data = self.data[seq]
            self.labels = self.labels[seq]

    def __len__(self):
        return len(self.data) // self.batch_size
            
    def pad_and_tokenize_(self, lines):
        maxlen = max([len(t.split()) for t in lines])
        unpadded_tokens, all_tokens = [], []

        for i in range(len(lines)):
            words = lines[i].split()
            tokens = self.tokenizer(' '.join(words))['input_ids']
            unpadded_tokens.append(tokens)

        maxlen = max([len(t) for t in unpadded_tokens])
        for i in range(len(unpadded_tokens)):
            all_tokens.append(unpadded_tokens[i] + [self.pad_token]*(maxlen - len(unpadded_tokens[i])))

        return all_tokens

    def flow(self):
        lines = self.data[self.ptr: self.ptr+self.batch_size]
        targets = self.labels[self.ptr: self.ptr+self.batch_size]
        tokens = self.pad_and_tokenize_(lines)

        self.ptr += self.batch_size
        if self.ptr >= len(self.data):
            self.ptr = 0

        return torch.LongTensor(tokens), torch.LongTensor(targets)


def get_dataloaders(root, val_size, batch_size):
    if os.path.exists(os.path.join(root, 'data.txt')):
        lines = load_data(root)
        data, labels = [], []
        for l in lines:
            if len(l) > 0:
                data.append(l.split('\t')[0])
                labels.append(int(l.split('\t')[1]))
        data, labels = np.asarray(data), np.asarray(labels)
    else:
        raise NotImplementedError(f'File data.txt not found at {root}. Please rename the file containing data to data.txt!')

    val_idx = np.random.choice(np.arange(len(data)), size=int(val_size * len(data)), replace=False)
    train_idx = np.array([i for i in np.arange(len(data)) if i not in val_idx])
    
    train_X, train_y = data[train_idx], labels[train_idx]
    val_X, val_y = data[val_idx], labels[val_idx]
    train_loader = DataLoader(train_X, train_y, batch_size, shuffle=True)
    val_loader = DataLoader(val_X, val_y, batch_size, shuffle=False)
    return train_loader, val_loader 


def load_data(root, name='data'):
    with open(os.path.join(root, f'{name}.txt'), 'r') as f:
        lines = f.read().split('\n')
    return lines