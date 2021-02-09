
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


class TextPreprocessor:

    def __init__(self):
        self.data = None
        self.vocab = None
        self.word2idx = None

    def clean_text_(self, texts):
        new = [t.lower() for t in texts]
        new = [t.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for t in new]
        new = [re.sub(r'\d+ml', 'ml', t) for t in new]
        new = [re.sub(r'\d+ ml', 'ml', t) for t in new]
        new = [t for t in new if len(t.split()) > 0]
        new = [' '.join(t.split()) for t in new]
        return new

    def get_vocab_(self, texts):
        words = []
        for t in texts:
            words.extend(t.split())
        words = list(set(words))
        return words

    def run(self, texts):
        clean_texts = self.clean_text_(texts)
        words = self.get_vocab_(clean_texts)
        self.data = np.asarray(clean_texts)
        self.main_words = words
        self.vocab = ['[CLS]', '[PAD]'] + words + ['[MASK]', '[UNK]']
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}


class MaskedLMDataLoader:

    def __init__(self, data, batch_size, shuffle=True):
        self.preprocessor = TextPreprocessor()
        self.preprocessor.run(data)
        self.batch_size = batch_size
        self.data = self.preprocessor.data
        self.main_words = self.preprocessor.main_words
        self.vocab = self.preprocessor.vocab
        self.word2idx = self.preprocessor.word2idx
        self.ptr = 0
        if shuffle:
            seq = np.random.permutation(np.arange(len(self.data)))
            self.data = self.data[seq]

    def __len__(self):
        return len(self.data) // self.batch_size
            
    def pad_mask_tokenize_(self, lines):
        maxlen = max([len(t.split()) for t in lines])
        tokens, targets, mask_idx = [], [], []
        for i in range(len(lines)):
            words = lines[i].split()
            idx_to_replace = np.random.choice(np.arange(len(words)), size=math.ceil(0.15*len(words)), replace=False)
            trg = []
            for j in idx_to_replace:
                trg.append(self.word2idx.get(words[j], '[UNK]'))
                if random.random() < 0.8:
                    words[j] = '[MASK]'
                else:
                    words[j] = random.choice(self.main_words)
            
            words = ['[CLS]'] + words + ['[PAD]'] * (maxlen - len(words))
            targets.append(torch.LongTensor(trg))
            mask_idx.append(torch.LongTensor([c+1 for c in idx_to_replace]))
            tokens.append([self.word2idx.get(w, '[UNK]') for w in words])
        return tokens, targets, mask_idx

    def flow(self):
        lines = self.data[self.ptr: self.ptr+self.batch_size]
        tokens, targets, mask_idx = self.pad_mask_tokenize_(lines)

        self.ptr += self.batch_size
        if self.ptr >= len(self.data):
            self.ptr = 0

        return torch.LongTensor(tokens), targets, mask_idx   


class ClassificationDataLoader:

    def __init__(self, data, labels, batch_size, shuffle=True, device=None):
        self.preprocessor = TextPreprocessor()
        self.preprocessor.run(data)
        self.batch_size = batch_size
        self.labels = labels
        self.data = self.preprocessor.data
        self.main_words = self.preprocessor.main_words
        self.vocab = self.preprocessor.vocab
        self.word2idx = self.preprocessor.word2idx
        self.ptr = 0  
        if shuffle:
            seq = np.random.permutation(np.arange(len(self.data)))
            self.data = self.data[seq]
            self.labels = self.labels[seq]

    def __len__(self):
        return len(self.data) // self.batch_size
            
    def pad_and_tokenize_(self, lines):
        maxlen = max([len(t.split()) for t in lines])
        tokens = []
        for i in range(len(lines)):
            words = lines[i].split()
            words = ['[CLS]'] + words + ['[PAD]'] * (maxlen - len(words))
            tokens.append([self.word2idx.get(w, '[UNK]') for w in words])
        return tokens

    def flow(self):
        lines = self.data[self.ptr: self.ptr+self.batch_size]
        targets = self.labels[self.ptr: self.ptr+self.batch_size]
        tokens = self.pad_and_tokenize_(lines)

        self.ptr += self.batch_size
        if self.ptr >= len(self.data):
            self.ptr = 0

        return torch.LongTensor(tokens), torch.LongTensor(targets)


def get_dataloaders(task, root, val_size, batch_size):
    if task == 'mlm':
        if os.path.exists(os.path.join(root, 'data.txt')):
            data = load_data(root)
            data = np.asarray(data)
        else:
            raise NotImplementedError(f'File data.txt not found at {root}. Please rename the file containing data to data.txt!')

    elif task == 'aec':
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

    if task == 'mlm':
        train_data, val_data = data[train_idx], data[val_idx]
        train_loader = MaskedLMDataLoader(train_data, batch_size, shuffle=True)
        val_loader = MaskedLMDataLoader(val_data, batch_size, shuffle=False)
    
    elif task == 'aec':
        train_X, train_y = data[train_idx], labels[train_idx]
        val_X, val_y = data[val_idx], labels[val_idx]
        train_loader = ClassificationDataLoader(train_X, train_y, batch_size, shuffle=True)
        val_loader = ClassificationDataLoader(val_X, val_y, batch_size, shuffle=False)

    return train_loader, val_loader 


def load_data(root):
    with open(os.path.join(root, 'data.txt'), 'r') as f:
        lines = f.read().split('\n')
    return lines


# Testing script
if __name__ == '__main__':

    sents = [
        'My name is nishant',
        'I have decided to go for a walk',
        'This is probably not going to work',
        'Thats very unfortunate honestly',
        'Who was at the door?',
        'Yeah this seems like a good idea',
        'I dont understand what the problem really is',
        "They're worried a litte too much about privacy",
        'Umm what?',
        "Phew! That was a tough one"
    ]

    mlm = MaskedLMDataLoader(sents, batch_size=2, shuffle=True)
    print("Num batches:", len(mlm))
    for _ in range(20):
        tokens = mlm.flow()
        print(tokens[0])
        print(tokens[1])
        print()
