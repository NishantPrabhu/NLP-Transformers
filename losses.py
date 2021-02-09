
"""
Loss functions and other criterion.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossentropyLoss(nn.Module):
    ''' Masked CE loss for MLM models '''

    def __init__(self, config):
        super().__init__()
        self.loss_fn = nn.NLLLoss()

    def forward(self, output, target, mask):
        '''
        output has size (bs, seq_length, n_classes)
        '''
        out = output[:, mask, :]
        loss = self.loss_fn(out, target)
        return loss


class ClassificationLoss(nn.Module):
    ''' Masked CE loss for MLM models '''

    def __init__(self, config):
        super().__init__()
        self.loss_fn = nn.NLLLoss()

    def forward(self, output, target):
        '''
        output has size (bs, seq_length, n_classes)
        Only the vector corresponding to [CLS] is used
        '''
        out = output[:, 0, :]
        loss = self.loss_fn(out, target)
        return loss