
"""
Loss functions and other criterion.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    ''' Masked CE loss for MLM models '''

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        '''
        output has size (bs, seq_length, n_classes)
        Only the vector corresponding to [CLS] is used
        '''
        out = output
        trg = target.to(self.device)
        loss = self.loss_fn(out, trg)
        pred = out.argmax(dim=-1)
        acc = pred.eq(trg.view_as(pred)).sum().item()/out.size(0)
        return loss, acc