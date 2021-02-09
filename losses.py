
"""
Loss functions and other criterion.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossentropyLoss(nn.Module):
    ''' Masked CE loss for MLM models '''

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = nn.NLLLoss()

    def forward(self, output, target, mask):
        '''
        output has size (bs, seq_length, n_classes)
        '''
        total_loss = 0
        total_correct = 0
        count = 0
        for i in range(output.size(0)):
            m = mask[i].to(self.device)
            trg = target[i].to(self.device)
            out = output[i, m, :]
            total_loss += self.loss_fn(out, trg)
            pred = out.argmax(dim=-1)
            total_correct += pred.eq(trg.view_as(pred)).sum().item()
            count += trg.numel()

        avg_loss = total_loss/output.size(0)
        acc = total_correct/count
        return avg_loss, acc


class ClassificationLoss(nn.Module):
    ''' Masked CE loss for MLM models '''

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = nn.NLLLoss()

    def forward(self, output, target):
        '''
        output has size (bs, seq_length, n_classes)
        Only the vector corresponding to [CLS] is used
        '''
        out = output[:, 0, :]
        trg = target.to(self.device)
        loss = self.loss_fn(out, trg)
        pred = out.argmax(dim=-1)
        acc = pred.eq(trg.view_as(pred)).sum().item()/out.size(0)
        return loss, acc