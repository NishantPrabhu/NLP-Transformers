
""" 
Networks for training.
"""
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.heads = num_heads
        self.model_dim = model_dim
        if self.model_dim % self.heads != 0:
            raise ValueError(f'Working dimension ({model_dim}) not a multiple of num_heads ({num_heads})')
        
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False) 
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        bs, n, _ = x.size()                     
        q = self.query(x).view(bs, n, self.heads, self.model_dim // self.heads)                     # (bs, n, heads, d)
        k = self.key(x).view(bs, n, self.heads, self.model_dim // self.heads)                       # (bs, n, heads, d)
        v = self.value(x).view(bs, n, self.heads, self.model_dim // self.heads)                     # (bs, n, heads, d)

        q = q.permute(0, 2, 1, 3).contiguous()                                                      # (bs, heads, n, d)
        k = k.permute(0, 2, 1, 3).contiguous()                                                      # (bs, heads, n, d)
        v = v.permute(0, 2, 1, 3).contiguous()                                                      # (bs, heads, n, d)
        attn_scores = torch.einsum('bhid,bhjd->bhij', [q, k]) / math.sqrt(self.model_dim)           # (bs, heads, n, n)                           
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.einsum('bhij,bhjd->bhid', [attn_probs, v])                                  # (bs, heads, n, d)
        context = context.permute(0, 2, 1, 3).contiguous().view(bs, n, -1)                            # (bs, n, model_dim)

        return self.layer_norm(context + x)

    
class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.fc_1 = nn.Linear(model_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        out = self.fc_2(self.relu(self.fc_1(x)))
        return self.layer_norm(out + x)


class EncoderBlock(nn.Module):

    def __init__(self, model_dim, ff_dim, num_heads):
        super().__init__()
        self.attention = SelfAttention(num_heads, model_dim)
        self.feedfwd = Feedforward(model_dim, ff_dim)

    def forward(self, x):
        return self.feedfwd(self.attention(x))


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_dim = config['model_dim']
        ff_dim = config['ff_dim']
        num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.blocks = nn.ModuleList([EncoderBlock(model_dim, ff_dim, num_heads) for _ in range(self.num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.blocks[i](x)
        return x


class ClassificationHead(nn.Module):

    def __init__(self, model_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        ''' Input will have size (bs, seq_length, model_dim) '''
        return self.fc(x)