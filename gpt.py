import torch
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout)
        self.drop_1 = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.drop_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.drop_1(self.attn(self.ln_1(x), x, x)[0])
        x = x + self.drop_2(self.mlp(self.ln_2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, block_size, device):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        # x is of shape (batch_size, block_size)
        B, T = x.shape
        x = self.embd(x) + self.pos_embd(torch.arange(T, device=self.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(x.transpose(1,2), targets)
        else:
            loss = None
        return x, loss
    
    def generate(self, x, max_len):
        for _ in range(max_len):
            x_cond = x[:, -self.block_size:]
            logits, _ = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, 1)
            x = torch.cat([x, x_next], dim=1)
        return x
