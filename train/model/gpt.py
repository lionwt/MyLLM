"""Decoder-only transformer components used by MiniGPT."""

import torch
from torch import nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """Multi-head self-attention restricted to current and previous tokens."""

    def __init__(self, dim, heads, dropout):
        """Create attention projections for a model width divisible by ``heads``."""
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply masked self-attention to a ``(batch, time, dim)`` tensor."""
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(z):
            """Expose the head dimension for parallel attention computation."""
            return z.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    """Pre-normalized transformer block with attention and feed-forward paths."""

    def __init__(self, dim, heads, dropout):
        """Construct one residual transformer block."""
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply attention and feed-forward residual updates."""
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class MiniGPT(nn.Module):
    """Small decoder-only GPT with learned positions and tied output weights."""

    def __init__(self, vocab_size, context=128, dim=256, heads=4, layers=4, dropout=0.1):
        """Construct token embeddings and a stack of transformer blocks."""
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(context, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        """Initialize linear and embedding weights with GPT-style values."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        """Return next-token logits for each position in a token-ID batch."""
        b, t = ids.shape
        if t > self.context:
            raise ValueError("Input longer than context")
        positions = torch.arange(t, device=ids.device)
        x = self.drop(self.token(ids) + self.position(positions)[None, :, :])
        return F.linear(self.norm(self.blocks(x)), self.token.weight)
