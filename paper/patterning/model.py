"""
Small transformer for parenthesis balancing classification.

Following Li et al. (2025): 2-3 layers, 4 heads.
Input: sequence of ( and ), output: TRUE/FALSE classification.
"""

import torch
import torch.nn as nn
import math


class ParenTransformer(nn.Module):
    """Small transformer for parenthesis balancing."""

    def __init__(self, d_model=64, n_heads=4, n_layers=2, max_len=42, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        # Token embedding: ( = 0, ) = 1, pad = 2
        self.token_embed = nn.Embedding(3, d_model, padding_idx=2)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head: pool then project to 2 classes
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) of token ids (0 = (, 1 = ), 2 = pad)
        Returns: (batch, 2) logits for [FALSE, TRUE]
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = self.token_embed(x) + self.pos_embed(positions)

        # Create padding mask
        pad_mask = (x == 2)  # True where padded

        h = self.transformer(h, src_key_padding_mask=pad_mask)

        # Pool: mean over non-padded positions
        mask_expanded = (~pad_mask).float().unsqueeze(-1)
        h_pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return self.classifier(h_pooled)


def tokenize_sequence(seq, max_len=42):
    """Convert parenthesis string to token ids."""
    ids = [0 if c == '(' else 1 for c in seq]
    # Pad to max_len
    ids = ids + [2] * (max_len - len(ids))
    return ids


def collate_batch(examples, max_len=42):
    """Collate a batch of examples into tensors."""
    seqs = [tokenize_sequence(ex["seq"], max_len) for ex in examples]
    labels = [1 if ex["label"] else 0 for ex in examples]
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
