import torch
import torch.nn as nn
import numpy as np


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, ff_dim=64, max_seq_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        out = self.transformer(emb)
        return out.mean(dim=1)  # Mean pooling for dense sentence embedding


def generate_embeddings(texts, vocab, max_len=20):
    model = TinyTransformer(vocab_size=len(vocab))
    model.eval()

    encoded = []
    for text in texts:
        tokens = [vocab.get(t, 1) for t in text]  # 1 is UNK
        padded = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        encoded.append(padded)

    tensor_input = torch.tensor(encoded, dtype=torch.long)
    with torch.no_grad():
        embeddings = model(tensor_input).numpy()
    return embeddings