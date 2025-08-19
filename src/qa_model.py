import torch
import torch.nn as nn

class SimpleQA(nn.Module):
    """
    Basic QA model: context + question -> answer embedding similarity
    """
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=128):
        super(SimpleQA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, vocab_size)  # generate answer as token logits

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.encoder(emb)
        out = self.fc(out[:, -1, :])
        return out
