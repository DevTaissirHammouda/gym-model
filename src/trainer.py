import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = torch.tensor(self.tokenizer(self.questions[idx]), dtype=torch.long)
        a = torch.tensor(self.tokenizer(self.answers[idx]), dtype=torch.long)
        return q, a

def train(model, dataset, epochs=3, batch_size=16, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for q, a in tqdm(loader):
            optimizer.zero_grad()
            out = model(q)
            loss = criterion(out, a)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
