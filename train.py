import torch
from torch import nn
from torch.utils.data import DataLoader

from models import LinearLTSF
from dataset import get_train_dataset, get_test_dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataloader, model, criterion, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = [_.to(torch.float32).to(device) for _ in (X, y,)]
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, criterion):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = [_.to(torch.float32).to(device) for _ in (X, y,)]
            pred = model(X)
            test_loss += criterion(pred, y).item()
    test_loss /= num_batches
    return test_loss


loss_data = {'Prediction Length': [], 'Sequence Length': [], 'Avg Loss': []}

for pred_length in [24, 720]:
    for seq_length in [48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720]:
        train_dataset = get_train_dataset(seq_length, pred_length)
        test_dataset = get_test_dataset(seq_length, pred_length)
        train_dataloader = DataLoader(train_dataset, batch_size=64)
        test_dataloader = DataLoader(test_dataset, batch_size=64)
        model = LinearLTSF(seq_length, pred_length).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            train(train_dataloader, model, criterion, optimizer)
        test_loss = test(test_dataloader, model, criterion)
        loss_data['Avg Loss'].append(test_loss)
        loss_data['Prediction Length'].append(pred_length)
        loss_data['Sequence Length'].append(seq_length)

print(loss_data)