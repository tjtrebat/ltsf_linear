import torch
from torch import nn
from torch.utils.data import DataLoader

from models import LinearLTSF
from dataset import get_train_dataset, get_test_dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
SEQUENCE_LENGTH = 96
PREDICTION_LENGTH = 720

train_dataset = get_train_dataset(SEQUENCE_LENGTH, PREDICTION_LENGTH)
test_dataset = get_test_dataset(SEQUENCE_LENGTH, PREDICTION_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = LinearLTSF(SEQUENCE_LENGTH, PREDICTION_LENGTH).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(torch.float32), y.to(torch.float32)
        pred = model(X.to(device))
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 10
for epoch in range(epochs):
    print(f'Epoch {epoch+1}\n-------------------------------')
    train()
print('Done!')
