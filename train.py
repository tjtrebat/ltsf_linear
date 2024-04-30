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

for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(torch.float32), y.to(torch.float32)
    y_hat = model(X.to(device))
    