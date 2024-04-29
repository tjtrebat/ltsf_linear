import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import get_train_dataset, get_test_dataset


SEQUENCE_LENGTH = 96
PREDICTION_LENGTH = 96
BATCH_SIZE = 64

train_dataset = get_train_dataset(SEQUENCE_LENGTH, PREDICTION_LENGTH)
test_dataset = get_test_dataset(SEQUENCE_LENGTH, PREDICTION_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    break
