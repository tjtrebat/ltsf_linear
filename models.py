import torch
import torch.nn as nn


class LinearLTSF(nn.Module):
    def __init__(self, sequence_length, prediction_length):
        super(LinearLTSF, self).__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.appliances_pred = nn.Linear(sequence_length, prediction_length)
        self.lights_pred = nn.Linear(sequence_length, prediction_length)

    def forward(self, x):
        appliances_out = self.appliances_pred(x[:, :, 0])
        lights_out = self.lights_pred(x[:, :, 1])
        out = torch.cat([appliances_out, lights_out], dim=-1)
        return out
