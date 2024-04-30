import torch
import torch.nn as nn


class LinearLTSF(nn.Module):
    def __init__(self, sequence_length, prediction_length, in_channels=2):
        super(LinearLTSF, self).__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.linears = nn.ModuleList([
            nn.Linear(sequence_length, prediction_length)
            for _ in range(in_channels)
        ])

    def forward(self, x):
        out = torch.tensor([], dtype=x.dtype, device=x.device)
        for channel, linear in enumerate(self.linears):
            channel_out = linear(x[:, :, channel]).unsqueeze(-1)
            out = torch.cat([out, channel_out], dim=-1)
        return out
