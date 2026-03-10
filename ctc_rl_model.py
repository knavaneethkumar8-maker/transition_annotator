import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################
# MODEL
#############################################

class CTCModel(nn.Module):

    def __init__(self, input_dim, vocab_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):

        out, _ = self.lstm(x)

        logits = self.fc(out)

        return logits
