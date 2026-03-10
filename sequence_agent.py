import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LABELS = 3
SEQ_LEN = 32


####################################################
# TRANSFORMER SEQUENCE MODEL
####################################################

class SequencePolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(NUM_LABELS,64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64,NUM_LABELS)


    def forward(self,x):

        x = self.embedding(x)

        x = self.transformer(x)

        return self.fc(x[:,-1])


####################################################
# RL AGENT
####################################################

class SequenceAgent:

    def __init__(self):

        self.model = SequencePolicy()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4
        )

        self.last_logprob = None
        self.last_action = None


####################################################
# PREDICT NEXT
####################################################

    def predict_next(self,seq):

        seq = torch.tensor(seq).long().unsqueeze(0)

        logits = self.model(seq)

        probs = F.softmax(logits,dim=-1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        self.last_action = action.item()

        self.last_logprob = dist.log_prob(action)

        return self.last_action


####################################################
# RL UPDATE
####################################################

    def update(self,gt):

        if self.last_action is None:
            return

        reward = 1.0 if gt == self.last_action else -1.0

        loss = -self.last_logprob * reward

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
