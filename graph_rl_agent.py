import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#################################################
# Graph RL Policy Network
#################################################

class GraphPolicy(nn.Module):

    def __init__(self, audio_dim, vocab_size):

        super().__init__()

        self.audio_encoder = nn.Linear(audio_dim, 128)

        self.token_embedding = nn.Embedding(vocab_size, 64)

        self.fc = nn.Linear(128 + 64, vocab_size)

    def forward(self, audio_feat, prev_token):

        a = self.audio_encoder(audio_feat)

        t = self.token_embedding(prev_token)

        x = torch.cat([a, t], dim=-1)

        logits = self.fc(x)

        return logits


#################################################
# Graph RL Agent
#################################################

class GraphRLAgent:

    def __init__(self, vocab):

        self.vocab = vocab
        self.token_to_id = {t:i for i,t in enumerate(vocab)}
        self.id_to_token = {i:t for t,i in self.token_to_id.items()}

        self.model = GraphPolicy(
            audio_dim=200,        # 5 frames × 40 MFCC
            vocab_size=len(vocab)
        )

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4
        )

        self.last_logprob = None
        self.last_action = None


#################################################
# Predict next token
#################################################

    def predict(self, audio_context, prev_token):

        audio_tensor = torch.tensor(audio_context).float().unsqueeze(0)

        prev_id = torch.tensor([self.token_to_id[prev_token]])

        logits = self.model(audio_tensor, prev_id)

        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        logprob = dist.log_prob(action)

        self.last_action = action.item()
        self.last_logprob = logprob

        return self.id_to_token[self.last_action]


#################################################
# RL Update
#################################################

    def update(self, true_token):

        true_id = self.token_to_id[true_token]

        reward = 1.0 if true_id == self.last_action else -1.0

        loss = -self.last_logprob * reward

        self.optim.zero_grad()

        loss.backward()

        self.optim.step()


#################################################
# Context window
#################################################

def get_audio_context(features, t):

    ctx = []

    for i in range(t-2, t+3):

        if i < 0:
            ctx.append(features[0])

        elif i >= len(features):
            ctx.append(features[-1])

        else:
            ctx.append(features[i])

    return np.concatenate(ctx)
