import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

SR = 16000
FRAME = int(0.054 * SR)

LABEL_MAP = {
    "∅":0,
    "C":1,
    "V":2
}

INV_LABEL = {v:k for k,v in LABEL_MAP.items()}


##################################################
# MODEL
##################################################

class PolicyNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128,3)

    def forward(self,x):

        out,_ = self.lstm(x)

        logits = self.fc(out[:,-1])

        return logits


##################################################
# RL AGENT
##################################################

class RLAgent:

    def __init__(self):

        self.model = PolicyNet()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4
        )

        self.last_logprob = None
        self.last_action = None


##################################################
# FEATURE EXTRACTION
##################################################

    def extract_features(self,wav):

        y, sr = librosa.load(wav, sr=SR)

        feats = []

        for i in range(0,len(y)-FRAME,FRAME):

            seg = y[i:i+FRAME]

            mfcc = librosa.feature.mfcc(
                y=seg,
                sr=sr,
                n_mfcc=40
            )

            feats.append(mfcc.mean(axis=1))

        return np.array(feats)


##################################################
# PREDICT NEXT LABEL
##################################################

    def predict_next(self,feature):

        x = torch.tensor(feature).float().unsqueeze(0).unsqueeze(0)

        logits = self.model(x)

        probs = F.softmax(logits,dim=-1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        logprob = dist.log_prob(action)

        self.last_logprob = logprob
        self.last_action = action.item()

        return self.last_action


##################################################
# RL UPDATE
##################################################

    def update(self,gt_label):

        if self.last_action is None:
            return

        if gt_label == self.last_action:
            reward = 1.0
        else:
            reward = -1.0

        loss = -self.last_logprob * reward

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


##################################################
# FINAL SEQUENCE CONVERSION
##################################################

def text_to_cv_sequence(text):

    swar = set("अआइईउऊएऐओऔ")

    seq = []

    for ch in text:

        if ch == " ":
            seq.append("∅")

        elif ch in swar:
            seq.append("V")

        else:
            seq.append("C")

    return seq


##################################################
# TRAIN LOOP WITH FINAL SEQUENCE
##################################################

def train_with_sequence(agent, wav, text):

    features = agent.extract_features(wav)

    seq = text_to_cv_sequence(text)

    for i in range(min(len(features),len(seq))):

        pred = agent.predict_next(features[i])

        gt = LABEL_MAP[seq[i]]

        agent.update(gt)
