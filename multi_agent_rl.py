from rl_agent import RLAgent
from sequence_agent import SequenceAgent
import numpy as np


class MultiAgent:

    def __init__(self):

        self.frame_agent = RLAgent()

        self.seq_agent = SequenceAgent()


###################################################
# COMBINED PREDICTION
###################################################

    def predict(self,feature,seq_context):

        p1 = self.frame_agent.predict_next(feature)

        p2 = self.seq_agent.predict_next(seq_context)

        # voting
        pred = int(round((p1+p2)/2))

        return pred


###################################################
# UPDATE BOTH AGENTS
###################################################

    def update(self,gt):

        self.frame_agent.update(gt)

        self.seq_agent.update(gt)
