import numpy as np

class MarkovDecisionProcess:
    def __init__(self, states, actions, transitions, rewards, step):
        self.states         = states
        self.actions        = actions
        self.transitions    = transitions
        self.rewards        = rewards
        self.step           = step

        self.n_states       = len(self.states)
        self.n_actions      = len(self.actions)

