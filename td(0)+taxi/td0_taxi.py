import gymnasium as gym
import sys
import math
import time
import numpy as np
from collections import defaultdict, deque

class Agent:
    def __init__(self, nA=6, alpha=0.2):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = 1.0
        self.epsilon = 0.0005

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = [self.epsilon / self.nA] * self.nA
        policy[np.argmax(self.Q[state])] += 1 - self.epsilon
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done):

        old_value = self.Q[state][action]

        next_max = np.max(self.Q[next_state])

        new_value = old_value + self.alpha * (reward + (self.gamma * next_max) - old_value)

        self.Q[state][action] = new_value