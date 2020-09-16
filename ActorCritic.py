import numpy as np
from copy import deepcopy

class ACAgent: # One-step, pg 332

    def __init__(self, initial = 'S', gamma = 1.0, alpha = 0.1, beta = 0.01): # Change to use ch 2 trick later
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.actions = ['L', 'R']
        self.initial = initial
        self.intensities = {}
        self.values = {}

        self.firstSeen(self.initial)
        self.reset()


    def reset(self):
        self.state = self.initial
        self.I = 1.0
       

    def firstSeen(self, state):
        self.intensities[state] = np.zeros(len(self.actions), dtype='float64')
        self.values[state] = 0.0

    
    def valueLookup(self, state):
        try:
            return self.values[state]
        except KeyError:
            self.values[state] = 0.0
            return 0.0


    def probs(self, state):
        try:
            magnitudes = np.exp(self.intensities[state])
        except KeyError:
            self.firstSeen(state)
            magnitudes = np.exp(self.intensities[state])

        return magnitudes / np.sum(magnitudes)


    def pickAction(self, state):
        probs = self.probs(state)        
        r = np.random.random()
        i = 0
        s = probs[i]
        while s < r:
            i += 1
            s += probs[i]
        
        return self.actions[i]
    

    def actionToIndex(self, action): # Hacky; change for future
        ind = self.actions.index(action)
        if ind >= 0: # Found
            return ind
        else:
            print("Not found; check action " + str(action))
            return None
    

    def eligibility(self, state, action): # Gradient of ln prob(a | s)
        # This math works for softmax; reader is welcome to rederive it
        ind = self.actionToIndex(action)
        g = np.zeros(len(self.actions), dtype='float64')
        g[ind] = 1.0
        g -= self.probs(state)
        return g


    def move(self, env):
        action = self.pickAction(self.state)
        newState, R = env.move(move = action)
        
        delta = R + self.gamma*self.valueLookup(newState) - self.valueLookup(self.state)

        self.values[self.state] += self.beta*delta

        g = self.eligibility(self.state, action)
        self.intensities[self.state] += self.alpha*self.I*delta*g
        
        self.I *= self.gamma
        self.state = newState


    def episode(self, env):
        self.reset()
        i = 0
        while not env.complete:
            i += 1
            self.move(env)
        return i

