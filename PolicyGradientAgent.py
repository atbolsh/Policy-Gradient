import numpy as np
from copy import deepcopy

class MCAgent: #Monte Carlo Agent; this uses the REINFORCE algorithm

    def __init__(self, initial = 'S', gamma = 1.0, alpha = 0.01):
        self.gamma = gamma
        self.alpha = alpha

        self.actions = ['L', 'R']
        self.initial = initial
        self.intensities = {}
        self.firstSeen(self.initial)
        self.reset()
    

    def reset(self):
        self.state = self.initial
        self.sTrace = [deepcopy(self.initial)]
        self.aTrace = []
        self.rTrace = []
        

    def firstSeen(self, state):
        self.intensities[state] = np.zeros(len(self.actions), dtype='float64')
    

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
    

    def move(self, env):
        a = self.pickAction(self.state)
        newState, R = env.move(move = a)
        self.aTrace.append(a)
        self.rTrace.append(R)
        self.state = newState
        self.sTrace.append(newState)
 

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
    
    
    def learn(self): # Process an episode
        Gs = np.zeros(len(self.rTrace), dtype='float64')
        for i in range(1, len(self.rTrace) + 1):
            Gs[-i] = self.rTrace[-i]
            if i > 1:
                Gs[-i] += self.gamma*Gs[1 - i]
        
        for t in range(len(self.rTrace)):
            g = self.eligibility(self.sTrace[t], self.aTrace[t]) 
            self.intensities[self.sTrace[t]] += self.alpha*(self.gamma**t)*g
        
        
    def episode(self, env, verbose = False):
        self.reset()
        while not env.complete:
            self.move(env)
        self.learn()
        if verbose:
            return self.sTrace, self.aTrace, self.rTrace
        else:
            return len(self.sTrace)





