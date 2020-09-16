import numpy as np
from copy import deepcopy

class exampleEnv:

    def __init__(self):
        self.state = 0
        self.complete = False
    
    def move(self, state=None, move='L'): #Moves are 'L' or 'R'
        if type(state) != type(None):
            self.state = state
        
        if move == 'L':
            update = -1
        elif move == 'R':
            update = 1
        else:
            print("Error: please provide valid move")
            return None
        
        if self.state == 1:
            update = -update
        
        self.state = max(self.state + update, 0)
        if self.state < 3:
            return 'S', -1
        else:
            self.complete = True
            return 'G', 0


