

"""
In this file we implement the Agent class with all its attributes
"""


class Agent:
    '''
    agent class where we initialize each agent of our environement
    '''
    def __init__(self, pos):
        self.pos = pos # initial pos of the agent
        self.reached_end_state = False # a boolean to check if the agent reached an end state
        self.action = [0, 0] # initial action is to stay at place
        self.next_state = 0 # the updated position ~ the next state

    def move(self):
        """
        This method is for updating the state (position) of the agent using the current action 
        """ 
        self.pos = self.next_state





        