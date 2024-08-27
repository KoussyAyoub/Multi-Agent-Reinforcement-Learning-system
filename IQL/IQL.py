import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
points to pay attention to and improve : 
    - when two agent's new state is the same 
"""


class IQL : 
    """
    Independent Q learning algorithm
    """
    def __init__(self, env , learning_rate, discount_rate, epsilon, max_iter, exploration_depth):

        # our game environment :
        self.env = env

        # some hyperparameters for our algo :
        self.lr, self.discount_rate, self.eps, self.max_iter = learning_rate, discount_rate, epsilon, max_iter

        # we set a max number of iterations in explorations for each episode
        self.depth = exploration_depth

        # some algo params 
        self.num_states = self.env.grid_length * self.env.grid_width
        self.action_list = list(self.env.actions.values())

        # we create a list of Q_matrices for all our agents
        self.Q_matrices_list = []

        for agent in self.env.agents :
            tmp_Q_matrix = np.zeros((self.num_states, len(self.env.actions)))
            self.Q_matrices_list.append([agent,tmp_Q_matrix])

        self.fig, self.ax = plt.subplots()
        self.ax.invert_yaxis()
        self.scat = self.ax.scatter([], []) # empty figure at first
        self.ax.set_xlim(0, self.env.grid_length) 
        self.ax.set_ylim(0, self.env.grid_width)

        # the positions to be displayed :
        self.X, self.Y = [], []


    
    def convert_state_idx(self, state):
        """
        its a method we need to convert a position state to a table index
        """
        return state[0]*self.env.grid_length + state[1]



    def convert_action_idx(self, action):
        """
        a method to get the index of the action in order to access the Q tabel
        """
        index = self.action_list.index(action)
        return index


    def check_invalid_index(self, state):
        """
        this method returns the set of actions to exclude from the the current state. In order to avoid getting out of bounds of the grid
        we have to make a distinction where, the agent in the corners of the grid where i should exclude two action
        maybe return the actions that i should exclude ??
        """

        x, y = state[0], state[1]   

        actions_to_exclude = []

        # check the x boundaries
        if x == self.env.grid_length - 1 :
            actions_to_exclude.append([1, 0])

        if x == 0 : 
            actions_to_exclude.append([-1, 0])

        # check the y boundaries
        if y == self.env.grid_width - 1 : 
            actions_to_exclude.append([0, 1])
        
        if y == 0 : 
            actions_to_exclude.append([0, -1])
        
        return actions_to_exclude



    def eps_greedy_policy(self, Q_matrix , state, exploration):
        """
        this method is to select an action using the q matrix and an exploration proba
        """
        u = np.random.uniform()
        if u < exploration : 
            action = random.choice(self.action_list)

        else : 
            action_idx = np.argmax(Q_matrix[state])
            action = self.action_list[action_idx] 

        return action
    

    def train(self):
        """
            Our main training function 
        """

        print("training has begun !!")

        for episode in range(self.max_iter):

            #print(f"***************** episode {episode} ******************")

            # reset the env in each ep for agents to test multiple  :
            self.env.reset_env()

            # for each tick, each agent continues exploring his state
            for tick in range(self.depth) :
        
                agent_itr = 0

                for agent in self.env.agents : 

                    # this agent's Q matrix :
                    agent_Q_matrix = self.Q_matrices_list[agent_itr][1]

                    if not agent.reached_end_state :     

                        # Let's get the current state (position) of the agent
                        current_state = agent.pos
                        curr_state_idx = self.convert_state_idx(current_state)
                        
                        # now we select and action using an epsilon greedy policy
                        action = self.eps_greedy_policy(agent_Q_matrix, curr_state_idx, self.eps ) #* ((self.depth - tick) / self.depth)
                        
                        # we check the actions that agent cant do :
                        actions_to_remove = self.check_invalid_index(current_state)

                        if action in actions_to_remove : 
                            
                            reward = -100 # we hard code a big sanction if the agent is out of bounds

                            # actions that wont take us out of bounds
                            tmp_actions = self.action_list[:]
                            for item in actions_to_remove : 
                                tmp_actions.remove(item)
                            

                            # we select the action with the biggest Q value from the actions we can do
                            action_idx_list = [self.action_list.index(item) for item in tmp_actions]
                            tmp_Q_matrix = self.Q_matrices_list[agent_itr][1][:,action_idx_list]

                            action_id = np.argmax(tmp_Q_matrix[curr_state_idx])
                            agent.action = tmp_actions[action_id]

                            # we now need to update the Q table
                            curr_state_idx, action_idx = self.convert_state_idx(agent.pos) ,self.convert_action_idx(agent.action)

                            # out old Q value : 
                            old_Q_value = agent_Q_matrix[curr_state_idx][action_idx]

                            agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]

                            agent.reached_end_state = self.env.reward_list[agent.next_state[0]][agent.next_state[1]][1]

                            Best_new_state_Q = max(agent_Q_matrix[self.convert_state_idx(agent.next_state)])
                        
                            self.Q_matrices_list[agent_itr][1][curr_state_idx][action_idx] = (1 - self.lr) * old_Q_value + self.lr*(reward + 
                                                                                                        self.discount_rate * Best_new_state_Q)
                            

                        else : 
                            
                            # we keep the same action
                            agent.action = action

                            # we get the indexes of pos and action in Q table 
                            curr_state_idx, action_idx = self.convert_state_idx(agent.pos) ,self.convert_action_idx(agent.action)

                            # out old Q value : 
                            old_Q_value = agent_Q_matrix[curr_state_idx][action_idx]

                            # we calculate the new state : 
                            agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]
                            
                            reward_values = self.env.reward_list[agent.next_state[0]][agent.next_state[1]]

                            reward , agent.reached_end_state = reward_values[0], reward_values[1]

                            # the best Q value in the next state
                            Best_new_state_Q = max(agent_Q_matrix[self.convert_state_idx(agent.next_state)])
                            

                            # we then update the Q table
                            self.Q_matrices_list[agent_itr][1][curr_state_idx][action_idx] = (1 - self.lr) * old_Q_value + self.lr*(reward + 
                                                                                                        self.discount_rate * Best_new_state_Q)
                            
                            
                        # the agent excecutes the action and updates to new state 
                        agent.move()
                    
                    agent_itr += 1
            
            # we can decay the exploration rate (self.eps) by 1 / max_iter
            #self.eps = self.eps * ((self.max_iter - episode) / (1 * self.max_iter))
            

        print("training finished successfully")

        return self.Q_matrices_list
    


    def update_agents(self, i):
        """
        a function that updates the states of agents with no exploration (simulation with trained Q tables)
        """

        agent_idx = 0 
        self.X , self.Y = [], []

        for agent in self.env.agents : 
                
            if not agent.reached_end_state :

                # this agent's Q matrix :
                agent_Q_matrix = self.Q_matrices_list[agent_idx][1]

                # Let's get the agents current state
                curr_state = agent.pos
                curr_state_idx = self.convert_state_idx(curr_state)

                # get the propriate action with no exploration : 
                action = self.eps_greedy_policy(agent_Q_matrix, curr_state_idx, 0)
                
                # we check the actions that agent cant do :
                actions_to_remove = self.check_invalid_index(curr_state)

                if action in actions_to_remove : 
                
                    # actions that wont take us out of bounds
                    tmp_actions = self.action_list[:]
                    for item in actions_to_remove : 
                        tmp_actions.remove(item)
                    
                    # the agent take a random choice from the actions he is presented (can be improved ila khdinaha mn Q table)
                    action_idx_list = [self.action_list.index(item) for item in tmp_actions]
                    tmp_Q_matrix = self.Q_matrices_list[agent_idx][1][:,action_idx_list]

                    action_id = np.argmax(tmp_Q_matrix[curr_state_idx])
                    agent.action = tmp_actions[action_id]

                    agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]

                    agent.reached_end_state = self.env.reward_list[agent.next_state[0]][agent.next_state[1]][1]
                
                else : 

                    # we continue with the chosen action
                    agent.action = action 

                    agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]

                    agent.reached_end_state = self.env.reward_list[agent.next_state[0]][agent.next_state[1]][1]

                self.env.rewards += self.env.reward_list[agent.next_state[0]][agent.next_state[1]][0]                
                
                # for updatting the plot : 
                self.X.append(agent.next_state[0])
                self.Y.append(agent.next_state[1])

                # update the agent position with the new state 
                agent.move()
            
            else : 
                # for updatting the plot : 
                self.X.append(agent.pos[0])
                self.Y.append(agent.pos[1])


            agent_idx += 1 

        test = np.c_[self.X, self.Y]

        #test = np.stack([self.X, self.Y]).T

        self.scat.set_offsets(offsets=test)
        #self.fig.savefig(f"frame_{i}.png")
        
        return self.scat, 
        
    def main(self, max_mov):
        """
        a function where we execute all the functions
        """
        # we train our Q table
        self.train()
      
        # we reset the environment 
        self.env.reset_env()

        animation = FuncAnimation(self.fig, self.update_agents, frames = max_mov, interval = 100)
        animation.save("animation.gif", writer='pillow', fps=10)
        plt.show()





            












