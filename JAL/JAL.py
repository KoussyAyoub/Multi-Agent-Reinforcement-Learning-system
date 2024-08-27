import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation 
import itertools

"""
in this file we will be trying to implement the Joint action learners MARL algorithm

Notes : 
        - the reward system is now centralized 
        - we add a contribution factor in the Q update rule

"""

class JointActionLearning : 
    '''
    Joint action learning class on a subgrid
    '''
    def __init__(self,env,learning_rate, discount_factor, exploration_ratio, max_iter ,exploration_depth):
        
        # the game environement 
        self.env = env

        # the number of states in our game 
        self.num_states = self.env.grid_length * self.env.grid_width

        # some algorithm hyperparameters
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.exploration_ratio = exploration_ratio
        self.depth, self.max_iter = exploration_depth, max_iter
        
        # our action space
        self.actions = {
            'STOP':[0, 0], # the agents stays still
            'UP':[0, 1],   # increase y by 1
            'DOWN':[0, -1], # decrease y by 1best_joint_action
            'RIGHT':[1, 0],# increase x by 1
            'LEFT':[-1, 0] # decrease x by 1
        }

        # a list of our actions :
        self.action_list = list(self.actions.values())


        # joint action space size :
        self.num_joint_actions = self.env.num_agents ** self.env.num_agents

        # our joint actions : 
        self.join_actions = list(itertools.product(list(range(len(self.action_list))), repeat=len(self.action_list)))

        # initiate the Q matrix of the size (num states , num of joint actions)
        self.Q_matrix = np.zeros(shape=(self.num_states, (self.num_joint_actions)))
        #print("the size of the Q matrix: (num_states, num_agents ** num_agents) ==>", self.Q_matrix.shape)
        
        # visualization stuff 
        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter([], []) # empty figure at first
        self.ax.set_xlim(0, self.env.grid_length) 
        self.ax.set_ylim(0, self.env.grid_width)

        # the positions to be displayed :
        self.X, self.Y = [], []



    def encode_joint_actions(self,joint_actions):
        """
        we encode our joint actions into indexes : (action1, action2, action3, ... ,actionn ) ==> unique Index
        """
        # we create a list of our action indexes : 
        action_idx = list(range(len(self.action_list)))

        # the size of our action space : 
        n =  len(action_idx)

        corr_index = 0
        for element in joint_actions:
            index = action_idx.index(element)
            corr_index = corr_index * n + index

        return corr_index
    
    def decode_index(self, index):
        """
        this function is to decode a column index of the Q table to the correponding joint actions : Idx ==> (a1, a2, a3, ..., an)
        """
        # we create a list of our action indexes : 
        action_idx = list(range(len(self.action_list)))

        # the size of our action space : 
        n =  len(action_idx)

        joint_actions = [action_idx[0]] * n

        if index > 0:
            for i in range(n - 1, -1, -1):
                remainder = index % n
                joint_actions[i] = action_idx[remainder]
                index //= n        

        return joint_actions
    
        
    def convert_state_idx(self, state):
        """
        its a method we need to convert a position state to a table index : (x, y) ==> idx
        """
        return state[0]*self.env.grid_length + state[1]

    def convert_action_idx(self, action):
        """
        a method to get the index of the action in order to access the Q tabel : idx ==> (x, y) 
        """
        index = self.action_list.index(action)
        return index
    
    
    def filter_joint_actions(self , state): # clear
        """
        remaves the joints action where at least one action of the joint action results into an agent getting out the environment bounds 
        """
        if state == 1 : 
            list_states = [agent.pos for agent in self.env.agents]
        else :
            list_states =[agent.next_state for agent in self.env.agents]

        test_list = []
        for state in list_states : 
            actions_to_exclude = [self.action_list.index(item) for item in self.check_invalid_index(state)]
            test_list.append(actions_to_exclude)


        possible_actions = []
        for joint_action in self.join_actions :
            flag = True 
            for agent_idx in range(self.env.num_agents):
                if len(test_list[agent_idx]) != 0 : 
                    if joint_action[agent_idx] in test_list[agent_idx] : 
                        flag = False
                        break
                        
            if flag :
                possible_actions.append(self.encode_joint_actions(joint_action))          
        return possible_actions # return the index of possible joint actions             

    
    
    
    def check_invalid_index(self, state):
        """
        this method returns the set of actions to exclude from the the current state. In order to avoid getting out of bounds of the grid
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

    
    def eps_greedy_policy(self, Q_matrix , list_states , exploration, available_positions):
        """
        this method is to select a joint action using the q matrix and an exploration probability epsilon
        """
        tmp_Q_matrix = Q_matrix[list_states]
        max_Q = np.max(tmp_Q_matrix, axis=1) 

        if np.random.uniform() < exploration : 
            joint_action = random.choice(available_positions) 
        else :
            # now we select the best joint action over all states
            state_joint_action = np.argmax(max_Q)
            joint_action = np.argmax(tmp_Q_matrix[state_joint_action])
            joint_action = available_positions[joint_action]

        # now we convert the joint action index to actions
        action_idx = self.decode_index(joint_action)
        actions = [self.action_list[item] for item in action_idx] 

        return actions, joint_action # in the form [[1, 0], [-1, 0]] ... 


    def train(self):
        """
        this method is to train our Q matrix:

        change : we need to apply this function to all the sub grids we divided our image into. (generalizing this code on the full image)
        """
        for episode in range(self.max_iter):

            print(f"*************episode{episode}*************")

            # we reset the environement to explore more state combinatinos 
            self.env.reset_env()
    
            for tick in range(self.depth):
                
                if all([agent.reached_end_state for agent in self.env.agents]):
                    print("all reached end state")
                    break

                # we get a list of possible joint actions in the current position 
                possible_joint_action_idx = self.filter_joint_actions(1)

                # now we get the current states of all our agents  :
                list_states = [self.convert_state_idx(agent.pos) for agent in self.env.agents]

                # now we get our joint actions following an eps greedy policy :
                actions, best_joint_action = self.eps_greedy_policy(self.Q_matrix[:, possible_joint_action_idx], list_states, self.exploration_ratio, possible_joint_action_idx)

                # we get the new states using our selected actions : 
                agent_idx = 0
                rewards, new_state_idx_list = [], []
                for agent in self.env.agents:
                    agent.action = actions[agent_idx] 
                    agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]
                    new_state_idx_list.append(self.convert_state_idx(agent.next_state))
                    rewards.append(self.env.reward_list[agent.next_state[0]][agent.next_state[1]])
                    agent_idx += 1

                # we now calculate the Q value of each state
                old_Q_values = [self.Q_matrix[state][best_joint_action] for state in list_states]

                # we need to calculate the contribution factor for each future state : 
                possible_futur_joint_action_idx = self.filter_joint_actions(0)
            
                # we select the best joint action for each state 
                best_next_Q = [max(self.Q_matrix[:, possible_futur_joint_action_idx][state]) for state in new_state_idx_list]

                # we now can calculate the new Q value : 
                agent_idx = 0 
                for agent in self.env.agents : 
                    if not agent.reached_end_state :
                        agent.reached_end_state = rewards[agent_idx][1]
                        # we update the Q value for each agent 
                        self.Q_matrix[list_states[agent_idx], best_joint_action] = (1 - self.lr) * old_Q_values[agent_idx] + self.lr * (rewards[agent_idx][0] + self.discount_factor * best_next_Q[agent_idx])
                    agent_idx += 1
                    

                # we update the environment with the new positions 
                self.env.update_env()

            # we can reduce the exploration proba where exploring the depth
            #self.exploration_ratio = self.exploration_ratio * ((self.max_iter - episode) / self.max_iter)
        
        return self.Q_matrix


        
    def update_agents(self, i):
        """
        this function is to test our trained Q matrix on Our environement
        """
        self.X , self.Y = [], []

        # now we get the current states of all our agents  :
        list_states = [self.convert_state_idx(agent.pos) for agent in self.env.agents]

        # the possible actions :
        possible_actions = self.filter_joint_actions(1)

        # now we get our joint actions following an eps greedy policy :
        actions, best_joint_action = self.eps_greedy_policy(self.Q_matrix[:, possible_actions], list_states, 0,possible_actions)  # empty available positions since no exploration

        print([agent.pos for agent in self.env.agents], actions)

        # update actions : 
        agent_idx = 0
        for agent in self.env.agents: 
            if not agent.reached_end_state :
                agent.action = actions[agent_idx]
                agent.next_state = [x + y for x, y in zip(agent.pos, agent.action)]
                agent.reached_end_state = self.env.reward_list[agent.next_state[0]][agent.next_state[1]][1]
                self.X.append(agent.next_state[0])
                self.Y.append(agent.next_state[1])   
                agent.move() 
            else :
                self.X.append(agent.pos[0])
                self.Y.append(agent.pos[1])  

            agent_idx += 1 
        
        #self.env.update_env()
        test = np.c_[self.X, self.Y]
        #test = np.stack([self.X, self.Y]).T
        self.scat.set_offsets(offsets=test)
        return self.scat, 


    def main(self, max_mov):
        """
        Our main function (training & simualation)
        """
        # we train our Q matrix 
        self.train()
        
        # we reset our environment for simulation
        self.env.reset_env()

        # i need to iterate over all the frames
        animation = FuncAnimation(self.fig, self.update_agents, frames = max_mov, interval = 100)

        #Save the animation frames as individual images
        animation.save("animation.gif", writer='pillow', fps=10)
        plt.show()
        



