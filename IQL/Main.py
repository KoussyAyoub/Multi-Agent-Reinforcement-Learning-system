from Environment import Environment
from IQL import IQL

"""
This is the excutable file to test the algorithm, we load the environment then we run the IQL algorithm
"""


#Let's load our environment 
env = Environment(grid_length=25, grid_width=25, num_agents=150, image_path="IQL\images\img3.png", extra_padding=True)

# Now we instanciate our algorithm
learning_algo = IQL(env, learning_rate=0.7, discount_rate=0.9, epsilon=0.9,  max_iter=1500 ,  exploration_depth = 500)
learning_algo.main(5)














