from Environment_JAL import Environment
from JAL import JointActionLearning

# Let's load our environment 
env = Environment(grid_length=5, grid_width=5, num_agents= 5, image_path="JAL\images\JAL_test.png", extra_padding=False)

joint = JointActionLearning(env, learning_rate=0.7, discount_factor=0.9, exploration_ratio=0.8, max_iter=1000, exploration_depth=1000)

joint.main(10)





















