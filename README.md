# Multi-Agent Reinforcement Learning System

This repository contains an implementation of the Independent Q Learning (IQL) and Joint Action Learning (JAL). These algorithms are used in a project where agents are trained to form a given shape based on an input image.

## Project Overview

The goal of this project is to train multiple agents to form a specific shape based on an input image. The agents interact with an environment and learn to navigate and position themselves in order to collectively create the desired shape.

## Algorithms

### Independent Q Learning (IQL)

The Independent Q Learning algorithm is used to train the agents independently. Each agent maintains its own Q-matrix and learns from its own experiences. The algorithm involves exploration and exploitation phases, where the agents take actions based on an epsilon-greedy policy and update their Q-values accordingly.

### Joint Action Learning (JAL)

The Joint Action Learning algorithm is used to train the agents collectively. The agents learn to coordinate their actions and make joint decisions to achieve the desired shape. The algorithm involves a centralized Q-matrix that represents the joint action-value function. The agents take actions based on a joint policy and update the Q-values based on the rewards received.

## Implementation

The implementation of the project consists of several Python files:

- `Environment.py`: This file contains the implementation of the environment where the agents interact. It includes methods for initializing the environment, updating the positions of the agents, and resetting the environment.

- `Agent.py`: This file contains the implementation of the Agent class. Each agent has attributes such as position, identifier, and action. The agents interact with the environment and update their positions based on the chosen actions.

- `IQL.py`: This file contains the implementation of the Independent Q Learning algorithm. It includes methods for initializing the algorithm, exploring and exploiting actions, and updating the Q-values based on rewards.

- `JAL.py`: This file contains the implementation of the Joint Action Learning algorithm. It includes methods for initializing the algorithm, coordinating actions among agents, and updating the joint Q-values based on rewards.

- `process_input.py`: This file contains the implementation of the image processing functionality. It includes methods for converting an input image to a black and white representation, adding padding to the image, and extracting agent positions from the image.

- `Visualization.py`: This file contains the implementation of the visualization functionality. It includes methods for visualizing the environment updates and creating an animation of the agents' movements.

## Usage

To use this project, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Prepare your input image by placing it in the appropriate directory.

3. Modify the necessary parameters in the code, such as the number of agents, the image path, and the algorithm parameters.

4. Run the main script to start the training process.

## Future Updates

This README file will be updated with more detailed information about the project, including instructions on how to run the code and explanations of the algorithms and implementation details.

Stay tuned for more updates!

## License

This project is licensed under the [MIT License](LICENSE).ible :=)
      




