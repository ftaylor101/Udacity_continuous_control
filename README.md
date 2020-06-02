# Project 2 - Udacity Reacher
## Challenge
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that 
the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target 
location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of 
the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in 
the action vector should be a number between -1 and 1.

## Distributed Training
Two separate versions of the environment are available for this project, this solution uses version 2, the environment 
with 20 agents. This version contains 20 identical agents, each with its own copy of the environment.  

## Solving the Environment
To solve the environment with 20 agents, each agent must get an average score of +30 (over 100 consecutive episodes, 
and over all agents).  This is calculated by:
+ After each episode, add up the reward that each agent received at each timestep, therefore getting a score for each 
agent for the entire episode and giving us 20 scores for an episode.
+ Taking the average of these 20 scores gives an avergae score for each episode across all 20 agents.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Development environment
+ This agent has been trained using __Python 3.6.8__ on __Windows 10__
+ The __requirements.txt__ file contains all required Python packages
+ To install these packages, navigate to your Python virtual 
environment, activate it and use: 
     - pip install -r requirements.txt 

The Reacher environment is provided by Udacity has been used for training.
The environment used for this project is:
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

__Ensure__ the location of the Udacity modified Reacher environment folder is in the same folder as the ac_run.py 
file. This will make sure the code finds the environment.

## Running code
ac_run.py contains the entry point to the code. Running this code will set up a network and an agent which will then 
commence training. Results are printed to the console and plotted at the end. The script will instantiate the Unity 
environment and an A2CAgent class and use these in the train_ac function. This function loops through episodes and 
applies the Bellman rollouts. The update algorithm is called at the end of each rollout to update the network.
Make sure the Unity Environment is given the correct argument for the file_name parameter, which should be the the file 
path, file name and extension of the Unity Environment.
