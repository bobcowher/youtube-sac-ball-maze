import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from model import *
from agent import Agent
from buffer import ReplayBuffer

if __name__ == '__main__':

    replay_buffer_size = 1000000
    episodes = 1000
    batch_size = 64
    updates_per_step = 4
    gamma = 0.99
    tau = 0.99
    alpha = 0.12
    target_update_interval = 1
    hidden_size = 512
    learning_rate = 0.0001
    env_name = "PointMaze_UMaze-v3"
    max_episode_steps = 100
    exploration_scaling_factor=1.5

    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]

    env = gym.make(env_name, max_episode_steps=max_episode_steps, render_mode='human', maze_map=STRAIGHT_MAZE)
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    observation_size = observation.shape[0]


    # Agent
    agent = Agent(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, target_update_interval=target_update_interval,
                  hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)
    
    agent.load_checkpoint(evaluate=True)

    agent.test(env=env, episodes=10, max_episode_steps=200)

    env.close()