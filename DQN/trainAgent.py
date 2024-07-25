import DQNclass as DQN
import torch as th
import gfootball.env as football_env

env = football_env.create_environment(env_name="gm_level1", representation='simple115_v2', stacked=True, render=True, rewards='scoring,checkpoints')
env.reset()
steps = 0

# TODO: Implement the training loop and save the model
CR7 = DQN.Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=0.0001, gamma=0.99, batch_size=64, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.2, max_size=10000, fc1_dims=128, fc2_dims=128, fc3_dims=128)