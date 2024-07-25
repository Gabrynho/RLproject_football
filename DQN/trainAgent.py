import DQNclass as DQN
import torch as th
import os
import gfootball.env as football_env

env = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=True, render=False, rewards='scoring,checkpoints')
env.reset()
steps = 0

CR7 = DQN.Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=0.0001, gamma=0.99, batch_size=64, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.2, max_size=10000, fc1_dims=128, fc2_dims=128, fc3_dims=128)
if os.path.exists('best_model.pth'):
    CR7.q_eval.load_state_dict(th.load('best_model.pth'))
    CR7.q_eval.eval()
    CR7.q_eval.to(CR7.q_eval.device)

for i in range(100): # number of episodes
    done = False
    observation = env.reset()
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        score += reward
        CR7.store_transition(observation, action, reward, next_observation, done)
        CR7.learn()
        observation = next_observation
        steps += 1
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('best_model.pth')
env.close()
CR7.save_model('best_model.pth')