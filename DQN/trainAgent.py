import DQNclass as DQN
import torch as th
import pandas as pd
import os
import gfootball.env as football_env

############################################
# Training the agent
############################################

# Create the environment
level1_env = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')

# Initialize the DQN agent
CR7 = DQN.Agent(input_dims=level1_env.observation_space.shape, n_actions=level1_env.action_space.n, lr=.00001, gamma=0.99, batch_size=64, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.01, max_size=10000, fc1_dims=128, fc2_dims=128, fc3_dims=128)

# Load the template if it exists
if os.path.exists('DQN/CR7_model.pth'):
    CR7.q_eval.load_state_dict(th.load('DQN/CR7_model.pth'))
    CR7.q_eval.eval()
    CR7.q_eval.to(CR7.q_eval.device)

############################################
# Level 1 Training
############################################

totalsteps = 0

# Training loops
for i in range(1000): # number of episodes
    done = False
    observation = level1_env.reset()
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        next_observation, reward, done, info = level1_env.step(action)
        score += reward
        CR7.store_transition(observation, action, reward, next_observation, done)
        CR7.learn()
        observation = next_observation
        steps += 1
        totalsteps += 1
        if score > 1 & done==True:
            print("Goal!")
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('DQN/CR7_model.pth')
level1_env.close()
CR7.save_model('DQN/CR7_model.pth')

# Test the agent and see how it acts
def add_observation(observation):
    observations.append(observation)

def save_observations_to_csv(folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, filename)
    
    df = pd.DataFrame(observations)
    df.to_csv(file_path, index=False)

folder = 'DQN/episode_observations'
test_env = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')

num_episodes = 100
observations = []
for episode in range(num_episodes):
    # Reset dell'ambiente e raccolta delle osservazioni
    observation = test_env.reset()
    add_observation(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env.step(action)
        add_observation(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level1_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env.close()

############################################
# Level 2 Training: coming soon
############################################