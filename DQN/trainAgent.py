import DQNclass as DQN
import torch as th
import pandas as pd
import os
import gfootball.env as football_env

############################################
# Utility functions
############################################

def add_observation(observation):
    observations.append(observation)

def save_observations_to_csv(folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, filename)
    
    df = pd.DataFrame(observations)
    df.to_csv(file_path, index=False)

############################################
# Inizialization
############################################

folder = 'DQN/episode_observations'

# Create the environment
env = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')

# Initialize the DQN agent
CR7 = DQN.Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=.00001, gamma=0.99, batch_size=256, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.01, max_size=10000, fc1_dims=128, fc2_dims=128, fc3_dims=128)

# Load the template if it exists
if os.path.exists('DQN/CR7_model.pth'):
    CR7.q_eval.load_state_dict(th.load('DQN/CR7_model.pth'))
    CR7.q_eval.eval()
    CR7.q_eval.to(CR7.q_eval.device)

# Number of episodes
num_episodes = 1000
num_test = 100
############################################
# Level 1 Training
############################################

# Initialize the environment
level1_env = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Training on level 1")
scst1 = []

# Training loops
for i in range(num_episodes): # number of episodes
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
        if score > 1 & done==True:
            print("Goal!")
    scst1.append(score/steps)
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('DQN/CR7_model.pth')
level1_env.close()
CR7.save_model('DQN/CR7_model.pth')
df1 = pd.DataFrame(scst1, columns=['Score per Step'])
df1.to_csv('DQN/level1_score_per_step.csv', index=False)
print("Training on level 1 completed")
print("############################################")

############################################

# Test the agent and see how it acts
test_env1 = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Testing on level 1")
observations = []

# Test loop
for episode in range(num_test):
    # Reset dell'ambiente e raccolta delle osservazioni
    observation = test_env1.reset()
    add_observation(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env1.step(action)
        add_observation(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level1_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env1.close()
print("Testing on level 1 completed")
print("############################################")

############################################
# Level 2 Training
############################################

# Initialize the environment
level2_env = football_env.create_environment(env_name="gm_level2", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Training on level 2")
scst2 = []

# Training loops
for i in range(num_episodes): # number of episodes
    done = False
    observation = level2_env.reset()
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        next_observation, reward, done, info = level2_env.step(action)
        score += reward
        CR7.store_transition(observation, action, reward, next_observation, done)
        CR7.learn()
        observation = next_observation
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
    scst2.append(score/steps)
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('DQN/CR7_model.pth')
level2_env.close()
CR7.save_model('DQN/CR7_model.pth')
df2 = pd.DataFrame(scst2, columns=['Score per Step'])
df2.to_csv('DQN/level2_score_per_step.csv', index=False)
print("Training on level 2 completed")
print("############################################")

############################################

# Test the agent and see how it acts
test_env2 = football_env.create_environment(env_name="gm_level2", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Testing on level 2")
observations = []

# Test loops
for episode in range(num_test):
    # Reset dell'ambiente e raccolta delle osservazioni
    observation = test_env2.reset()
    add_observation(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env2.step(action)
        add_observation(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level2_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env2.close()
print("Testing on level 2 completed")
print("############################################")

############################################
# Level 3 Training
############################################

# Initialize the environment
level3_env = football_env.create_environment(env_name="gm_level3", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Training on level 3")
scst3 = []

# Training loops
for i in range(num_episodes): # number of episodes
    done = False
    observation = level3_env.reset()
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        next_observation, reward, done, info = level3_env.step(action)
        score += reward
        CR7.store_transition(observation, action, reward, next_observation, done)
        CR7.learn()
        observation = next_observation
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
    scst3.append(score/steps)
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('DQN/CR7_model.pth')
level3_env.close()
CR7.save_model('DQN/CR7_model.pth')
df3 = pd.DataFrame(scst3, columns=['Score per Step'])
df3.to_csv('DQN/level3_score_per_step.csv', index=False)
print("Training on level 3 completed")
print("############################################")

############################################

# Test the agent and see how it acts
test_env3 = football_env.create_environment(env_name="gm_level3", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Testing on level 3")
observations = []

# Test loops
for episode in range(num_test):
    # Reset dell'ambiente e raccolta delle osservazioni
    observation = test_env3.reset()
    add_observation(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env3.step(action)
        add_observation(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level3_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env3.close()
print("Testing on level 3 completed")
print("############################################")

############################################
# Level 4 Training
############################################

# Initialize the environment
level4_env = football_env.create_environment(env_name="gm_level4", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Training on level 4")
scst4 = []

# Training loops
for i in range(num_episodes): # number of episodes
    done = False
    observation = level4_env.reset()
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        next_observation, reward, done, info = level4_env.step(action)
        score += reward
        CR7.store_transition(observation, action, reward, next_observation, done)
        CR7.learn()
        observation = next_observation
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
    scst4.append(score/steps)
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('DQN/CR7_model.pth')
level4_env.close()
CR7.save_model('DQN/CR7_model.pth')
df4 = pd.DataFrame(scst4, columns=['Score per Step'])
df4.to_csv('DQN/level4_score_per_step.csv', index=False)
print("Training on level 4 completed")
print("############################################")

############################################

# Test the agent and see how it acts
test_env4 = football_env.create_environment(env_name="gm_level4", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Testing on level 4")
observations = []

# Test loops
for episode in range(num_test):
    # Reset dell'ambiente e raccolta delle osservazioni
    observation = test_env4.reset()
    add_observation(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env4.step(action)
        add_observation(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level4_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env4.close()
print("Testing on level 4 completed")
print("############################################")

############################################
# Level 5 Training
############################################

# Initialize the environment
level5_env = football_env.create_environment(env_name="gm_level5", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Training on level 5")
scst5 = []

# Training loops
for i in range(num_episodes): # number of episodes
    done = False
    observation = level5_env.reset()
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        next_observation, reward, done, info = level5_env.step(action)
        score += reward
        CR7.store_transition(observation, action, reward, next_observation, done)
        CR7.learn()
        observation = next_observation
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
    scst5.append(score/steps)
    print(f'Episode {i}, Score: {score}, Steps: {steps}')
    if i % 10 == 0:
        CR7.save_model('DQN/CR7_model.pth')
level5_env.close()
CR7.save_model('DQN/CR7_model.pth')
df5 = pd.DataFrame(scst5, columns=['Score per Step'])
df5.to_csv('DQN/level5_score_per_step.csv', index=False)
print("Training on level 5 completed")
print("############################################")

############################################

# Test the agent and see how it acts
test_env5 = football_env.create_environment(env_name="gm_level5", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("Testing on level 5")
observations = []

# Test loops
for episode in range(num_test):
    # Reset dell'ambiente e raccolta delle osservazioni
    observation = test_env5.reset()
    add_observation(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env5.step(action)
        add_observation(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level5_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env4.close()
print("Testing on level 5 completed")
print("############################################")