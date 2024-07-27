import DQNclass as DQN
import torch as th
import pandas as pd
import os
import gfootball.env as football_env

############################################
# Utility functions
############################################   

def save_observations_to_csv(folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, filename)
    
    df = pd.DataFrame(observations)
    df.to_csv(file_path, index=False)

def convert_scores_to_csv(scst, level):
    df1 = pd.DataFrame(scst, columns=['Score per Step'])
    df1.to_csv(f'DQN/level{level}_score_per_step.csv', index=False)

############################################
# Inizialization
############################################

folder = 'DQN/episode_observations'

# Create the environment of reference
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

def train_agent(level, agent, num_episodes):

    print("############################################")
    print(f"Training on level {level}")
    scst = []

    env = football_env.create_environment(env_name=f"gm_level{level}", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')

    # Training loops
    for i in range(num_episodes): # number of episodes
        done = False
        observation = env.reset()
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation
            steps += 1
            if score > 1 and done==True:
                print("Goal!")
        scst.append(score/steps)
        print(f'Episode {i}, Score: {score}, Steps: {steps}')
        if i % 10 == 0:
            agent.save_model('DQN/CR7_model.pth')
    env.close()
    agent.save_model('DQN/CR7_model.pth')

    print(f"Training on level {level} completed")
    
    return scst

def test_agent(level, agent, num_test):

    print("############################################")
    print(f"Testing on level {level}")
    observations = []

    env = football_env.create_environment(env_name=f"gm_level{level}", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
    
    # Test loop
    for episode in range(num_test):
        # Reset the environment and collect observations
        observation = env.reset()
        observations.append(observation)
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            # Update the score
            score += reward
            steps += 1
            if score > 1 and done == True:
                print("Goal!")
                save_observations_to_csv(folder, f'level1_episode_{episode+1}_observations.csv')
        print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
        observations = []
    
    env.close()
    print(f"Testing on level {level} completed")
############################################
# Level 1 Training
############################################

# Train
scst1 = train_agent(1, CR7, num_episodes)

convert_scores_to_csv(scst1, 1)

############################################

# Test 
test_agent(1, CR7, num_test)

############################################
# Level 2 Training
############################################

# Train
scst2 = train_agent(2, CR7, num_episodes)

convert_scores_to_csv(scst2, 2)

############################################

# Test 
test_agent(2, CR7, num_test)

############################################
# Level 3 Training
############################################

# Train
scst3 = train_agent(3, CR7, num_episodes)

convert_scores_to_csv(scst3, 3)

############################################

# Test 
test_agent(3, CR7, num_test)

############################################
# Level 4 Training
############################################

# Train
scst4 = train_agent(4, CR7, num_episodes)

convert_scores_to_csv(scst4) , 4

############################################

# Test 
test_agent(4, CR7, num_test)

############################################
# Level 5 Training
############################################

# Number of episodes
num_episodes = 100
num_test = 10

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
    observations.append(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        action = CR7.choose_action(observation)
        observation, reward, done, info = test_env5.step(action)
        observations.append(observation)
        # Update the score
        score += reward
        steps += 1
        if score > 1 & done==True:
            print("Goal!")
            save_observations_to_csv(folder, f'level5_episode_{episode+1}_observations.csv')
    print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
    observations = []
test_env5.close()
print("Testing on level 5 completed")
print("############################################")