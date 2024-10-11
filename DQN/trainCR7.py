import DQNclass as DQN
import torch as th
import pandas as pd
import os
import time
import gfootball.env as football_env

############################################
# Utility functions
############################################

def save_observations_to_csv(folder, filename, observations):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, filename)
    
    df = pd.DataFrame(observations)
    df.to_csv(file_path, index=False)

def convert_to_csv(scst, level):
    df1 = pd.DataFrame(scst, columns=['Score per Step'])
    df1.to_csv(f'DQN/level{level}_score_per_step_DL.csv', index=False)

############################################
# Hyperparameters
############################################

LR = 1e-5                  # learning rate
GAMMA = 0.999               # discount factor
BATCH_SIZE = 128           # batch size

EPSILON = 0.9              # starting value of epsilon
EPSILON_DECAY = 1e-5       # decay rate of epsilon
EPSILON_MIN = 0.01          # minimum value of epsilon

MAX_SIZE = 5000           # max size of the replay buffer

FC1_DIMS = 256              # number of neurons in the first layer
FC2_DIMS = 128              # number of neurons in the second layer
FC3_DIMS = 64               # number of neurons in the third layer

TARGET_UPDATE_FREQ = 2500  # frequency to update target network

############################################
# Initialization
############################################

folder = 'DQN/episode_observations_DL'

# Create the environment of reference
env = football_env.create_environment(env_name="gm_level1", representation='simple115', stacked=False, render=False, rewards='scoring')

# Initialize the DQN agent
CR7 = DQN.Agent(
    input_dims=env.observation_space.shape,
    n_actions=env.action_space.n,
    lr=LR,
    gamma=GAMMA,
    batch_size=BATCH_SIZE,
    epsilon_max=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    max_size=MAX_SIZE,
    fc1_dims=FC1_DIMS,
    fc2_dims=FC2_DIMS,
    fc3_dims=FC3_DIMS,
    target_update_freq=TARGET_UPDATE_FREQ
)

# Check if the agent has been initialized with cuda or mps
print(f"Device used: {CR7.q_eval.device}")

# Load the model if it exists
model_path = 'DQN/CR7_2.0_model.pth'
if os.path.exists(model_path):
    CR7.load_model(model_path)
    print("Model loaded successfully.")

# Number of episodes
num_episodes = 1000
num_test = 100

def train_agent(level, agent, num_episodes):
    print("############################################")
    print(f"Training on level {level}")
    scst = []
    total_rewards = []
    total_reward = 0

    env = football_env.create_environment(env_name=f"gm_level{level}", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')

    # Training loops
    start_time = time.time()
    for i in range(num_episodes):  # number of episodes
        done = False
        observation = env.reset()
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            total_reward += reward

            # Store the transition
            agent.store_transition(observation, action, reward, next_observation, done)

            # Learn
            agent.learn()
            observation = next_observation
            steps += 1
            if score > 1.1 and done:
                print("Goal!")
        scst.append(score / steps if steps > 0 else 0)
        total_rewards.append(total_reward)
        print(f'Level {level} Episode {i+1}: Score = {score}, Steps = {steps}')
        if i % 50 == 0:
            agent.save_model(model_path)

    agent.exploration.reset()
    env.close()

    agent.save_model(model_path)
    print("Final model saved.")

    df_rewards = pd.DataFrame(total_rewards, columns=['Total Reward'])
    df_rewards.to_csv(f'DQN/level{level}_total_rewards_DL.csv', index=False)



    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Training on level {level} completed")
    print(f"Time: {computation_time} seconds")

    return scst

def test_agent(level, agent, num_test):
    print("############################################")
    print(f"Testing on level {level}")
    observations = []

    env = football_env.create_environment(env_name=f"gm_level{level}", representation='simple115', stacked=False, rewards='scoring,checkpoints')
    start_time = time.time()

    # Set epsilon to minimum value for testing (mostly exploitation)
    agent.exploration.epsilon = agent.exploration.epsilon_min

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
            if score > 1.1 and done:
                print("Goal!")
                save_observations_to_csv(folder, f'level{level}_episode_{episode+1}_observations.csv', observations)
        print(f"Level {level} Game {episode+1}: Score = {score}, Steps = {steps}")
        observations = []
    env.close()

    end_time = time.time()
    print(f"Testing on level {level} completed")
    print(f"Time: {end_time - start_time} seconds")

############################################
# Level 0 Training: Forward vs Empty Goal
############################################

# Train
scst0 = train_agent(0, CR7, num_episodes)
convert_to_csv(scst0, 0)

# Test
test_agent(0, CR7, num_test)

############################################
# Level 1 Training: Forward vs Goalkeeper
############################################

# Train
scst1 = train_agent(1, CR7, num_episodes)
convert_to_csv(scst1, 1)

# Test
test_agent(1, CR7, num_test)

############################################
# Level 2 Training: Forward vs Defender (and GK)
############################################

# Train
scst2 = train_agent(2, CR7, num_episodes)
convert_to_csv(scst2, 2)

# Test
test_agent(2, CR7, num_test)

############################################
# Level 3 Training: 2 Forward vs Defender (and GK)
############################################

# Train
scst3 = train_agent(3, CR7, num_episodes)
convert_to_csv(scst3, 3)

# Test
test_agent(3, CR7, num_test)

############################################
# Level 4 Training: 3 Forward vs 2 Defender (and GK)
############################################

# Train
scst4 = train_agent(4, CR7, num_episodes)
convert_to_csv(scst4, 4)

# Test
test_agent(4, CR7, num_test)

############################################
# Level 5 Training: 7vs7
############################################

# Number of episodes
# num_match = 75
# num_test = 10

# # Initialize the environment
# level5_env = football_env.create_environment(env_name="gm_level5", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
# print("############################################")
# print("Training on level 5")
# start_time = time.time()
# diff_goal = []

# # Training loops
# for i in range(num_match):  # number of episodes
#     done = False
#     observation = level5_env.reset()
#     score = 0
#     steps = 0
#     rlagent_goal = 0
#     computer_goal = 0
#     while not done:
#         action = CR7.choose_action(observation)
#         next_observation, reward, done, info = level5_env.step(action)
#         score += reward
#         CR7.store_transition(observation, action, reward, next_observation, done)
#         CR7.learn()
#         observation = next_observation
#         steps += 1
#         if reward > 0.995:
#             rlagent_goal += 1
#             print("Goal!")
#         if reward < -0.995:
#             computer_goal += 1
#             print("Goal for the computer!")
#     print(f"Training Match {i+1} RL Agent - Computer: {rlagent_goal}-{computer_goal}")
#     diff_goal.append(rlagent_goal - computer_goal)
#     if i % 5 == 0:
#         CR7.save_model(model_path)
#         print("Model saved.")
# level5_env.close()
# CR7.save_model(model_path)
# print("Final model saved.")

# end_time = time.time()
# df1 = pd.DataFrame(diff_goal, columns=['Difference Goal'])
# df1.to_csv(f'DQN/level5_diff_goal.csv', index=False)
# print("Training on level 5 completed")
# print(f"Time: {end_time - start_time} seconds")

# ############################################
# # Test the agent and see how it acts
# test_env5 = football_env.create_environment(env_name="gm_level5", representation='simple115', stacked=False, rewards='scoring,checkpoints')
# print("############################################")
# print("Testing on level 5")
# start_time = time.time()
# diff_goal = []
# observations = []

# # Set epsilon to minimum value for testing
# CR7.exploration.epsilon = CR7.exploration.epsilon_min

# # Test loops
# for episode in range(num_test):
#     # Reset the environment and collect observations
#     observation = test_env5.reset()
#     observations.append(observation)
#     done = False
#     score = 0
#     rlagent_goal = 0
#     computer_goal = 0
#     while not done:
#         action = CR7.choose_action(observation)
#         observation, reward, done, info = test_env5.step(action)
#         observations.append(observation)
#         # Update the score
#         score += reward
#         if reward > 0.995:
#             rlagent_goal += 1
#             print("Goal!")
#         if reward < -0.995:
#             computer_goal += 1
#             print("Goal for the computer!")
#     print(f"Testing Match {episode+1} RL Agent - Computer: {rlagent_goal}-{computer_goal}")
#     diff_goal.append(rlagent_goal - computer_goal)
#     save_observations_to_csv(folder, f'level5_episode_{episode+1}_observations.csv', observations)
#     observations = []
# test_env5.close()
# df2 = pd.DataFrame(diff_goal, columns=['Difference Goal'])
# df2.to_csv(f'DQN/level5_test_diff_goal.csv', index=False)
# print("Testing on level 5 completed")
# end_time = time.time()
# print(f"Time: {end_time - start_time} seconds")
# print("############################################")

# print("Training completed!!!")

# print("############################################")