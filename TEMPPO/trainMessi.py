import TEMPPOclass as TEMPPO
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
    df1.to_csv(f'DQN/level{level}_score_per_step.csv', index=False)

############################################
# Hyperparameters
############################################

K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO (from the paper)
gamma = 0.99                # discount factor
tesla = 0.0005              # entropy factor 

lr_actor = 0.0003           # learning rate for actor network
lr_critic = 0.001           # learning rate for critic network

fc1_dims=128                # number of neurons in the first layer
fc2_dims=128                # number of neurons in the second layer      
fc3_dims=128                # number of neurons in the third layer

############################################
# Inizialization
############################################

folder = 'TEMPPO/episode_observations'

env = football_env.create_environment(env_name='gm_level1', representation='simple115')

# initialize a PPO agent
Messi = TEMPPO.PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                       lr_actor=lr_actor, lr_critic=lr_critic,
                       gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
                       fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims)

# Check if the agent has been initialized with cuda
print(Messi.device)

if os.path.exists('TEMPPO/Messi_model.pth'):
    Messi.load_model('TEMPPO/Messi_model.pth')

num_episodes = 1000
num_test = 100

def train_agent(level, agent, num_episodes):

    print("############################################")
    print(f"Training on level {level}")
    scst = []

    env = football_env.create_environment(env_name=f"gm_level{level}", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')

    start_time = time.time()
    # training loop
    for i in range(num_episodes): # number of episodes
        done = False
        state = env.reset()
        score = 0
        steps = 0
        while not done:
            # select action with policy
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            steps +=1
            score += reward

            # update PPO agent every K epochs
            if num_episodes % 5 == 0:
                agent.update(tesla)
                if tesla < 0.9:
                    tesla += 0.0005
                agent.save_model(f'TEMPPO/Messi_model.pth')
        scst.append(score/steps)
        print(f'Episode {i}, Score: {score}, Steps: {steps}')
    env.close()
    Messi.save_model('TEMPPO/Messi_model.pth')

    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Training on level {level} completed")
    print(f"Time: {computation_time} seconds")

    return scst


def test_agent(level, agent, num_test):

    print("############################################")
    print(f"Training on level {level}")
    observations = []

    env = football_env.create_environment(env_name=f"gm_level{level}", representation='simple115', stacked=False, write_goal_dumps=True, rewards='scoring,checkpoints')
    start_time = time.time()

    # Test loop
    for episode in range(num_test): # number of episodes
        done = False
        observation = env.reset()
        observations.append(observation)
        score = 0
        steps = 0
        while not done:
            # select action with policy
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            observations.append(observation)
            
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            observation = next_observation
            
            steps +=1
            score += reward
        
        if score > 1 and done == True:
                print("Goal!")
                save_observations_to_csv(folder, f'level{level}_episode_{episode+1}_observations.csv', observations)
                env.write_dump(f'level{level}_episode_{episode+1}_observations.dmp')
        print(f"Game {episode+1}: Score = {score}, Steps = {steps}")
        observations = []
    env.close()

    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Training on level {level} completed")
    print(f"Time: {computation_time} seconds")


############################################
# Level 1 Training: Forward vs Goalkeeper
############################################

# Train
scst1 = train_agent(1, Messi, num_episodes)

convert_to_csv(scst1, 1)

############################################

# Test 
test_agent(1, Messi, num_test)

############################################
# Level 2 Training: Forward vs Defender (and GK)
############################################

# Train
scst2 = train_agent(2, Messi, num_episodes)

convert_to_csv(scst2, 2)

############################################

# Test 
test_agent(2, Messi, num_test)

############################################
# Level 3 Training: 2 Forward vs Defender (and GK)
############################################

# Train
scst3 = train_agent(3, Messi, num_episodes)

convert_to_csv(scst3, 3)

############################################

# Test 
test_agent(3, Messi, num_test)

############################################
# Level 4 Training: 3 Forward vs 2 Defender (and GK)
############################################

# Train
scst4 = train_agent(4, Messi, num_episodes)

convert_to_csv(scst4, 4)

############################################

# Test 
test_agent(4, Messi, num_test)

############################################
# Level 5 Training: 7vs7
############################################

# Number of episodes
num_match = 100
num_test = 10 # DO NOT CHANGE THIS VALUE

# Initialize the environment
level5_env = football_env.create_environment(env_name="gm_level5", representation='simple115', stacked=False, render=False, rewards='scoring,checkpoints')
print("############################################")
print("Training on level 5")

start_time = time.time()
# training loop
for i in range(num_match): # number of episodes
    done = False
    state = level5_env.reset()
    score = 0
    steps = 0
    rlagent_goal = 0
    computer_goal = 0
    diff_goal = []
    while not done:
        # select action with policy
        action = Messi.select_action(state)
        state, reward, done, info = level5_env.step(action)
        
        # saving reward and is_terminals
        Messi.buffer.rewards.append(reward)
        Messi.buffer.is_terminals.append(done)
        
        steps +=1
        score += reward

        # update PPO agent every K epochs
        if num_episodes % 5 == 0:
            Messi.update(tesla)
            if tesla < 0.9:
                tesla += 0.0005
            Messi.save_model(f'TEMPPO/Messi_model.pth')

        # check if goal
        if reward > 0.995:
            rlagent_goal += 1  
            print("Goal!")
        if reward < -0.995:
            computer_goal += 1
            print("Goal for the computer!")
    print(f"Training Match {i} RL Agent - Computer: {rlagent_goal}-{computer_goal}")
    diff_goal.append(rlagent_goal - computer_goal)

level5_env.close()
Messi.save_model('TEMPPO/Messi_model.pth')

df1 = pd.DataFrame(diff_goal, columns=['Difference Goal'])
df1.to_csv(f'DQN/level5_test_diff_goal.csv', index=False)

end_time = time.time()
computation_time = end_time - start_time
print("Training on level 5 completed")
print(f"Time: {computation_time} seconds")

test_env5 = football_env.create_environment(env_name="gm_level5", representation='simple115', stacked=False, rewards='scoring,checkpoints')
print("############################################")
print("Testing on level 5")
start_time = time.time()
diff_goal = []

for i in range(num_match): # number of episodes
    done = False
    state = test_env5.reset()
    score = 0
    rlagent_goal = 0
    computer_goal = 0
    while not done:
        # select action with policy
        action = Messi.select_action(state)
        state, reward, done, info = test_env5.step(action)
        
        # saving reward and is_terminals
        Messi.buffer.rewards.append(reward)
        Messi.buffer.is_terminals.append(done)
        
        score += reward

        # check if goal
        if reward > 0.995:
            rlagent_goal += 1  
            print("Goal!")
        if reward < -0.995:
            computer_goal += 1
            print("Goal for the computer!")
    print(f"Training Match {i} RL Agent - Computer: {rlagent_goal}-{computer_goal}")
    diff_goal.append(rlagent_goal - computer_goal)
level5_env.close()
df2 = pd.DataFrame(diff_goal, columns=['Difference Goal'])
df2.to_csv(f'DQN/level5_test_diff_goal.csv', index=False)
print("Testing on level 5 completed")
end_time = time.time()
print(f"Time: {end_time - start_time} seconds")
print("############################################")

print("Training completed!!!")

print("############################################")