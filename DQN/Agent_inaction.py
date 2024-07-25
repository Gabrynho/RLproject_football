import torch as th
import gfootball.env as football_env
from trainAgent import CR7

# Now I want to see the agent play the game 10 times
env = football_env.create_environment(env_name="gm_level1", representation='pixels', render=True)
CR7.q_eval.eval()
CR7.q_eval.to(CR7.q_eval.device)

for i in range(10):
    done = False
    observation = env.reset()
    score = 0
    steps = 0
    while not done:
        # Get action from the agent
        action = CR7.choose_action(observation)
        
        # Take the action in the environment
        observation, reward, done, info = env.step(action)
        
        # Update the score
        score += reward
        steps += 1
        
    print(f"Game {i+1}: Score = {score}, Steps = {steps}")
env.close()