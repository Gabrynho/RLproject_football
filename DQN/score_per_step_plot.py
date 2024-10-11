import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 50

# List of levels
levels = [0, 1, 2, 3, 4]

# Initialize lists to store the data
score_per_step_data = []
total_reward_data = []

# Read Score per Step and Total Reward data for levels 0 to 4
for level in levels:
    # Read Score per Step data
    score_per_step_file = f'DQN/level{level}_score_per_step_DL.csv'
    df_sps = pd.read_csv(score_per_step_file)
    score_per_step_data.append(df_sps['Score per Step'])

    # Read Total Reward data
    total_reward_file = f'DQN/level{level}_total_rewards_DL.csv'
    df_tr = pd.read_csv(total_reward_file)
    total_reward_data.append(df_tr['Total Reward'])

# Read Difference Goal data for level 5
# df_dg = pd.read_csv('DQN/level5_diff_goal.csv')
# difference_goal_data = df_dg['Difference Goal']

# Create a figure for Score per Step
fig_sps, axs_sps = plt.subplots(3, 2, figsize=(14, 10))
axs_sps = axs_sps.flatten()

# Plot Score per Step for levels 0 to 4
for i, d in enumerate(score_per_step_data):
    x = np.arange(len(d))
    y = d
    axs_sps[i].plot(x, y, label=f'Level {i} Score per Step')
    axs_sps[i].set_xlabel('Episode')
    axs_sps[i].set_ylabel('Score per Step')
    axs_sps[i].set_title(f'Level {i} Score per Step vs Episode')
    axs_sps[i].legend()

    # Calculate and plot moving average
    
    if len(y) >= WINDOW_SIZE:
        moving_avg = np.convolve(y, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
        axs_sps[i].plot(x[WINDOW_SIZE-1:], moving_avg, color='red', linestyle='--', label='Moving Average')
        axs_sps[i].legend()

# Plot Difference Goal for level 5
# x_dg = np.arange(len(difference_goal_data))
# y_dg = difference_goal_data
# axs_sps[5].plot(x_dg, y_dg, label='Level 5 Difference Goal')
# axs_sps[5].set_xlabel('Episode')
# axs_sps[5].set_ylabel('Difference Goal')
# axs_sps[5].set_title('Level 5 Difference Goal vs Episode')
# axs_sps[5].legend()

# # Calculate and plot moving average for Difference Goal
# WINDOW_SIZE_dg = 5
# if len(y_dg) >= WINDOW_SIZE_dg:
#     moving_avg_dg = np.convolve(y_dg, np.ones(WINDOW_SIZE_dg)/WINDOW_SIZE_dg, mode='valid')
#     axs_sps[5].plot(x_dg[WINDOW_SIZE_dg-1:], moving_avg_dg, color='red', linestyle='--', label='Moving Average')
#     axs_sps[5].legend()

# plt.tight_layout()
# plt.show()

# Create a figure for Total Reward
fig_tr, axs_tr = plt.subplots(3, 2, figsize=(14, 10))
axs_tr = axs_tr.flatten()

# Plot Total Reward for levels 0 to 4
for i, d in enumerate(total_reward_data):
    x = np.arange(len(d))
    y = d
    axs_tr[i].plot(x, y, label=f'Level {i} Total Reward')
    axs_tr[i].set_xlabel('Episode')
    axs_tr[i].set_ylabel('Total Reward')
    axs_tr[i].set_title(f'Level {i} Total Reward vs Episode')
    # axs_tr[i].set_yscale('log')
    axs_tr[i].legend()

    # Calculate and plot moving average
    
    if len(y) >= WINDOW_SIZE:
        moving_avg = np.convolve(y, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
        axs_tr[i].plot(x[WINDOW_SIZE-1:], moving_avg, color='red', linestyle='--', label='Moving Average')
        axs_tr[i].legend()

# Remove the unused subplot (since we have only 5 plots)
fig_tr.delaxes(axs_tr[5])

plt.tight_layout()
plt.show()

# Optionally, save the plots
# fig_sps.savefig('DQN/score_per_step_plot.png')
# fig_tr.savefig('DQN/total_reward_plot.png')
