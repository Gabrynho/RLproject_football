import pandas as pd
import os
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.animation as animation
import re

def load_and_process_data(filename):
    df = pd.read_csv(f'DQN/episode_observations_DL/{filename}')
    data = df[['88', '89', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41']]*100
    data = data.iloc[:-1]
    data.columns = ['ball_x', 'ball_y', 'gk_x', 'gk_y', 'def1_x', 'def1_y', 'def2_x', 'def2_y', 'def3_x', 'def3_y', 'def4_x', 'def4_y', 'def5_x', 'def5_y', 'def6_x', 'def6_y', 'gk_opp_x', 'gk_opp_y', 'att1_x', 'att1_y', 'att2_x', 'att2_y', 'att3_x', 'att3_y', 'att4_x', 'att4_y', 'att5_x', 'att5_y', 'att6_x', 'att6_y']
    data['ball_x'] += 100
    data['ball_y'] += 42
    data['gk_x'] += 100
    data['gk_y'] += 42
    data['def1_x'] += 100
    data['def1_y'] += 42
    data['def2_x'] += 100
    data['def2_y'] += 42
    data['def3_x'] += 100
    data['def3_y'] += 42
    data['def4_x'] += 100
    data['def4_y'] += 42
    data['def5_x'] += 100
    data['def5_y'] += 42
    data['def6_x'] += 100
    data['def6_y'] += 42
    data['gk_opp_x'] += 100
    data['gk_opp_y'] += 42
    data['att1_x'] += 100
    data['att1_y'] += 42
    data['att2_x'] += 100
    data['att2_y'] += 42
    data['att3_x'] += 100
    data['att3_y'] += 42
    data['att4_x'] += 100
    data['att4_y'] += 42
    data['att5_x'] += 100
    data['att5_y'] += 42
    data['att6_x'] += 100
    data['att6_y'] += 42
    return data

def create_animation(data, filename):
    # Create the pitch
    pitch = Pitch(pitch_type='custom',
                    pitch_color='grass', line_color='white',
                    stripe=True,
                    pitch_length=200, pitch_width=84,
                    axis=True, label=True, half=False)

    fig, ax = pitch.draw()

    # Initialize the scatter plots for each player and the ball
    ball_scatter = ax.scatter([], [], color='black', edgecolors='black', zorder=12)
    gk_scatter = ax.scatter([], [], color='green', edgecolors='black', zorder=12)
    def1_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    def2_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    def3_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    def4_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    def5_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    def6_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    gk_opp_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att1_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att2_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att3_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att4_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att5_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att6_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)

    # Set the size of the scatter plots
    ball_scatter.set_sizes([100])
    gk_scatter.set_sizes([150])
    def1_scatter.set_sizes([150])
    def2_scatter.set_sizes([150])
    def3_scatter.set_sizes([150])
    def4_scatter.set_sizes([150])
    def5_scatter.set_sizes([150])
    def6_scatter.set_sizes([150])
    gk_opp_scatter.set_sizes([150])
    att1_scatter.set_sizes([150])
    att2_scatter.set_sizes([150])
    att3_scatter.set_sizes([150])
    att4_scatter.set_sizes([150])
    att5_scatter.set_sizes([150])
    att6_scatter.set_sizes([150])

    # Update function for the animation
    def update(frame):
        # Update the positions of each player and the ball
        ball_scatter.set_offsets([data['ball_x'][frame], data['ball_y'][frame]])
        gk_scatter.set_offsets([data['gk_x'][frame], data['gk_y'][frame]])
        def1_scatter.set_offsets([data['def1_x'][frame], data['def1_y'][frame]])
        def2_scatter.set_offsets([data['def2_x'][frame], data['def2_y'][frame]])
        def3_scatter.set_offsets([data['def3_x'][frame], data['def3_y'][frame]])
        def4_scatter.set_offsets([data['def4_x'][frame], data['def4_y'][frame]])
        def5_scatter.set_offsets([data['def5_x'][frame], data['def5_y'][frame]])
        def6_scatter.set_offsets([data['def6_x'][frame], data['def6_y'][frame]])
        gk_opp_scatter.set_offsets([data['gk_opp_x'][frame], data['gk_opp_y'][frame]])
        att1_scatter.set_offsets([data['att1_x'][frame], data['att1_y'][frame]])
        att2_scatter.set_offsets([data['att2_x'][frame], data['att2_y'][frame]])
        att3_scatter.set_offsets([data['att3_x'][frame], data['att3_y'][frame]])
        att4_scatter.set_offsets([data['att4_x'][frame], data['att4_y'][frame]])
        att5_scatter.set_offsets([data['att5_x'][frame], data['att5_y'][frame]])
        att6_scatter.set_offsets([data['att6_x'][frame], data['att6_y'][frame]])

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=100)

    # Save the animation as a video
    ani.save(f'DQN/episode_observations_DL/{filename}', writer='ffmpeg')

    # Close the plot
    plt.close(fig)

# Get a list of all .csv files in the directory that begins with level{level} and ends with .csv
files = [f for f in os.listdir('./DQN/episode_observations_DL') if re.match(r'level5.*\.csv', f)]
for file in files:
    data = load_and_process_data(file)
    create_animation(data, file.replace('.csv', '.mp4'))