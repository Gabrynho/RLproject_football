import pandas as pd
import os
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.animation as animation

def load_and_process_data(filename):
    df = pd.read_csv(filename)
    data = df[['88', '89', '0', '1', '2', '3', '4', '5', '6', '7', '16', '17', '18', '19', '20', '21']]*100
    data = data.iloc[:-1]
    data.columns = ['ball_x', 'ball_y', 'gk_x', 'gk_y', 'att1_x', 'att1_y', 'cr7_x', 'cr7_y', 'att2_x', 'att_y', 'gk_opp_x', 'gk_opp_y','defender1_x','defender1_y', 'defender2_x', 'defender2_y']
    data['ball_x'] += 100
    data['ball_y'] += 42
    data['gk_x'] += 100
    data['gk_y'] += 42
    data['cr7_x'] += 100
    data['cr7_y'] += 42
    data['gk_opp_x'] += 100
    data['gk_opp_y'] += 42
    data['defender1_x'] += 100
    data['defender1_y'] += 42
    data['att1_x'] += 100
    data['att1_y'] += 42
    data['defender2_x'] += 100
    data['defender2_y'] += 42
    data['att2_x'] += 100
    data['att_y'] += 42
    return data

def create_animation(data, filename):
    # Create the pitch
    pitch = Pitch(pitch_type='custom',
                  pitch_color='grass', line_color='white',
                  stripe=True,
                  pitch_length=150, pitch_width=84,
                  axis=True, label=True, half=True) 

    fig, ax = pitch.draw()

    # Initialize the scatter plots for each player and the ball
    ball_scatter = ax.scatter([], [], color='black', edgecolors='black', zorder=12)
    cr7_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att1_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    att2_scatter = ax.scatter([], [], color='red', edgecolors='black', zorder=12)
    gk_opp_scatter = ax.scatter([], [], color='green', edgecolors='black', zorder=12)
    defender1_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)
    defender2_scatter = ax.scatter([], [], color='blue', edgecolors='black', zorder=12)

    # Set the size of the scatter plots
    ball_scatter.set_sizes([100])
    cr7_scatter.set_sizes([150])
    att1_scatter.set_sizes([150])
    att2_scatter.set_sizes([150])
    gk_opp_scatter.set_sizes([150])
    defender1_scatter.set_sizes([150])
    defender2_scatter.set_sizes([150])

    # Update function for the animation
    def update(frame):
        # Update the positions of each player and the ball
        ball_scatter.set_offsets([data['ball_x'][frame], data['ball_y'][frame]])
        cr7_scatter.set_offsets([data['cr7_x'][frame], data['cr7_y'][frame]])
        gk_opp_scatter.set_offsets([data['gk_opp_x'][frame], data['gk_opp_y'][frame]])
        defender1_scatter.set_offsets([data['defender_x'][frame], data['defender_y'][frame]])
        defender2_scatter.set_offsets([data['defender_x'][frame], data['defender_y'][frame]])
        att1_scatter.set_offsets([data['att_x'][frame], data['att_y'][frame]])
        att2_scatter.set_offsets([data['att_x'][frame], data['att_y'][frame]])

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=100)

    # Save the animation as a video
    ani.save(filename, writer='ffmpeg')

# Get a list of all .csv files in the directory that match the specific pattern
csv_files = [f for f in os.listdir('./DQN/episode_observations') if os.path.isfile(f) and f.startswith('level4_episode') and f.endswith('_observations.csv')]

for file in csv_files:
    # Check if the part between 'level4_episode' and '_observations.csv' is a number
    episode_number = file[len('level4_episode'):-len('_observations.csv')]
    if episode_number.isdigit():
        # Load and process the data
        data = load_and_process_data(file)
        
        # Create a video for each file
        video_filename = file.replace('.csv', '.mp4')
        create_animation(data, video_filename)