import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of file names
file_names = ['DQN/level0_score_per_step.csv','DQN/level1_score_per_step.csv', 'DQN/level2_score_per_step.csv', 'DQN/level3_score_per_step.csv', 'DQN/level4_score_per_step.csv']

# Initialize an empty list to store the data
data = []

# Read the data from each file and append it to the list
for file_name in file_names:
    df = pd.read_csv(file_name)
    data.append(df['Score per Step'])

df1 = pd.read_csv('DQN/level5_diff_goal.csv')
data.append(df1['Difference Goal'])

fig, axs = plt.subplots(3, 2, figsize=(14, 8))
axs = axs.flatten()

for i, d in enumerate(data):
    if i == 5:
        axs[i].plot(d, label='Difference Goal')
        axs[i].set_xlabel('Episode')
        axs[i].set_ylabel('Difference Goal')
        axs[i].set_title('Difference Goal vs Episode (Level 5)')
        axs[i].legend()
    else:
        axs[i].plot(d, label=f'Level {i}')
        axs[i].set_xlabel('Episode')
        axs[i].set_ylabel('Score per Step')
        axs[i].set_title(f'Score per Step vs Episode (Level {i})')
        axs[i].legend()

    x = np.arange(len(d))
    y = d

    best_degree = None
    min_chi2 = float('inf')

    # Test del chi-quadro per selezionare il miglior grado del polinomio
    for degree in range(2, 16):
        coeffs = np.polyfit(x, y, degree)
        poly_eq = np.poly1d(coeffs)
        fitted_values = poly_eq(x)
        chi2_value = np.sum(((y - fitted_values) ** 2) / fitted_values)

        if chi2_value < min_chi2:
            min_chi2 = chi2_value
            best_coeffs = coeffs

    poly_eq = np.poly1d(best_coeffs)
    axs[i].plot(x, poly_eq(x), color='red', linestyle='--')
    degree = len(best_coeffs) - 1
    coeffs_str = f'Grade {degree}'
    axs[i].legend(['Agent during Training', f'Fit: Degree {degree}, $\\chi^2$ = {min_chi2:.2f}'])

#plt.tight_layout()
plt.show()

# Save the plot
#plt.savefig('DQN/score_per_step_plot.png')