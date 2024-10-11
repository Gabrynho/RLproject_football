import torch as th
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims, fc3_dims, dropout_rate=0.3):
        super(DeepQNetwork, self).__init__()

        self.model = nn.Sequential(
            # Encoder Part
            nn.Linear(*input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU6(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU6(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(fc2_dims, fc3_dims),
            nn.LayerNorm(fc3_dims),
            nn.ReLU6(),
            nn.Dropout(p=dropout_rate),

            # Decoder Part
            nn.Linear(fc3_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU6(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(fc2_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU6(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(fc1_dims, *input_dims),
            nn.LayerNorm(*input_dims),
            nn.ReLU6(),
            nn.Dropout(p=dropout_rate),

            # Output Layer
            nn.Linear(*input_dims, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        if th.cuda.is_available():
            self.device = th.device('cuda:0')
        elif th.backends.mps.is_available():
            self.device = th.device('mps')
        else:
            self.device = th.device('cpu')
        self.to(self.device)

        # Initialize weights
        self.init_weights()

    def forward(self, state):
        return self.model(state)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        if th.cuda.is_available():
            self.device = th.device('cuda:0')
        elif th.backends.mps.is_available():
            self.device = th.device('mps')
        else:
            self.device = th.device('cpu')

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        states = th.tensor(states, dtype=th.float32).to(self.device)
        actions = th.tensor(actions, dtype=th.long).to(self.device)
        rewards = th.tensor(rewards, dtype=th.float32).to(self.device)
        next_states = th.tensor(next_states, dtype=th.float32).to(self.device)
        dones = th.tensor(dones, dtype=th.bool).to(self.device)

        return states, actions, rewards, next_states, dones


class ExplorationMethod:
    def __init__(self, epsilon_max, epsilon_decay, epsilon_min):
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.steps = 0

    def get_epsilon(self):
        return self.epsilon

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def update_steps(self, steps):
        self.steps += steps
        return self.steps

    def reset(self):
        self.steps = 0
        self.epsilon = self.epsilon_max


class Agent:
    def __init__(self, input_dims, n_actions, lr, gamma, batch_size, epsilon_max, epsilon_decay, epsilon_min,
                 max_size, fc1_dims, fc2_dims, fc3_dims, target_update_freq):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.max_size = max_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        self.q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims)
        self.q_target = copy.deepcopy(self.q_eval)
        self.memory = ReplayBuffer(max_size)
        self.exploration = ExplorationMethod(epsilon_max, epsilon_decay, epsilon_min)

    def choose_action(self, observation):
        # No need to switch between train and eval modes
        if random.random() > self.exploration.get_epsilon():  # exploitation
            state = th.tensor(np.array([observation]), dtype=th.float32).to(self.q_eval.device)
            with th.no_grad():
                actions = self.q_eval.forward(state)
            action = th.argmax(actions).item()
        else:  # exploration
            action = random.choice([i for i in range(self.n_actions)])
        self.exploration.decay_epsilon()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:  # if buffer is not full
            return  # do not learn

        self.q_eval.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_pred = self.q_eval.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = self.q_target.forward(next_states).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss_fn(q_pred, q_target.detach()).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        self.exploration.update_steps(self.batch_size)

    def save_model(self, filename):
        th.save(self.q_eval.state_dict(), filename)

    def load_model(self, filename):
        self.q_eval.load_state_dict(th.load(filename))
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_eval.eval()
        self.q_target.eval()
