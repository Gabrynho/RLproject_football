import torch as th
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions=19):
        super(DeepQNetwork, self).__init__()

        self.model = nn.Sequential(
                        nn.Linear(*input_dims, fc1_dims),
                        nn.Tanh(),
                        nn.Linear(fc1_dims, fc2_dims),
                        nn.Tanh(),
                        nn.Linear(fc2_dims, fc3_dims),
                        nn.Tanh(),
                        nn.Linear(fc3_dims, n_actions),
                        nn.Softmax(dim=-1)
                        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.model(state)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return th.tensor(states).to(self.device), th.tensor(actions).to(self.device), th.tensor(rewards).to(self.device), th.tensor(next_states).to(self.device), th.tensor(dones).to(self.device)

class ExplorationMethod:
    def __init__(self, epsilon, epsilon_decay, epsilon_min):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.steps = 0

    def get_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return self.epsilon

    def update_steps(self, steps):
        self.steps += steps
        return self.steps

    def reset(self):
        self.steps = 0

class Agent:
    def __init__(self, input_dims, n_actions, lr, gamma, batch_size, epsilon, epsilon_decay, epsilon_min, max_size, fc1_dims, fc2_dims, fc3_dims):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.max_size = max_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims)
        self.memory = ReplayBuffer(max_size)
        self.exploration = ExplorationMethod(epsilon, epsilon_decay, epsilon_min)

    def choose_action(self, observation):
        if random.random() > self.exploration.get_epsilon(): # exploitation
            state = th.tensor(np.array([observation])).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = th.argmax(actions).item()
        else: # exploration
            action = random.choice([i for i in range(self.n_actions)])
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory.buffer) < self.batch_size: # if buffer is not full
            return # do not learn

        self.q_eval.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # sample from buffer
        indices = th.arange(self.batch_size) # indices for actions

        q_pred = self.q_eval.forward(states)[indices, actions] # predicted q values
        q_next = self.q_eval.forward(next_states).max(dim=1)[0] # max q values for next states

        q_next[dones] = 0.0 # if done, q value is 0

        q_target = rewards + self.gamma * q_next # target q value

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device) # loss
        loss.backward() # backpropagation
        self.q_eval.optimizer.step() # update weights

        self.exploration.update_steps(self.batch_size) # update steps

    def save_model(self, filename):
        th.save(self.q_eval.state_dict(), filename)

    def load_model(self, filename):
        self.q_eval.load_state_dict(th.load(filename))
        self.q_eval.eval()