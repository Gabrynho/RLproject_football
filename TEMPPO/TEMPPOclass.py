import torch as th
import torch.nn as nn
from torch.distributions import Categorical

device = th.device('cpu')

if(th.cuda.is_available()): 
    device = th.device('cuda:0') 
    th.cuda.empty_cache()

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dims, fc2_dims, fc3_dims):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, fc1_dims),
                        nn.Tanh(),
                        nn.Linear(fc1_dims, fc2_dims),
                        nn.Tanh(),
                        nn.Linear(fc2_dims, fc3_dims),
                        nn.Tanh(),
                        nn.Linear(fc3_dims, action_dim),
                        nn.Softmax(dim=-1)
                        )

        # critic1
        self.critic1 = nn.Sequential(
                        nn.Linear(state_dim, fc1_dims),
                        nn.Tanh(),
                        nn.Linear(fc1_dims, fc2_dims),
                        nn.Tanh(),
                        nn.Linear(fc2_dims, fc3_dims),
                        nn.Tanh(),
                        nn.Linear(fc3_dims, 1)
                    )
        
        # critic2
        self.critic2 = nn.Sequential(
                        nn.Linear(state_dim, fc1_dims),
                        nn.Tanh(),
                        nn.Linear(fc1_dims, fc2_dims),
                        nn.Tanh(),
                        nn.Linear(fc2_dims, fc3_dims),
                        nn.Tanh(),
                        nn.Linear(fc3_dims, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values1 = self.critic1(state) ### my change
        state_values2 = self.critic2(state)
        state_values = th.min(state_values1, state_values2)

        return action_logprobs, state_values, dist_entropy
    

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, fc1_dims, fc2_dims, fc3_dims):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, fc1_dims, fc2_dims, fc3_dims).to(device)
        self.optimizer = th.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic1.parameters(), 'lr': lr_critic},
                        {'params': self.policy.critic2.parameters(), 'lr': lr_critic}
                    ])
        
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        self.policy_old = ActorCritic(state_dim, action_dim, fc1_dims, fc2_dims, fc3_dims).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with th.no_grad():
            state = th.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()


    def update(self, tesla):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = th.tensor(rewards, dtype=th.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = th.squeeze(th.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = th.squeeze(th.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = th.squeeze(th.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = th.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = th.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages1 = rewards - state_values.detach() 
            advantages2 = rewards - state_values.detach() - 0.01*dist_entropy 

            surr1 = ratios * advantages1
            surr2 = th.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages1

            # final loss of clipped objective PPO
            loss1 = -th.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            surr1 = ratios * advantages2
            surr2 = th.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages2

            # final loss of clipped objective PPO
            loss2 = -th.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards)
            
            loss = tesla * loss2 + (1 - tesla) * loss1
            # print("this is loss", loss)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
    
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save_model(self, filename):
        th.save(self.policy_old.state_dict(), filename)
   

    def load_model(self, filename):
        self.policy_old.load_state_dict(th.load(filename))
        self.policy.load_state_dict(th.load(filename))