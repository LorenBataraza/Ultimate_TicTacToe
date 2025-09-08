import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

##
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.25
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_network()
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, valid_actions=None):
        if np.random.rand() <= self.epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        
        # Filter invalid actions if provided
        if valid_actions:
            act_values = act_values.detach().numpy()[0]
            # Set very low value for invalid actions
            mask = np.ones(self.action_size) * -np.inf
            mask[valid_actions] = act_values[valid_actions]
            return np.argmax(mask)
        
        return np.argmax(act_values.detach().numpy())
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convertir listas de arrays de NumPy a arrays de NumPy únicos
        states = np.array(states) 
        next_states = np.array(next_states) 
        
        states = torch.FloatTensor(states)  
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states) 
        dones = torch.FloatTensor(dones)
        
        # Resto del código sin cambios...
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        
    def save(self, name):
        torch.save(self.model.state_dict(), name)