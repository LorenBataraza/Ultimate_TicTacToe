""" evaluate.py """
import numpy as np
import sys
import os

from UT3_env import UTT_Env
from dqn_model import DQNAgent

def evaluate_agent(model_path, num_episodes=10):
    env = UTT_Env()
    state_size = 81 + 9 + 1
    action_size = 81
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # Disable exploration
    
    wins = 0
    losses = 0
    ties = 0
    
    for e in range(num_episodes):
        state, info = env.reset()
        state = preprocess_state(state)
        done = False
        
        while not done:
            valid_actions = info["valid_moves"]
            action = agent.act(state, valid_actions)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = preprocess_state(next_state)
            done = terminated or truncated
            
            if done:
                if reward == WIN_REWARD:
                    wins += 1
                elif reward == LOSS_REWARD:
                    losses += 1
                else:
                    ties += 1
                    
        print(f"Episode {e}: Result - {['Loss', 'Tie', 'Win'][int(reward) + 1]}")
    
    print(f"Results: Wins: {wins}, Losses: {losses}, Ties: {ties}")
    print(f"Win rate: {wins/num_episodes*100:.2f}%")
