import numpy as np
import sys
import os

from dqn_model import DQNAgent
from UT3_env import *

def preprocess_state(state_dict):
    # Convert state dictionary to a flat array
    board = state_dict["board"]
    active_board = state_dict["active_board"]
    current_player = state_dict["current_player"]
    
    # One-hot encode active board
    active_encoded = np.zeros(9)
    if active_board[0] != -1 and active_board[1] != -1:
        active_idx = active_board[0] * 3 + active_board[1]
        active_encoded[active_idx] = 1
    # Combine all features
    return np.concatenate([board, active_encoded, [current_player]])


def train_dqn(episodes=1000, batch_size=64, model_path=None):
    env = UT3_Env()
    state_size = 81 + 9 + 1  # board + active_encoded + current_player
    action_size = 81
    agent = DQNAgent(state_size, action_size)
    
    if model_path is not None:
        agent.load(model_path)

    scores = []
    
    for e in range(episodes):
        state, info = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            valid_actions = info["valid_moves"]
            action = agent.act(state, valid_actions)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated or (valid_actions==[])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        scores.append(total_reward)
        
        # Update target network every 10 episodes
        if e % 10 == 0:
            agent.update_target_network()
            
        print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Save model every 100 episodes
        if e % 100 == 0:
            agent.save(f"models/dqn_ut3_{e}.pth")

    return scores

if __name__ == "__main__":
    scores = train_dqn()
