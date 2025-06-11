# train.py
from env import WarehouseEnv
from agent import DQNAgent
import numpy as np
import torch
from utils import plot_rewards, save_model
import os

def train_agent(episodes=500, batch_size=64, target_update=10):
    env = WarehouseEnv()
    agent = DQNAgent(state_dim=3, action_dim=4)

    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
            state = next_state
            total_reward += reward

        if ep % target_update == 0:
            agent.update_target()

        print(f"Episode {ep} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        all_rewards.append(total_reward)

    save_model(agent.q_net, "models/dqn_robot.pth")
    plot_rewards(all_rewards, "outputs/reward_curve.png")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    train_agent()