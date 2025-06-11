from env import WarehouseEnv
from agent import DQNAgent
from visualizer import WarehouseVisualizer
import torch
import time
import pygame
import sys

def test_agent(model_path="models/dqn_robot.pth", episodes=5, render=True):
    env = WarehouseEnv()
    visualizer = WarehouseVisualizer(env)
    agent = DQNAgent(state_dim=3, action_dim=4)
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.q_net.eval()

    success = 0
    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        print(f"\n=== Episode {ep+1} ===\n")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    visualizer.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if render:
                visualizer.draw()

        total_rewards.append(total_reward)
        if reward >= 10:
            success += 1

        print(f"Episode {ep+1} - Reward: {total_reward:.2f}")

    print("\n===== TEST SUMMARY =====")
    print(f"Success episodes: {success}/{episodes}")
    print(f"Average reward: {sum(total_rewards)/episodes:.2f}")

    visualizer.quit()

if __name__ == "__main__":
    test_agent()
