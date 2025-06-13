from env import WarehouseEnv
from agent import DQNAgent
from visualizer import WarehouseVisualizer
import torch
import time
import pygame
import sys
def test_agent(model_path="models/dqn_robot.pth"):
    env = WarehouseEnv()
    visualizer = WarehouseVisualizer(env)
    agent = DQNAgent(state_dim=3, action_dim=4)
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.q_net.eval()

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                visualizer.quit()
                return

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

        visualizer.draw()

    print(f"Total reward: {total_reward:.2f}")
    pygame.time.delay(1000)
    visualizer.quit()

if __name__ == "__main__":
    test_agent()