# test_agent.py
from env import WarehouseEnv
from agent import DQNAgent
import numpy as np

env = WarehouseEnv()
agent = DQNAgent(state_dim=3, action_dim=4)

state = env.reset()
for step in range(20):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    agent.buffer.push(state, action, reward, next_state, done)
    agent.update(batch_size=4)
    state = next_state
    print(f"Step {step}, Action: {action}, Reward: {reward}, Done: {done}")
    if done:
        break

env.render()

