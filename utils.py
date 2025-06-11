# utils.py
import torch
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def plot_rewards(rewards, save_path=None):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    if save_path:
        plt.savefig(save_path)
    plt.show()