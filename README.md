# warehouse-robot
# 🤖 Warehouse Robot Navigation using Deep Reinforcement Learning

A final project for the Reinforcement Learning course (강화학습), where an agent (robot) learns to **navigate a grid-based warehouse** environment, avoid obstacles, pick up goods, and deliver them efficiently using **Deep Q-Network (DQN)** and its variants: **Double DQN** and **Dueling DQN**.

---

## 📚 Overview

In this project, we simulate a **10×10 warehouse** with optional dynamic obstacles. The agent learns to move from a **start position** to a **pickup point**, and then reach the **delivery location**, optimizing for speed and safety using **deep reinforcement learning**.

- ✅ Custom environment (GridWorld-style)
- ✅ Dynamic rendering using `pygame`
- ✅ Flexible config via `config.yaml`
- ✅ Experiment with classic DQN, Double DQN, and Dueling DQN
- ✅ Visual plots, logs, and trained model export

---

## 🗂️ Project Structure

```bash
.
├── agent.py            # DQN agent with Double/Dueling variants
├── env.py              # Warehouse grid environment
├── train.py            # Training pipeline
├── test.py             # Run trained model
├── utils.py            # Replay buffer, plotters, helpers
├── visualize.py        # Real-time environment rendering (pygame)
├── config.yaml         # All training & environment settings
├── models/             # Saved models
├── outputs/            # Reward plots & logs
├── logs/               # Training logs (CSV)
└── README.md           # This file
```

🧠 Environment Design
| Component | Description |
| -------------- | ---------------------------------------------------- |
| Grid Size | 10 × 10 cells |
| Agent | Red circle — starts at random free cell |
| Pickup Point | Blue square — must reach first |
| Delivery Point | Green square — final goal after pickup |
| Obstacles | Black squares — static or randomly moving (optional) |
| Max Steps | 200 steps per episode (can be changed via config) |

Rendering (via pygame)
🔺 Agent: Red

🔵 Pickup location

🟢 Delivery location

⬛ Obstacles

🔁 State & Action Space
| Element | **State** | 10×10 grid matrix (flattened to 1D) with encoded positions of agent, goals, and obstacles |
| Details | **Action** | Discrete(4): `0=Up`, `1=Down`, `2=Left`, `3=Right`

|
🏆 Reward Function
| Event | Reward |
| -------------------------------- | ------ |
| Move to empty cell | `-0.1` |
| Invalid move / hit wall/obstacle | `-1.0` |
| Reach pickup location | `+10` |
| Deliver item successfully | `+50` |
| Exceed max steps (timeout) | `-10` |

🧠 Agent & Algorithms
Implemented in agent.py with modular support for DQN variants.

✅ Features:
Deep Q-Network (DQN)

Optional: Double DQN

Optional: Dueling DQN

Experience Replay (with ReplayBuffer)

Epsilon-Greedy Exploration

Soft/Hard Target Network Updates

Network Architecture
Input (100,)
→ Linear(100 → 256) → ReLU
→ Linear(256 → 256) → ReLU
→ Linear(256 → 4) → Q-values for 4 actions

⚙️ Configuration: config.yaml
env:
grid_size: [10, 10]
max_steps: 200
dynamic_obstacles: false

agent:
gamma: 0.99
lr: 0.001
buffer_size: 10000
batch_size: 64
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995
double_dqn: true
dueling_dqn: true

train:
episodes: 500
target_update: 10
save_path: "models/dqn_model.pth"
log_path: "logs/train_log.csv"

✅ You can easily switch between vanilla DQN, Double DQN, and Dueling DQN via this file.

🛠️ Installation
Requires Python 3.10+

git clone https://github.com/mdt190100/warehouse-robot.git
cd warehouse-robot-rl
pip install -r requirements.txt

🏋️ Training

python train.py

Training logs will be saved to logs/train_log.csv

Plots and model saved to outputs/ and models/

📈 Results
| Metric | Value (after 500 episodes) |
| ----------------- | -------------------------- |
| Average reward | \~+45 per episode |
| Success rate | \~90% delivery success |
| Steps to complete | < 100 steps on average |

📊 Reward Curve
Automatically saved to outputs/reward_curve.png

🧪 Testing & Visualization
To visualize a trained agent:

python test.py
🔁 You’ll see a dynamic rendering of the warehouse, showing the agent moving from pickup to delivery.

📄 Report
A complete PDF report is available in report.pdf with:

Environment design and motivation

Agent architecture and algorithms

Reward shaping rationale

Experimental setup and results

Discussion & improvements

🔍 Future Improvements
Add moving forklifts as dynamic obstacles

Add multi-agent support

Integrate warehouse layout from real data

Web dashboard for monitoring
